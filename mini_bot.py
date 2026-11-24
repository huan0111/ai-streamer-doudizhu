import argparse
import ctypes
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import ImageGrab

import mini_perceive
from capture_window import POINT, client_bbox, find_window, print_window_image, window_rect
from streamer_events import StreamerEventStream

ROOT = Path(__file__).resolve().parent
LOG_PATH = ROOT / 'outputs' / 'mini_bot.jsonl'
BASE_WIDTH = 933
BASE_HEIGHT = 580


class CaptureError(RuntimeError):
    pass

STATIC_BOXES = {
    'score': (250, 230, 700, 380),
    'claim': (385, 455, 550, 500),
    'confirm': (420, 365, 515, 415),
    'continue_btn': (508, 450, 655, 496),
    'start_btn': (500, 315, 655, 365),
    'quick_start': (680, 495, 900, 562),
}

STATIC_POINTS = {
    'claim': (465, 475),
    'confirm': (465, 390),
    'continue': (578, 472),
    'start': (575, 337),
    'quick_start': (782, 526),
    'lobby_collection': (210, 240),
    'back': (84, 98),
    'friend_cancel': (331, 394),
}

UI_ROI = (230, 298, 740, 372)
BADGE_ROIS = {'left': (96, 266, 140, 310), 'right': (793, 266, 837, 310)}
TUOGUAN_BOX = (340, 400, 595, 535)
TUOGUAN_POINT = (466, 462)
TUOGUAN_TEMPLATE = ROOT / 'templates' / 'mini_ui_tuoguan_cancel.png'
BID_TEMPLATE_DIR = ROOT / 'templates' / 'mini_ui_buttons'
RANK_VALUE = {'3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14, '2': 15, 'LJ': 16, 'BJ': 17}
STRAIGHT_ORDER = ['3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
CARD_NAMES = {'LJ': '小王', 'BJ': '大王'}


def estimate_moves(hand):
    from collections import Counter
    counts = Counter(hand)
    moves = 0
    if counts.get('BJ') and counts.get('LJ'):
        counts['BJ'] -= 1
        counts['LJ'] -= 1
        moves += 1
    for rank in list(counts):
        while counts[rank] >= 4:
            counts[rank] -= 4
            moves += 1
    def longest_run(min_len, pair):
        best = []
        run = []
        for rank in STRAIGHT_ORDER:
            need = 2 if pair else 1
            if counts.get(rank, 0) >= need:
                run.append(rank)
                if len(run) > len(best):
                    best = list(run)
            else:
                run = []
        return best if len(best) >= min_len else []
    while True:
        run = longest_run(5, False)
        if not run:
            break
        for rank in run:
            counts[rank] -= 1
        moves += 1
    while True:
        run = longest_run(3, True)
        if not run:
            break
        for rank in run:
            counts[rank] -= 2
        moves += 1
    for rank in sorted(counts, key=lambda r: RANK_VALUE[r]):
        while counts[rank] >= 3:
            counts[rank] -= 3
            moves += 1
    for rank in list(counts):
        while counts[rank] >= 2:
            counts[rank] -= 2
            moves += 1
    moves += sum(counts.values())
    return moves


def score_lead(candidate, hand):
    from collections import Counter
    ranks = candidate['ranks']
    counts = Counter(hand)
    remaining = list(hand)
    for rank in ranks:
        remaining.remove(rank)
    total = 1 + estimate_moves(remaining)
    high = sum(max(0, RANK_VALUE.get(rank, 0) - 13) for rank in ranks)
    breaks_bomb = any(counts[rank] >= 4 and ranks.count(rank) < 4 for rank in set(ranks))
    breaks_rocket = ('BJ' in counts and 'LJ' in counts) and len({'BJ', 'LJ'} & set(ranks)) == 1
    uses_bomb = any(counts[rank] >= 4 and ranks.count(rank) >= 4 for rank in set(ranks))
    keeps_two = counts.get('2', 0) > 0 and '2' not in ranks
    keeps_joker = ('BJ' in remaining or 'LJ' in remaining)
    score = total * 10.0
    score += 8.0 if breaks_bomb else 0.0
    score += 9.0 if breaks_rocket else 0.0
    if uses_bomb and len(ranks) != len(hand):
        score += 25.0
    score += high * 0.4
    score -= len(ranks) * 0.5
    if keeps_two:
        score -= 0.8
    if keeps_joker:
        score -= 0.6
    if len(ranks) == len(hand):
        score -= 100.0
    return score


def pick_lead(hand, candidates):
    return min(candidates, key=lambda candidate: score_lead(candidate, hand))


def fractional(rgb, box, kind, scale_x, scale_y):
    x0, y0, x1, y1 = box
    x0 = round(x0 * scale_x)
    x1 = round(x1 * scale_x)
    y0 = round(y0 * scale_y)
    y1 = round(y1 * scale_y)
    roi = rgb[y0:y1, x0:x1]
    if roi.size == 0:
        return 0.0
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    if kind == 'orange':
        mask = (hsv[:, :, 0] >= 5) & (hsv[:, :, 0] <= 30) & (hsv[:, :, 1] > 120) & (hsv[:, :, 2] > 150)
    elif kind == 'blue':
        mask = (hsv[:, :, 0] >= 100) & (hsv[:, :, 0] <= 135) & (hsv[:, :, 1] > 80) & (hsv[:, :, 2] > 150)
    elif kind == 'result_blue':
        mask = (hsv[:, :, 0] >= 95) & (hsv[:, :, 0] <= 115) & (hsv[:, :, 1] > 170) & (hsv[:, :, 2] > 180)
    elif kind == 'yellow':
        mask = (hsv[:, :, 0] >= 18) & (hsv[:, :, 0] <= 38) & (hsv[:, :, 1] > 120) & (hsv[:, :, 2] > 190)
    else:
        mask = (hsv[:, :, 1] < 40) & (hsv[:, :, 2] > 200)
    return float(mask.mean())


def detect_result_outcome(rgb, scale_x, scale_y):
    title_box = (330, 130, 610, 250)
    yellow = fractional(rgb, title_box, 'yellow', scale_x, scale_y)
    blue = fractional(rgb, title_box, 'result_blue', scale_x, scale_y)
    if max(yellow, blue) < 0.03:
        return None
    return 'win' if yellow > blue else 'loss'


def find_continue_point(rgb, scale_x, scale_y):
    x0, y0, x1, y1 = 440, 440, 740, 580
    roi = rgb[round(y0 * scale_y):round(y1 * scale_y), round(x0 * scale_x):round(x1 * scale_x)]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    mask = ((hsv[:, :, 0] >= 15) & (hsv[:, :, 0] <= 40) & (hsv[:, :, 1] > 120)
            & (hsv[:, :, 2] > 180)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if width < 60 * scale_x or height < 24 * scale_y or area < 800 * scale_x * scale_y:
            continue
        if best is None or area > best[0]:
            best = (area, (x + width / 2) / scale_x + x0, (y + height / 2) / scale_y + y0)
    if best is None:
        return None
    return [round(best[1]), round(best[2])]


def count_hand_in_frame(rgb):
    try:
        return mini_perceive.detect_hand(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))['count']
    except ValueError:
        return None


def evaluate_play_execution(state, current_state, hand_before, hand_after, expected_remaining):
    if state == 'result' or hand_after == expected_remaining:
        return 'confirmed'
    if state == current_state and hand_after in {None, hand_before}:
        return 'rejected'
    return 'uncertain'


def off_by_one_target(played):
    for seat in ('left', 'right'):
        info = played.get(seat) or {}
        ranks = info.get('ranks') or []
        if ranks and not info.get('reliable', True) and info.get('count_est') == len(ranks) + 1:
            return seat, list(ranks)
    return None, None


def counts_for_target(counts, seat, target, from_detection):
    result = dict(counts)
    if from_detection and seat in result:
        result[seat] = max(0, result[seat] - len(target))
    return result


def find_ui(rgb, scale_x, scale_y):
    x0, y0, x1, y1 = UI_ROI
    roi = rgb[round(y0 * scale_y):round(y1 * scale_y), round(x0 * scale_x):round(x1 * scale_x)]
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    mask = ((hsv[:, :, 0] >= 3) & (hsv[:, :, 0] <= 32) & (hsv[:, :, 1] > 120) & (hsv[:, :, 2] > 150)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found = {'orange': [], 'clock': [], 'blue': []}
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if cv2.contourArea(contour) < 1200 or height < 28 or height > 75:
            continue
        ratio = width / max(height, 1)
        entry = {
            'bbox': [x, y, width, height],
            'center': [round((x + width / 2) / scale_x + x0), round((y + height / 2) / scale_y + y0)],
        }
        if ratio >= 1.8 and width >= 70:
            found['orange'].append(entry)
        elif 0.6 <= ratio <= 1.5 and 32 <= width <= 72:
            center = entry['center']
            if 340 <= center[0] <= 540 and 305 <= center[1] <= 365:
                found['clock'].append(entry)
    blue_mask = ((hsv[:, :, 0] >= 95) & (hsv[:, :, 0] <= 125) & (hsv[:, :, 1] > 80) & (hsv[:, :, 2] > 150)).astype(np.uint8) * 255
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in blue_contours:
        x, y, width, height = cv2.boundingRect(contour)
        if cv2.contourArea(contour) < 1200 or height < 28 or height > 75:
            continue
        ratio = width / max(height, 1)
        if ratio >= 1.8 and width >= 70:
            found['blue'].append({
                'bbox': [x, y, width, height],
                'center': [round((x + width / 2) / scale_x + x0), round((y + height / 2) / scale_y + y0)],
            })
    for key in found:
        found[key].sort(key=lambda item: item['center'][0])
    return found


def classify(rgb, scale_x, scale_y):
    values = static_probe(rgb, scale_x, scale_y)
    if values['score'] > 0.5 and values['claim'] > 0.4:
        return 'reward', values, None
    if values['confirm'] > 0.45 and values['score'] > 0.5:
        return 'warning_dialog', values, None
    if values['score'] > 0.5 and values['continue'] > 0.3:
        return 'result', values, None
    if detect_result_outcome(rgb, scale_x, scale_y):
        return 'result_candidate', values, None
    ui = find_ui(rgb, scale_x, scale_y)
    if ui['clock'] and not ui['orange']:
        return 'pass_only', values, ui
    if ui['clock'] and ui['orange']:
        x = ui['orange'][0]['center'][0]
        if x > 600:
            return 'turn', values, ui
        if x >= 540:
            return 'lead', values, ui
        if x >= 450:
            return 'lead', values, ui
        return 'bid', values, ui
    if values['quick_start'] > 0.4:
        return 'room_select', values, ui
    if values['start'] > 0.5:
        return 'room', values, ui
    return 'idle', values, ui


def static_probe(rgb, scale_x, scale_y):
    return {
        'score': fractional(rgb, STATIC_BOXES['score'], 'white', scale_x, scale_y),
        'claim': fractional(rgb, STATIC_BOXES['claim'], 'orange', scale_x, scale_y),
        'confirm': fractional(rgb, STATIC_BOXES['confirm'], 'orange', scale_x, scale_y),
        'continue': fractional(rgb, STATIC_BOXES['continue_btn'], 'yellow', scale_x, scale_y),
        'start': fractional(rgb, STATIC_BOXES['start_btn'], 'orange', scale_x, scale_y),
        'quick_start': fractional(rgb, STATIC_BOXES['quick_start'], 'orange', scale_x, scale_y),
    }


def compact_scene_text(text):
    return ''.join(str(text or '').split())


def confirm_result_scene(text):
    text = compact_scene_text(text)
    scoreboard = '底分' in text and '倍数' in text and any(word in text for word in ('欢乐豆', '继续游戏'))
    partial_ocr = any(word in text for word in ('继续游戏', '继游戏')) and any(word in text for word in ('底分', '倍数', '欢乐豆'))
    return scoreboard or partial_ocr


def dialog_scene_plan(text):
    text = compact_scene_text(text)
    plan = {
        'action': 'dismiss_dialog',
        'utterance': '有个提示，我先确认一下。',
        'reason': '关闭游戏提示弹窗',
        'point': STATIC_POINTS['confirm'],
        'phase': 'dialog',
    }
    if '好友信息' in text and ('申请' in text or '授权' in text or '好友列表' in text):
        plan.update({
            'action': 'deny_friend_authorization',
            'utterance': '这是微信好友信息授权，点取消，不授权。',
            'reason': '拒绝微信好友信息授权',
            'point': STATIC_POINTS['friend_cancel'],
            'phase': 'dialog',
        })
        return plan
    if '视频' in text and any(word in text for word in ('欢乐豆', '返豆', '返还', '领', '暴击')):
        plan.update({
            'action': 'skip_video_beans',
            'utterance': '这是看视频返豆的按钮，用户不让点，我不点，停在这里等老师处理。',
            'reason': '免费视频返豆已禁用，不点击',
            'phase': 'beans',
            'blocked': True,
            'wait': True,
        })
        plan.pop('point', None)
        return plan
    if '离开房间' in text:
        plan.update({
            'action': 'acknowledge_kick',
            'utterance': '因长时间未操作被请出房间，确认提示后重新进入游戏。',
            'reason': '太久未操作，被请出房间',
        })
        return plan
    if '欢乐豆' in text and any(word in text for word in ('不足', '不够', '补充', '购买', '破产')):
        if '免费' in text and any(word in text for word in ('领取', '补豆', '救济')):
            plan.update({
                'action': 'claim_free_beans',
                'utterance': '欢乐豆不够了，这里有免费补充，我先领一下。',
                'reason': '领取免费欢乐豆',
                'phase': 'beans',
            })
        elif any(word in text for word in ('购买', '充值', '支付', '人民币')):
            plan.update({
                'action': 'purchase_required',
                'utterance': '欢乐豆不够了，这里需要购买或充值，我先停下来，不自动付款。',
                'reason': '补豆涉及购买或充值，需要用户处理',
                'phase': 'beans',
                'blocked': True,
            })
        else:
            plan.update({
                'action': 'dismiss_bean_notice',
                'utterance': '欢乐豆不够了，我先看看怎么补充。',
                'reason': '关闭欢乐豆不足提示，继续查看补豆入口',
                'phase': 'beans',
            })
    return plan


def idle_scene_plan(text):
    text = compact_scene_text(text)
    if '正在加载' in text:
        return {
            'action': 'wait_loading', 'utterance': '资源还在加载，我等它一下。',
            'reason': '等待游戏资源加载', 'phase': 'lobby', 'wait': True,
        }
    if '好友信息' in text and ('申请' in text or '授权' in text or '好友列表' in text):
        return {
            'action': 'deny_friend_authorization', 'utterance': '这是微信好友信息授权，点取消，不授权。',
            'reason': '拒绝微信好友信息授权', 'phase': 'dialog', 'point': STATIC_POINTS['friend_cancel'],
        }
    if '好友房' in text and ('开设房间' in text or '加入房间' in text):
        return {
            'action': 'leave_friend_room', 'utterance': '点到好友房了，先返回大厅。',
            'reason': '好友房不是当前目标，返回大厅', 'phase': 'lobby', 'point': STATIC_POINTS['back'],
        }
    if '欢乐豆' in text and any(word in text for word in ('购买', '充值', '支付', '人民币')):
        return {
            'action': 'purchase_required', 'utterance': '这里是购买或充值页面，我先停下，不自动付款。',
            'reason': '购买或充值需要用户确认', 'phase': 'beans', 'blocked': True,
        }
    if '快速开始' in text and '新手场' in text:
        return {
            'action': 'quick_start', 'utterance': '先进新手场，快速开始。',
            'reason': '选场页快速开始新手场', 'phase': 'lobby', 'point': STATIC_POINTS['quick_start'],
        }
    if '斗地主合集' in text or ('斗地' in text and '合' in text and any(word in text for word in ('百变', '斗仙牌', '趣玩'))):
        return {
            'action': 'open_collection', 'utterance': '先去斗地主合集，找个新手场。',
            'reason': '从主大厅进入斗地主合集', 'phase': 'lobby', 'point': STATIC_POINTS['lobby_collection'],
        }
    if '新手场' in text and ('普通场' in text or '高级场' in text):
        return {
            'action': 'select_novice_room', 'utterance': '进入新手场。',
            'reason': '选场页选择新手场', 'phase': 'lobby',
        }
    return {
        'action': 'unknown_scene', 'utterance': '这个页面我暂时不会操作，先不乱点，找老师问问怎么办。',
        'reason': '未识别的大厅或弹窗页面', 'phase': 'lobby', 'wait': True,
    }


def room_select_point(words, scale_x, scale_y):
    words = [word for word in words or [] if str(word.get('text') or '').strip()]
    combined = [word for word in words if '新手场' in str(word['text'])]
    anchors = [word for word in words if str(word['text']).strip() == '新']
    for anchor in anchors:
        group = [anchor]
        for word in words:
            text = str(word['text']).strip()
            if text not in {'手', '场'}:
                continue
            if abs(word['y'] - anchor['y']) <= anchor['h'] and 0 <= word['x'] - anchor['x'] <= anchor['w'] * 4:
                group.append(word)
        if len(group) >= 3:
            combined.append({
                'x': min(item['x'] for item in group),
                'y': min(item['y'] for item in group),
                'w': max(item['x'] + item['w'] for item in group) - min(item['x'] for item in group),
                'h': max(item['y'] + item['h'] for item in group) - min(item['y'] for item in group),
            })
            break
    if not combined:
        return None
    box = combined[0]
    return [round((box['x'] + box['w'] / 2) / scale_x), round((box['y'] + box['h'] / 2) / scale_y)]


def follow_shortlist(result):
    candidates = list(result.get('candidates', []))
    non_pass = [item for item in candidates if item.get('id') != 'pass']
    if non_pass and result.get('recommended') != 'pass':
        return non_pass
    shortlist = candidates if candidates else non_pass
    if result.get('can_pass', True) and not any(item.get('id') == 'pass' for item in shortlist):
        shortlist = shortlist + [{
            'id': 'pass', 'type': 'pass', 'ranks': [], 'indices': [],
            'reasons': ['只能过' if not non_pass else '放弃这一手'], 'local_score': 50,
        }]
    return shortlist


class Bot:
    def __init__(self, method='message', interval=0.2, hand_limit=0, overlay=None):
        LOG_PATH.parent.mkdir(exist_ok=True)
        self.hand_limit = hand_limit
        self.hands_done = 0
        self.overlay = overlay
        self.events = StreamerEventStream()
        self.hand_id = None
        self.method = method
        self.interval = interval
        self.user32 = ctypes.WinDLL('user32', use_last_error=True)
        self.user32.PostMessageW.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p]
        self.user32.PostMessageW.restype = ctypes.c_bool
        self.last_action = 0.0
        self.state = None
        self.state_since = 0.0
        self.lead_key = None
        self.lead_repeat = 0
        self.role = None
        self.last_hand_count = None
        self.last_hand_probe = 0.0
        self.last_count_probe = 0.0
        self.last_client_size = None
        self.last_capture_geometry = None
        self.last_overlay_rect = None
        self.small_client_warned = False
        self.lead_cache_key = None
        self.lead_cache = None
        self.follow_cache_key = None
        self.follow_cache = None
        self.bid_done = False
        self.bid_complete = False
        self.bid_last = None
        self.bid_clicks = {}
        self.bid_click_keys = {}
        self.bid_button_cache = None
        self.hand_active = False
        self.landlord_seat = None
        self.landlord_count_adjusted = False
        self.counts = {'left': 17, 'right': 17}
        self.has_played = {'left': False, 'right': False}
        self.badge_at = {'left': 0.0, 'right': 0.0}
        self.play_at = {'left': 0.0, 'right': 0.0}
        self.badge_counts = {'left': None, 'right': None}
        self.badge_last_read = {'left': None, 'right': None}
        self.played_total = {'left': 0, 'right': 0}
        self.played = []
        self.play_candidate = {'left': (), 'right': ()}
        self.play_candidate_at = {'left': 0.0, 'right': 0.0}
        self.play_state = {'left': (), 'right': ()}
        self.play_unreliable = {'left': (), 'right': ()}
        self.play_unreliable_candidate = {'left': (), 'right': ()}
        self.unreliable_since = {'left': 0.0, 'right': 0.0}
        self.last_play = None
        self.last_play_probe = 0.0
        self.last_scene_ocr = 0.0
        self.last_idle_notice = None
        self.blocked_scene = None
        self.scene_action_times = {}
        self.lead_blocked_key = None
        self.lead_failure_counts = {}
        self.paused = False
        self.hotkey_down = False
        self.follow_retry_key = None
        self._tuoguan_missing_logged = False
        self._capture_warn = 0.0
        self.selection_trace = False
        self.selection_batch = 3
        self.selection_batches = 3
        self.last_selection_check = None
        if self.overlay is not None:
            self.overlay.on_takeover = self.toggle_pause

    def tuoguan_score(self, rgb, scale_x, scale_y):
        template = getattr(self, '_tuoguan_template', None)
        if template is None:
            if TUOGUAN_TEMPLATE.exists():
                data = np.fromfile(str(TUOGUAN_TEMPLATE), dtype=np.uint8)
                template = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if template is not None:
                    template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
            self._tuoguan_template = template if template is not None else False
        if template is None or template is False:
            if not self._tuoguan_missing_logged:
                self._tuoguan_missing_logged = True
                self.log({'state': 'tuoguan_template_missing', 'path': str(TUOGUAN_TEMPLATE)})
                self.notify('【异常】找不到取消托管模板，托管检测已停用（不会误点桌子中央）', 'error')
            return 0.0
        scale = min(scale_x, scale_y)
        if scale != 1.0:
            template = cv2.resize(
                template, (round(template.shape[1] * scale), round(template.shape[0] * scale)),
                interpolation=cv2.INTER_AREA,
            )
        x0, y0, x1, y1 = TUOGUAN_BOX
        roi = rgb[round(y0 * scale_y):round(y1 * scale_y), round(x0 * scale_x):round(x1 * scale_x)]
        if roi.shape[0] < template.shape[0] or roi.shape[1] < template.shape[1]:
            return 0.0
        result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
        return float(result.max())

    def window(self):
        try:
            _, (hwnd, title, _) = find_window('腾讯欢乐斗地主', 'Chrome_WidgetWin_0')
        except RuntimeError:
            return None
        return hwnd

    def toggle_pause(self):
        self.paused = not self.paused
        if self.overlay is not None:
            self.overlay.set_paused(self.paused)
        action = 'pause' if self.paused else 'resume'
        utterance = '已暂停自动操作，由人工接管。' if self.paused else '人工接管结束，恢复自动操作。'
        self.write_decision_event({
            'phase': 'control', 'action': action, 'role': None, 'hand': [], 'target': None,
            'chosen': action, 'chosen_ranks': [], 'reason': '用户主动切换操作权',
            'utterance': utterance, 'source': 'user_control', 'execution': 'applied',
        })
        self.log({'state': 'manual_takeover' if self.paused else 'manual_resume'})

    def check_takeover_hotkey(self):
        down = all(self.user32.GetAsyncKeyState(key) & 0x8000 for key in (0x11, 0x12, 0x50))
        if down and not self.hotkey_down:
            self.toggle_pause()
        self.hotkey_down = down

    def ensure_foreground(self, hwnd):
        if self.paused:
            return False
        if self.user32.GetForegroundWindow() == hwnd:
            return True
        if self.user32.IsIconic(hwnd):
            self.user32.ShowWindow(hwnd, 9)
            time.sleep(0.1)
        self.user32.SetForegroundWindow(hwnd)
        time.sleep(0.15)
        if self.user32.GetForegroundWindow() != hwnd:
            import pydirectinput
            pydirectinput.press('alt')
            self.user32.SetForegroundWindow(hwnd)
            time.sleep(0.15)
        return self.user32.GetForegroundWindow() == hwnd

    def capture_record(self, hwnd, bbox, image_size, method, attempt):
        try:
            rect = window_rect(self.user32, hwnd)
            outer = [rect.left, rect.top, rect.right, rect.bottom]
        except Exception:
            outer = None
        try:
            dpi = int(self.user32.GetDpiForWindow(hwnd))
        except (AttributeError, OSError):
            dpi = None
        return {
            'client_bbox': list(bbox),
            'client_size': [bbox[2] - bbox[0], bbox[3] - bbox[1]],
            'image_size': list(image_size) if image_size is not None else None,
            'window_rect': outer,
            'dpi': dpi,
            'capture_method': method,
            'attempt': attempt,
        }

    def capture(self, hwnd):
        reason = 'capture_unavailable'
        record = None
        for attempt in range(1, 3):
            bbox = client_bbox(self.user32, hwnd)
            expected = (bbox[2] - bbox[0], bbox[3] - bbox[1])
            image = print_window_image(hwnd, *expected) if min(expected) > 0 else None
            method = 'PrintWindow'
            if image is not None:
                rgb = np.array(image)
                record = self.capture_record(hwnd, bbox, image.size, method, attempt)
                if image.size == expected and rgb.shape[:2] == (expected[1], expected[0]):
                    if np.any(rgb):
                        geometry = (tuple(bbox), image.size, record['window_rect'], record['dpi'], method)
                        if geometry != self.last_capture_geometry:
                            self.last_capture_geometry = geometry
                            self.log({'state': 'client_size', 'size': list(expected), **record,
                                      'rgb_shape': list(rgb.shape)})
                        return rgb, bbox
                    reason = 'capture_black_frame'
                else:
                    reason = 'capture_size_mismatch'
                self.log({'state': reason, **record, 'rgb_shape': list(rgb.shape)})
            if self.user32.GetForegroundWindow() == hwnd:
                method = 'ImageGrab'
                try:
                    image = ImageGrab.grab(bbox=bbox, all_screens=True)
                except Exception as error:
                    image = None
                    reason = 'imagegrab_failed'
                    record = self.capture_record(hwnd, bbox, None, method, attempt)
                    self.log({'state': reason, **record, 'error': str(error)})
                if image is not None:
                    rgb = np.array(image)
                    record = self.capture_record(hwnd, bbox, image.size, method, attempt)
                    if image.size == expected and rgb.shape[:2] == (expected[1], expected[0]):
                        if np.any(rgb):
                            geometry = (tuple(bbox), image.size, record['window_rect'], record['dpi'], method)
                            if geometry != self.last_capture_geometry:
                                self.last_capture_geometry = geometry
                                self.log({'state': 'client_size', 'size': list(expected), **record,
                                          'rgb_shape': list(rgb.shape)})
                            return rgb, bbox
                        reason = 'capture_black_frame'
                    else:
                        reason = 'capture_size_mismatch'
                    self.log({'state': reason, **record, 'rgb_shape': list(rgb.shape)})
            elif image is None:
                reason = 'printwindow_failed_and_not_foreground'
                record = self.capture_record(hwnd, bbox, None, method, attempt)
            if attempt == 1:
                time.sleep(0.05)
        self.log({'state': 'capture_failed', 'reason': reason, **(record or {})})
        raise CaptureError(reason)

    def sync_overlay(self, hwnd):
        if self.overlay is None:
            return
        rect = window_rect(self.user32, hwnd)
        game_rect = (rect.left, rect.top, rect.right, rect.bottom)
        if game_rect != self.last_overlay_rect:
            self.overlay.place_outside(game_rect)
            self.last_overlay_rect = game_rect

    def click(self, hwnd, bbox, point, scale_x, scale_y, physical=None):
        if self.paused:
            self.log({'state': 'click_skipped', 'reason': 'manual_takeover', 'point': list(point)})
            return False
        if hwnd is not None and bbox is not None:
            try:
                live_bbox = client_bbox(self.user32, hwnd)
            except Exception as error:
                self.log({'state': 'click_skipped', 'reason': 'window_lookup_failed', 'error': str(error)})
                return False
            captured_size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
            live_size = (live_bbox[2] - live_bbox[0], live_bbox[3] - live_bbox[1])
            if live_size != captured_size:
                self.log({'state': 'click_skipped', 'reason': 'window_resized',
                          'captured_size': list(captured_size), 'live_size': list(live_size)})
                return False
            if tuple(live_bbox) != tuple(bbox):
                self.log({'state': 'window_moved', 'captured': list(bbox), 'live': list(live_bbox)})
            bbox = live_bbox
        x = round(point[0] * scale_x)
        y = round(point[1] * scale_y)
        use_physical = self.method == 'physical' if physical is None else physical
        if use_physical:
            import pydirectinput
            if self.user32.GetForegroundWindow() != hwnd and not self.ensure_foreground(hwnd):
                self.log({'state': 'click_skipped', 'reason': 'foreground', 'point': [x, y]})
                return False
            pydirectinput.PAUSE = 0.05
            self.user32.SetCursorPos(bbox[0] + x, bbox[1] + y)
            time.sleep(0.02)
            actual = POINT()
            self.user32.GetCursorPos(ctypes.byref(actual))
            expected = [bbox[0] + x, bbox[1] + y]
            if [actual.x, actual.y] != expected:
                self.log({'state': 'cursor_mismatch', 'expected': expected, 'actual': [actual.x, actual.y]})
            if max(abs(actual.x - expected[0]), abs(actual.y - expected[1])) > 6:
                self.user32.SetCursorPos(*expected)
                time.sleep(0.03)
                actual = POINT()
                self.user32.GetCursorPos(ctypes.byref(actual))
                if max(abs(actual.x - expected[0]), abs(actual.y - expected[1])) > 6:
                    self.log({'state': 'click_blocked', 'reason': 'cursor_not_at_target', 'expected': expected,
                              'actual': [actual.x, actual.y]})
                    return False
            if self.user32.GetForegroundWindow() != hwnd and not self.ensure_foreground(hwnd):
                self.log({'state': 'click_blocked', 'reason': 'foreground_changed', 'point': expected})
                return False

            def press():
                pydirectinput.mouseDown()
                time.sleep(0.05)
                pydirectinput.mouseUp()

            try:
                press()
            except Exception as error:
                self.log({'state': 'click_retry', 'error': str(error), 'point': [x, y]})
                self.user32.SetCursorPos(bbox[0] + x, bbox[1] + y)
                time.sleep(0.05)
                press()
            foreground_after = self.user32.GetForegroundWindow()
            if foreground_after != hwnd:
                self.log({'state': 'foreground_changed_after_click', 'expected_hwnd': hwnd,
                          'actual_hwnd': foreground_after, 'point': expected})
        else:
            packed = (y << 16) | (x & 0xFFFF)
            self.user32.PostMessageW(hwnd, 0x0200, 0, packed)
            self.user32.PostMessageW(hwnd, 0x0201, 1, packed)
            time.sleep(0.03)
            self.user32.PostMessageW(hwnd, 0x0202, 0, packed)
        self.last_hand_count = None
        return True

    def notify(self, text, tag=None):
        if self.overlay is not None:
            self.overlay.push(text, tag)

    def log(self, record):
        record['timestamp'] = datetime.now().astimezone().isoformat()
        with LOG_PATH.open('a', encoding='utf-8') as log:
            log.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(json.dumps(record, ensure_ascii=True), flush=True)

    def save_action_frame(self, rgb, prefix):
        stamp = datetime.now().strftime('%H%M%S_%f')
        path = ROOT / 'outputs' / f'{prefix}_{stamp}.png'
        mini_perceive.save_image(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        self.log({'state': 'action_frame_saved', 'kind': prefix, 'frame': str(path)})
        return path

    def card_point(self, detection, index, scale_x=1.0, scale_y=1.0):
        lefts = detection.get('card_lefts') or []
        glyphs = detection['glyphs']
        if index >= len(lefts) or index >= len(glyphs):
            return None
        pitch = detection.get('pitch') or 46 * scale_x
        return [(lefts[index] + pitch * 0.55) / scale_x, (glyphs[index][1] + 55 * scale_y) / scale_y]

    def set_selection(self, hwnd, bbox, detection, indices, scale_x, scale_y, limit=None):
        current = set(detection['selected_indices'])
        target = set(indices)
        operations = [(index, 'deselect') for index in sorted(current - target)]
        operations += [(index, 'select') for index in sorted(target - current)]
        if limit is not None and len(operations) > limit:
            self.log({'state': 'selection_batch', 'planned': len(operations), 'limit': limit,
                      'deferred': len(operations) - limit})
            operations = operations[:limit]
        for index, operation in operations:
            if index >= len(detection['glyphs']):
                self.log({'state': 'selection_skip', 'index': index, 'count': len(detection['glyphs'])})
                return False
            point = self.card_point(detection, index, scale_x, scale_y)
            if point is None:
                return False
            self.log({
                'state': 'selection_click', 'operation': operation, 'index': index,
                'point': [round(value, 2) for value in point],
                'screen_point': [round(bbox[0] + point[0] * scale_x), round(bbox[1] + point[1] * scale_y)],
                'window_size': [bbox[2] - bbox[0], bbox[3] - bbox[1]],
                'scale': [scale_x, scale_y],
                'input_method': 'physical',
            })
            if self.click(hwnd, bbox, point, scale_x, scale_y, physical=True) is False:
                return False
            time.sleep(0.24)
            if self.selection_trace:
                trace_rgb, _ = self.capture(hwnd)
                trace_check = self.selection_check(trace_rgb, sorted(target))
                if trace_check:
                    self.log({'state': 'selection_click_state', 'after': index, 'operation': operation,
                              'selected': trace_check['selected']})
        return True

    def selection_check(self, rgb, chosen):
        try:
            observation = mini_perceive.read_hand(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        except ValueError:
            return None
        selected = set(observation['selected_indices'])
        return {
            'observation': observation,
            'selected': sorted(selected),
            'missing': [index for index in chosen if index not in selected],
            'extra': [index for index in selected if index not in chosen],
        }

    def apply_lift(self, check, before, chosen, scale_y):
        if not check or not before:
            return check
        selected = set(check['selected']) | set(before.get('selected_indices', []))
        before_glyphs = before.get('glyphs') or []
        after_glyphs = check['observation'].get('glyphs') or []
        for index in range(min(len(before_glyphs), len(after_glyphs))):
            if after_glyphs[index][1] <= before_glyphs[index][1] - 5 * scale_y:
                selected.add(index)
            elif after_glyphs[index][1] >= before_glyphs[index][1] + 5 * scale_y:
                selected.discard(index)
        check['selected'] = sorted(selected)
        check['observation']['selected_indices'] = list(check['selected'])
        check['missing'] = [index for index in chosen if index not in selected]
        check['extra'] = [index for index in selected if index not in chosen]
        return check

    def sample_selection(self, hwnd, chosen, seconds=3.0, step=0.3):
        start = time.perf_counter()
        while time.perf_counter() - start < seconds:
            time.sleep(step)
            rgb, _ = self.capture(hwnd)
            check = self.selection_check(rgb, chosen)
            if not check:
                continue
            self.log({'state': 'selection_trace', 'elapsed': round(time.perf_counter() - start, 2),
                      'selected': check['selected'], 'missing': check['missing'], 'extra': check['extra']})

    def read_selection(self, hwnd, rgb, chosen, before, scale_y):
        check = self.apply_lift(self.selection_check(rgb, chosen), before, chosen, scale_y)
        expected = len((before or {}).get('glyphs') or [])
        if check and expected and check['observation']['count'] != expected:
            counts = [check['observation']['count']]
            retry_rgb = rgb
            for _ in range(2):
                time.sleep(0.35)
                retry_rgb, _ = self.capture(hwnd)
                retry = self.apply_lift(self.selection_check(retry_rgb, chosen), before, chosen, scale_y)
                counts.append((retry or {}).get('observation', {}).get('count'))
                if retry and retry['observation']['count'] == expected:
                    return retry, retry_rgb
            self.log({'state': 'selection_read_unstable', 'expected': expected, 'counts': counts})
            self.save_action_frame(retry_rgb, 'selection_unstable')
            return None, retry_rgb
        return check, rgb

    def confirm_selection(self, hwnd, bbox, rgb, chosen, scale_x, scale_y, before=None):
        if self.selection_trace:
            self.sample_selection(hwnd, chosen)
            rgb, _ = self.capture(hwnd)
        check, rgb = self.read_selection(hwnd, rgb, chosen, before, scale_y)
        self.last_selection_check = check
        attempts = 0
        needed = (len(chosen) + self.selection_batch - 1) // self.selection_batch
        rounds = max(self.selection_batches, needed)
        source = before
        while attempts < rounds - 1 and (check is None or check['missing'] or check['extra']):
            attempts += 1
            if check is None:
                self.log({'state': 'selection_read_retry', 'attempt': attempts, 'expected': list(chosen)})
            else:
                self.log({'state': 'selection_mismatch', 'attempt': attempts, 'expected': list(chosen),
                          'selected': check['selected'], 'missing': check['missing'], 'extra': check['extra']})
                source = check['observation']
            if source is None or self.set_selection(hwnd, bbox, source, chosen, scale_x, scale_y,
                                                   limit=self.selection_batch) is False:
                self.log({'state': 'selection_correction_failed', 'expected': list(chosen), 'attempt': attempts})
                return False, rgb
            time.sleep(0.8)
            rgb, _ = self.capture(hwnd)
            check, rgb = self.read_selection(hwnd, rgb, chosen, source, scale_y)
            self.last_selection_check = check
        if check is None:
            self.log({'state': 'selection_unconfirmed', 'expected': list(chosen), 'reason': 'read_unstable',
                      'attempts': attempts})
            return False, rgb
        if check['missing'] or check['extra']:
            self.log({'state': 'selection_unconfirmed', 'expected': list(chosen), 'selected': check['selected'],
                      'missing': check['missing'], 'extra': check['extra'], 'attempts': attempts})
            return False, rgb
        if attempts:
            self.log({'state': 'selection_corrected', 'expected': list(chosen), 'attempts': attempts})
        return True, rgb

    def pick_lead_selection_fallback(self, rules_hand, index_map):
        try:
            result = self.bot_decision('playing', rules_hand, role=self.role, can_pass=False, mode='candidates')
        except Exception as error:
            self.log({'state': 'lead_selection_fallback_failed', 'error': str(error)})
            return None
        options = [item for item in result.get('candidates', []) if item['id'] != 'pass']
        if not options:
            return None
        candidate = next((item for item in options if item.get('type') == 'single'), None)
        if candidate is None:
            candidate = min(options, key=lambda item: len(item.get('ranks') or []))
        chosen = [index_map[position] for position in candidate['indices']]
        return {
            'chosen': chosen,
            'candidate': {**candidate, 'id': 'selection_fallback'},
            'decision': {
                'reason': '原选牌点击未全部生效，改出更简单的一手',
                'utterance': f'原选牌没点上，改出{" ".join(candidate.get("ranks") or [])}。',
            },
        }

    def pick_safe_single(self, known):
        from collections import Counter
        if not known:
            return None
        counts = Counter(rank for _, rank in known)
        singles = [(index, rank) for index, rank in known if counts[rank] == 1]
        return min(singles or known, key=lambda item: RANK_VALUE[item[1]])

    def pick_local(self, hand, candidates):
        from collections import Counter
        counts = Counter(hand)

        def score(candidate):
            move = candidate['move']
            kind = move.get('type')
            ranks = candidate['ranks']
            breaks_quad = any(counts[rank] >= 4 and ranks.count(rank) < 4 for rank in set(ranks))
            return (1 if kind in {'bomb', 'rocket'} else 0, 1 if breaks_quad else 0, -len(ranks), move.get('mainValue', 0))

        return min(candidates, key=score)

    def bot_decision(self, phase, hand, role='farmer', target=None, target_seat=None, can_pass=False, counts=None, mode=None):
        hand_counts = dict(counts or self.counts)
        request = {
            'phase': phase,
            'difficulty': 'hell',
            'hand': hand,
            'role': role,
            'target': target or [],
            'targetSeat': target_seat,
            'canPass': can_pass,
            'handCounts': {**hand_counts, 'leftLandlord': self.landlord_seat == 'left',
                           'rightLandlord': self.landlord_seat == 'right'},
            'played': self.clean_played(),
            'mode': mode,
            'candidateLimit': 6,
        }
        (ROOT / 'outputs' / 'bot_input.json').write_text(json.dumps(request, ensure_ascii=False), encoding='utf-8')
        subprocess.run(
            ['node', 'bot_cli.mjs', 'outputs/bot_input.json', 'outputs/bot_output.json'],
            cwd=str(ROOT), check=True, capture_output=True, timeout=10,
        )
        return json.loads((ROOT / 'outputs' / 'bot_output.json').read_text(encoding='utf-8'))

    def fixed_role_context(self, role):
        if role == 'landlord':
            landlord_seat = 'self'
            seat_roles = {'self': 'landlord', 'left': 'farmer', 'right': 'farmer'}
        elif self.landlord_seat in {'left', 'right'}:
            landlord_seat = self.landlord_seat
            other = 'right' if landlord_seat == 'left' else 'left'
            seat_roles = {'self': role, landlord_seat: 'landlord', other: 'farmer'}
        else:
            landlord_seat = 'unknown'
            seat_roles = {'self': role, 'left': 'unknown', 'right': 'unknown'}
        return {
            'role': role,
            'landlord_seat': landlord_seat,
            'seat_roles': seat_roles,
            'role_locked': role in {'landlord', 'farmer'},
            'identity_rule': '本局身份开局确认后锁定，每轮沿用；不得根据当前画面重新解释身份。',
        }

    def deepseek_pick(self, hand, role, candidates, target_info=None, counts=None, recommended=None):
        from model_decision import load_api_key, parse_content, post_chat
        config = json.loads((ROOT / 'api_config.json').read_text(encoding='utf-8'))
        config['timeout_seconds'] = 4
        compact = [{
            'id': candidate['id'],
            'type': candidate.get('type'),
            'label': candidate.get('label'),
            'ranks': candidate['ranks'],
            'size': candidate.get('size'),
            'moves_after': candidate.get('moves_after'),
            'wins': candidate.get('wins'),
            'unanswerable': candidate.get('unanswerable'),
            'uses_bomb': candidate.get('uses_bomb'),
            'uses_high': candidate.get('uses_high'),
            'breaks_bomb': candidate.get('breaks_bomb'),
            'local_score': round(candidate.get('local_score', 0), 2),
            'reasons': candidate.get('reasons', []),
        } for candidate in candidates]
        role_context = self.fixed_role_context(role)
        state = {
            'game': '欢乐斗地主',
            **role_context,
            'my_hand': hand,
            'hand_size': len(hand),
            'card_names': CARD_NAMES,
            'last_play': target_info,
            'remaining_counts': counts or {},
            'opponents': {
                'left': {'count': (counts or {}).get('left'), 'role': role_context['seat_roles']['left']},
                'right': {'count': (counts or {}).get('right'), 'role': role_context['seat_roles']['right']},
            },
            'candidates': compact,
            'local_recommendation': recommended,
            'instruction': (
                f'这是本地引擎筛出的 {len(candidates)} 个合法候选（local_score 越小越优）。'
                f'{recommended or "无"} 是本地引擎的正式推荐，**必须优先选择它**。'
                '只有当你判断它会在当前局面造成明显更差的后果（例如直接输掉这一局）时，才允许改选其他候选，并在 reason 里写清楚为什么推荐不合适。'
                '局面信息见上面字段：my_hand、role、last_play、remaining_counts、candidates（含评分和理由）。'
                '请从 candidates 里选一个，给出简短理由和 1–2 句与所选牌一致的中性口播素材；只能选 candidates 里的 id。'
                '素材要包含最终动作/牌型、压制的目标（如有）以及一个核心理由，与 reason 一致；不要念候选 ID 或内部评分。'
                '不使用语气词、口头禅、人设称呼或情绪表演。'
                '输出 JSON：{"candidate_id":"c002","reason":"简短中文原因","utterance":"包含动作、目标和核心理由的中性口播素材"}。不要输出 Markdown。'
            ),
        }
        messages = [
            {'role': 'system', 'content': '你是斗地主决策器与中性解说素材生成器。只从给定候选里选一个，只输出一个 JSON 对象。口播必须与所选牌和理由一致，不要编造牌局信息，不使用语气词、口头禅或固定人设。'},
            {'role': 'user', 'content': json.dumps(state, ensure_ascii=False)},
        ]
        response, _, _ = post_chat(config, load_api_key(config), messages)
        decision, _ = parse_content(response)
        candidate_id = decision.get('candidate_id')
        if not any(candidate['id'] == candidate_id for candidate in candidates):
            raise ValueError(f'invalid candidate from deepseek: {candidate_id}')
        return {
            'candidate_id': candidate_id,
            'reason': str(decision.get('reason') or ''),
            'utterance': str(decision.get('utterance') or ''),
        }

    def write_decision_event(self, event):
        if self.hand_id is None and event.get('phase') in {'lead', 'follow', 'bid', 'double'}:
            self.hand_id = f'{self.events.session_id}-hand-{self.hands_done + 1}'
        record = self.events.publish({'hand_id': self.hand_id, **event})
        if self.overlay is not None and record['speak']:
            self.overlay.show_speech(record['utterance'], record.get('event_kind'))
        return record

    def write_execution_result(self, decision_event, execution, utterance=''):
        failed = execution in {'failed', 'rejected'}
        fields = (
            'phase', 'action', 'role', 'hand', 'target', 'chosen', 'chosen_ranks',
            'selected_indices', 'source', 'bid_power',
        )
        event = {key: decision_event[key] for key in fields if key in decision_event}
        return self.write_decision_event({
            **event,
            'event_kind': 'execution_result',
            'action_id': decision_event['action_id'],
            'result_for': decision_event['event_id'],
            'reason': utterance if failed and utterance else ('执行未成功' if failed else '执行状态回执'),
            'utterance': utterance if failed else '',
            'execution': execution,
            'speak': failed and bool(utterance),
        })

    def scene_event(self, plan, text, execution='decided'):
        return self.write_decision_event({
            'phase': plan.get('phase', 'lobby'),
            'action': plan['action'],
            'role': None,
            'hand': [],
            'target': None,
            'chosen': plan['action'],
            'chosen_ranks': [],
            'reason': plan.get('reason', '处理非对局场景'),
            'utterance': plan.get('utterance', ''),
            'source': plan.get('source', 'scene_ocr'),
            'execution': execution,
            'scene_text': compact_scene_text(text)[:200],
            'outcome': plan.get('outcome'),
        })

    def perform_scene_action(self, hwnd, bbox, scale_x, scale_y, plan, text):
        if plan.get('blocked') or plan.get('wait'):
            key = plan['action']
            if self.last_idle_notice == key:
                return True
            self.last_idle_notice = key
            execution = 'blocked' if plan.get('blocked') else 'waiting'
            self.scene_event(plan, text, execution)
            self.log({'state': 'scene_waiting', 'action': plan['action'], 'text': compact_scene_text(text)[:200]})
            self.notify(f"【场景】{plan.get('utterance')}", 'warn')
            return True
        now = time.perf_counter()
        if now - self.scene_action_times.get(plan['action'], -100) < 2.5:
            return False
        self.scene_action_times[plan['action']] = now
        self.last_idle_notice = None
        decision_event = self.scene_event(plan, text)
        clicked = False
        if self.ensure_foreground(hwnd):
            clicked = self.click(hwnd, bbox, plan['point'], scale_x, scale_y, physical=True)
        self.write_execution_result(
            decision_event,
            'submitted' if clicked else 'failed',
            '游戏场景按钮未响应点击，准备重新识别后重试。',
        )
        self.last_action = time.perf_counter()
        self.log({'state': 'scene_action', 'action': plan['action'], 'click': plan['point'],
                  'text': compact_scene_text(text)[:200], 'clicked': clicked})
        self.notify(f"【场景】{plan.get('utterance')}", 'decision')
        return clicked

    def read_scene_ocr(self, rgb):
        image_path = ROOT / 'outputs' / 'mini_scene_ocr.png'
        output_path = ROOT / 'outputs' / 'mini_scene_ocr.json'
        mini_perceive.save_image(image_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        subprocess.run(
            [
                'powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass',
                '-File', str(ROOT / 'ocr_words.ps1'),
                '-Path', str(image_path), '-Out', str(output_path),
            ],
            cwd=str(ROOT), check=True, capture_output=True, timeout=12,
        )
        return json.loads(output_path.read_text(encoding='utf-8'))

    def rapidocr(self):
        engine = getattr(self, '_rapidocr', None)
        if engine is None:
            from rapidocr_onnxruntime import RapidOCR
            engine = RapidOCR()
            self._rapidocr = engine
        return engine

    def read_badge_count(self, rgb, seat, scale_x, scale_y):
        x0, y0, x1, y1 = BADGE_ROIS[seat]
        crop = rgb[round(y0 * scale_y):round(y1 * scale_y), round(x0 * scale_x):round(x1 * scale_x)]
        if crop.size == 0:
            return None
        inverted = 255 - cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        image = cv2.resize(cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR), None, fx=4, fy=4,
                           interpolation=cv2.INTER_CUBIC)
        result, _ = self.rapidocr()(image)
        digits = ''.join(re.sub(r'\D', '', str(text)) for _, text, _ in result or [])
        if len(digits) >= 2:
            value = int(digits[:2])
            return value if 10 <= value <= 20 else None
        ocr_digit = int(digits) if digits else None
        scores = mini_perceive.match_badge_digit(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        if not scores:
            return None
        score, digit = scores[0]
        second_score = scores[1][0] if len(scores) > 1 else -1.0
        if score >= 0.72 and score - second_score >= 0.04:
            value = int(digit)
            return value if ocr_digit is None or ocr_digit == value else None
        return None

    def update_counts_from_badges(self, rgb, scale_x, scale_y, force=False):
        now = time.perf_counter()
        if not force and now - self.last_count_probe < 1.0:
            return
        self.last_count_probe = now
        for seat in ('left', 'right'):
            value = self.read_badge_count(rgb, seat, scale_x, scale_y)
            if value is None:
                self.badge_last_read[seat] = None
                continue
            if value > self.counts[seat] + 2 and self.has_played[seat]:
                self.log({'state': 'count_read_rejected', 'seat': seat, 'count': value,
                          'current': self.counts[seat]})
                self.badge_last_read[seat] = None
                continue
            if value < 10 or abs(value - self.counts[seat]) > 2:
                if self.badge_last_read[seat] != value:
                    self.badge_last_read[seat] = value
                    self.log({'state': 'count_read_pending', 'seat': seat, 'count': value,
                              'current': self.counts[seat]})
                    continue
            self.badge_last_read[seat] = value
            if value != self.counts[seat]:
                self.log({'state': 'count_read', 'seat': seat, 'count': value,
                          'current': self.counts[seat]})
            self.counts[seat] = value
            self.badge_counts[seat] = value
            self.badge_at[seat] = now

    def read_ui_buttons(self, rgb, scale_x, scale_y):
        x0, y0, x1, y1 = UI_ROI
        crop = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)[
            round(y0 * scale_y):round(y1 * scale_y),
            round(x0 * scale_x):round(x1 * scale_x),
        ]
        result, _ = self.rapidocr()(crop)
        buttons = []
        for box, text, confidence in result or []:
            center_x = (min(point[0] for point in box) + max(point[0] for point in box)) / 2
            center_y = (min(point[1] for point in box) + max(point[1] for point in box)) / 2
            buttons.append({
                'text': str(text).strip(),
                'confidence': round(float(confidence), 3),
                'source': 'rapidocr',
                'center': [round((round(x0 * scale_x) + center_x) / scale_x), round((round(y0 * scale_y) + center_y) / scale_y)],
            })
        return buttons

    def find_hint_button(self, rgb, scale_x, scale_y):
        for button in self.read_ui_buttons(rgb, scale_x, scale_y):
            if compact_scene_text(button['text']) == '提示':
                return button
        return None

    def read_bid_buttons(self, rgb, scale_x, scale_y):
        x0, y0, x1, y1 = UI_ROI
        crop = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)[
            round(y0 * scale_y):round(y1 * scale_y),
            round(x0 * scale_x):round(x1 * scale_x),
        ]
        pixels = crop.tobytes()
        now = time.perf_counter()
        cached = self.bid_button_cache
        if cached and cached['pixels'] == pixels and now - cached['time'] < 1.0:
            return cached['buttons']
        buttons = []
        for filename, text in [('call_landlord.png', '叫地主'), ('pass_bid.png', '不叫')]:
            path = BID_TEMPLATE_DIR / filename
            if not path.exists():
                continue
            template = mini_perceive.load_image(path)
            template = cv2.resize(template, (round(template.shape[1] * scale_x), round(template.shape[0] * scale_y)),
                                  interpolation=cv2.INTER_AREA)
            if crop.shape[0] < template.shape[0] or crop.shape[1] < template.shape[1]:
                continue
            result = cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED)
            _, score, _, location = cv2.minMaxLoc(result)
            if score < 0.72:
                continue
            center_x = location[0] + template.shape[1] / 2
            center_y = location[1] + template.shape[0] / 2
            buttons.append({
                'text': text,
                'confidence': round(float(score), 3),
                'source': 'template',
                'center': [round((round(x0 * scale_x) + center_x) / scale_x), round((round(y0 * scale_y) + center_y) / scale_y)],
            })
        if not buttons:
            buttons = self.read_ui_buttons(rgb, scale_x, scale_y)
        self.bid_button_cache = {'pixels': pixels, 'time': now, 'buttons': buttons}
        return buttons

    def handle_dialog_scene(self, hwnd, bbox, rgb, scale_x, scale_y):
        try:
            text = self.read_scene_ocr(rgb).get('text', '')
        except Exception as error:
            text = ''
            self.log({'state': 'scene_ocr_failed', 'error': str(error), 'scene': 'dialog'})
        plan = dialog_scene_plan(text)
        if plan.get('blocked'):
            key = compact_scene_text(text)
            if self.blocked_scene != key:
                self.blocked_scene = key
                self.scene_event(plan, text, 'blocked')
                if plan['action'] == 'skip_video_beans':
                    self.log({'state': 'video_beans_skipped', 'text': key[:200]})
                    self.notify('【返豆】看视频返豆的按钮我不点，先停在这里。', 'warn')
                else:
                    self.log({'state': 'purchase_confirmation_required', 'text': key[:200]})
                    self.notify(f"【补豆】{plan['utterance']}", 'error')
            return
        self.blocked_scene = None
        self.perform_scene_action(hwnd, bbox, scale_x, scale_y, plan, text)

    def handle_idle_scene(self, hwnd, bbox, rgb, scale_x, scale_y):
        if self.hand_active:
            return
        now = time.perf_counter()
        if now - self.last_scene_ocr < 2.0:
            return
        self.last_scene_ocr = now
        try:
            if mini_perceive.detect_hand(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))['count'] >= 2:
                return
        except ValueError:
            pass
        try:
            ocr = self.read_scene_ocr(rgb)
        except Exception as error:
            ocr = {}
            self.log({'state': 'scene_ocr_failed', 'error': str(error), 'scene': 'idle'})
        text = ocr.get('text', '')
        plan = idle_scene_plan(text)
        if plan['action'] == 'select_novice_room':
            point = room_select_point(ocr.get('words') or [], scale_x, scale_y)
            if point is None:
                self.log({'state': 'room_select_unlocated', 'text': compact_scene_text(text)[:200]})
                self.notify('【场景】看到选场页，但没定位到新手场，先不乱点', 'warn')
                return
            plan['point'] = point
        self.perform_scene_action(hwnd, bbox, scale_x, scale_y, plan, text)

    def is_bomb_play(self, ranks):
        if not ranks:
            return False
        if sorted(ranks) == ['BJ', 'LJ']:
            return True
        return len(ranks) == 4 and len(set(ranks)) == 1

    def clean_played(self):
        from collections import Counter
        seen = Counter()
        result = []
        for rank in self.played:
            limit = 1 if rank in {'BJ', 'LJ'} else 4
            if seen[rank] < limit:
                result.append(rank)
                seen[rank] += 1
        return result

    def narrate_final(self, hand, role, target_info, candidate, counts=None):
        from model_decision import load_api_key, parse_content, post_chat
        config = json.loads((ROOT / 'api_config.json').read_text(encoding='utf-8'))
        config['timeout_seconds'] = 3
        move = '过' if candidate.get('id') == 'pass' else ' '.join(candidate.get('ranks') or [])
        state = {
            'final_move': move,
            'move_type': candidate.get('type'),
            **self.fixed_role_context(role),
            'my_hand': hand,
            'last_play': target_info,
            'card_names': CARD_NAMES,
            'remaining_counts': counts or dict(self.counts),
            'instruction': (
                '最终出牌已经确定为 final_move，不要再改牌。'
                '请写简短理由和 1–2 句中性口播素材，包含最终动作、压制的目标（如有）和一个核心理由；'
                '素材必须与 final_move 和理由一致，可以说明保留的关键牌，但不要把未打出的牌说成已出。'
                '不使用语气词、口头禅、人设称呼或情绪表演。'
                '输出 JSON：{"reason":"简短原因","utterance":"包含动作、目标和核心理由的中性口播素材"}。不要输出 Markdown。'
            ),
        }
        messages = [
            {'role': 'system', 'content': '你是斗地主中性解说素材生成器。只解释已经确定的这一手，只输出一个 JSON 对象，不添加固定人设。'},
            {'role': 'user', 'content': json.dumps(state, ensure_ascii=False)},
        ]
        try:
            response, _, _ = post_chat(config, load_api_key(config), messages)
            decision, _ = parse_content(response)
            reason = str(decision.get('reason') or '').strip()
            utterance = str(decision.get('utterance') or '').strip()
            if not utterance:
                raise ValueError('empty utterance')
            return {'reason': reason, 'utterance': utterance}
        except Exception as error:
            self.log({'state': 'narration_failed', 'error': str(error), 'move': move})
            reasons = candidate.get('reasons') or []
            reason = '；'.join(reasons[:2]) or '采用本地引擎推荐'
            action = '选择过牌' if move == '过' else f'选择出{move}'
            target = ''
            if target_info and target_info.get('ranks'):
                target = f'目标是{" ".join(target_info["ranks"])}，'
            return {'reason': reason, 'utterance': f'{action}，{target}{reason}。'}

    def eligible_candidates(self, shortlist, recommended, limit=3):
        by_id = {item['id']: item for item in shortlist}
        rec = by_id.get(recommended)
        if rec is None:
            return shortlist[:limit]
        score = rec.get('local_score', 0)
        alternatives = [item for item in shortlist if item['id'] != rec['id']]
        alternatives.sort(key=lambda item: (abs(item.get('local_score', 0) - score), item.get('local_score', 0)))
        return [rec] + alternatives[:limit - 1]

    def hand_strength(self, hand):
        from collections import Counter
        counts = Counter(hand)
        score = 0.0
        score += 4.0 if 'BJ' in counts else 0.0
        score += 3.0 if 'LJ' in counts else 0.0
        if 'BJ' in counts and 'LJ' in counts:
            score += 4.0
        score += 2.0 * counts.get('2', 0)
        score += 4.0 * sum(1 for rank, count in counts.items() if count == 4 and rank not in {'BJ', 'LJ'})
        score += 0.5 * counts.get('A', 0)
        score += 0.3 * counts.get('K', 0)
        return round(score, 2)

    def read_hand_with_retry(self, hwnd, attempts=2, delay=0.3):
        best_rgb = None
        best_observation = None
        best_unknown = None
        for attempt in range(1, attempts + 1):
            rgb, _ = self.capture(hwnd)
            error = None
            try:
                observation = mini_perceive.read_hand(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                unknown = observation['unknown_indices']
            except ValueError as caught:
                observation = None
                unknown = []
                error = str(caught)
            if best_rgb is None:
                best_rgb = rgb
            if observation is not None and (best_unknown is None or len(unknown) < best_unknown):
                best_rgb = rgb
                best_observation = observation
                best_unknown = len(unknown)
            known_count = len(observation['my_hand']) - len(unknown) if observation is not None else 0
            if observation is not None and len(unknown) <= 1 and known_count >= 2:
                if attempt > 1:
                    self.log({'state': 'hand_read_recovered', 'attempt': attempt,
                              'unknown_indices': unknown})
                return rgb, observation
            if attempt < attempts:
                self.log({'state': 'hand_read_retry', 'attempt': attempt,
                          'unknown_indices': unknown, 'error': error,
                          'hand': observation['my_hand'] if observation is not None else []})
                time.sleep(delay)
        self.log({'state': 'hand_read_unreliable', 'attempts': attempts,
                  'unknown_indices': (best_observation or {}).get('unknown_indices', []),
                  'hand': (best_observation or {}).get('my_hand', []),
                  'count': (best_observation or {}).get('count')})
        if best_rgb is not None:
            self.save_action_frame(best_rgb, 'hand_unreliable')
        return best_rgb, best_observation

    def handle_bid(self, hwnd, bbox, scale_x, scale_y, ui):
        if self.bid_complete:
            return
        if self.bid_last is not None:
            previous_key, previous_time = self.bid_last
            click_key = self.bid_click_keys.get(previous_key)
            if click_key and self.bid_clicks.get(click_key, 0) >= 2 and time.perf_counter() - previous_time < 15:
                self.log({'state': 'bid_click_deduped', 'stage': click_key[0], 'label': click_key[1]})
                return
        rgb, observation = self.read_hand_with_retry(hwnd)
        hand = [rank for rank in observation['my_hand'] if rank != 'UNKNOWN'] if observation is not None else []
        score = 0
        power = None
        source = 'local_engine'
        try:
            result = self.bot_decision('bidding', hand)
            power = result.get('power')
            score = int((result.get('action') or {}).get('score', 0) or 0)
        except Exception as error:
            power = self.hand_strength(hand)
            source = 'local_fallback'
            self.log({'state': 'bid_engine_failed', 'error': str(error), 'hand': hand})
        try:
            buttons = self.read_bid_buttons(rgb, scale_x, scale_y)
        except Exception as error:
            buttons = []
            self.log({'state': 'bid_ocr_failed', 'error': str(error)})
        button_texts = {item['text'] for item in buttons}
        grab_stage = '抢地主' in button_texts
        first_stage_buttons = button_texts & {'叫地主', '抢地主', '不叫'}
        double_stage_buttons = button_texts & {'加倍', '超级加倍', '不加倍'}
        if first_stage_buttons and not double_stage_buttons:
            first_bid = True
        elif double_stage_buttons and not first_stage_buttons:
            first_bid = False
        else:
            first_bid = not self.bid_done
        if grab_stage:
            self.hand_id = f'{self.events.session_id}-hand-{self.hands_done + 1}'
            call = power is not None and power >= 20
        elif first_bid:
            self.hand_id = f'{self.events.session_id}-hand-{self.hands_done + 1}'
            if self.bid_done:
                # 已经叫过牌，再出现“第一轮”按钮只可能是抢地主/不抢；
                # OCR 常把“抢”读成“叫”，所以门槛一律按 20 算，宁可不抢也不误抢。
                grab_stage = True
                call = power is not None and power >= 20
            else:
                call = power is not None and power >= 10
        else:
            call = power is not None and power >= 20
        self.bid_done = True
        self.bid_complete = not first_bid
        self.role = None
        self.hand_active = True
        if grab_stage:
            label = '抢地主' if call else '不抢'
        else:
            label = ('叫地主' if call else '不叫') if first_bid else ('加倍' if call else '不加倍')
        super_buttons = [item for item in buttons if '超级加倍' in item['text']]
        if not first_bid and call:
            double_items = [
                item for item in buttons
                if item['text'] == '加倍'
                and not any(abs(item['center'][0] - super_item['center'][0]) < 80 for super_item in super_buttons)
            ]
            if not double_items and buttons:
                call = False
                label = '不加倍'
        if grab_stage:
            labels = ('抢地主',) if call else ('不抢',)
        else:
            labels = ('叫地主', '抢地主') if first_bid and call else (('不叫',) if first_bid else (('加倍',) if call else ('不加倍', '不叫')))
        matched = next((item for item in buttons if any(label == item['text'] for label in labels)
                        and not (call and not first_bid and '超级加倍' in item['text'])
                        and not (call and not first_bid and any(
                            abs(item['center'][0] - super_item['center'][0]) < 80 for super_item in super_buttons))), None)
        if not first_bid and call and matched is None:
            call = False
            label = '不加倍'
            matched = next((item for item in buttons if item['text'] in ('不加倍', '不叫')), None)
        orange = ui['orange'][0]['center'] if ui['orange'] else [374, 333]
        blue = ui.get('blue') or []
        pass_point = max(blue, key=lambda item: item['center'][0])['center'] if blue else [orange[0] + 230, orange[1]]
        if matched:
            point = matched['center']
            point_source = matched.get('source', 'rapidocr')
        else:
            point = orange if call else pass_point
            point_source = 'orange_button' if call else ('blue_button' if blue else 'legacy_offset')
        if grab_stage:
            action = 'grab_landlord' if call else 'pass_grab'
            utterance = f'牌力 {power}，{"达到" if call else "未达到"}抢地主门槛（20），选择{label}。'
        elif first_bid:
            action = 'call_landlord' if call else 'pass_bid'
            utterance = f'牌力 {power}，{"达到" if call else "未达到"}叫地主门槛，选择{label}。'
        else:
            action = 'double' if call else 'no_double'
            utterance = f'牌力 {power}，{"达到" if call else "未达到"}加倍门槛，选择{label}。'
        bid_key = (first_bid, action, tuple(hand))
        now = time.perf_counter()
        repeated = self.bid_last is not None and self.bid_last[0] == bid_key and now - self.bid_last[1] < 15
        self.bid_last = (bid_key, now)
        click_key = ('grab' if grab_stage else ('double' if not first_bid else 'bid'), label)
        clicked_times = self.bid_clicks.get(click_key, 0)
        self.bid_click_keys[bid_key] = click_key
        if clicked_times >= 2:
            self.log({'state': 'bid_click_deduped', 'stage': click_key[0], 'label': label,
                      'attempts': clicked_times, 'power': power})
            return
        self.bid_clicks[click_key] = clicked_times + 1
        decision_event = self.write_decision_event({
            'phase': 'bid' if first_bid else 'double',
            'action': action,
            'role': self.role,
            'hand': hand,
            'target': None,
            'chosen': action,
            'chosen_ranks': [],
            'reason': f'{label}；牌力 {power}，当前门槛 {20 if grab_stage or not first_bid else 10}',
            'utterance': utterance,
            'source': source,
            'execution': 'decided',
            'bid_power': power,
            'speak': not repeated,
        })
        clicked = self.click(hwnd, bbox, point, scale_x, scale_y, physical=True)
        self.write_execution_result(
            decision_event,
            'submitted' if clicked else 'failed',
            f'{label}操作未点击成功，准备重试。',
        )
        self.last_action = time.perf_counter()
        self.log({'state': 'bid_decision', 'first_bid': first_bid, 'action': 'call' if call else 'pass',
                  'bid_score': score, 'power': power, 'hand': hand, 'click': point,
                  'point_source': point_source, 'blue_buttons': [item['center'] for item in blue],
                  'button_texts': [item['text'] for item in buttons]})
        self.notify(f"【{'叫牌' if first_bid else '加倍'}】{label}｜bidPower {power}")

    def handle_lead(self, hwnd, bbox, scale_x, scale_y, ui):
        self.hand_active = True
        if self.check_tuoguan(hwnd, bbox, scale_x, scale_y):
            return
        rgb, observation = self.read_hand_with_retry(hwnd)
        if observation is None:
            self.log({'state': 'lead_unreadable'})
            return
        hand = observation['my_hand']
        key = tuple(hand)
        self.log({'state': 'lead_observe', 'hand': hand, 'selected': observation['selected_indices'], 'unknown': observation['unknown_indices']})
        if self.lead_blocked_key == key:
            return
        cached = self.lead_cache if self.lead_cache_key == (key, self.role) else None
        if cached:
            chosen = cached['chosen']
            candidate = cached['candidate']
            source = 'cached_decision'
            decision = cached['decision']
            self.log({'state': 'lead_cache_hit', 'ranks': candidate.get('ranks'), 'source': source})
            observation = mini_perceive.read_hand(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        known = [(index, rank) for index, rank in enumerate(hand) if rank != 'UNKNOWN']
        index_map = [index for index, _ in known]
        if cached:
            rules_hand = [rank for _, rank in known]
        elif len(observation['unknown_indices']) > 1 or len(known) < 2:
            picked = self.pick_safe_single(known)
            if picked is None:
                self.log({'state': 'lead_fallback_unavailable', 'hand': hand})
                return
            index, rank = picked
            chosen = [index]
            candidate = {'id': 'fallback_single', 'type': 'single', 'ranks': [rank], 'indices': [index]}
            decision = {
                'reason': '手牌识别不完整，为保留已知对子和三张，选择已识别单张',
                'utterance': f'手牌识别不完整，为保留已知对子和三张，选择单张{rank}。',
            }
            source = 'fallback_single'
            rules_hand = [rank for _, rank in known]
        else:
            rules_hand = [rank for _, rank in known]
            index_map = [index for index, _ in known]
            request = {'my_hand': rules_hand, 'selected_ranks': [], 'selected_indices': [], 'last_play': None}
            (ROOT / 'outputs' / 'mini_rule_input.json').write_text(json.dumps(request, ensure_ascii=False), encoding='utf-8')
            subprocess.run(
                ['node', 'rules_cli.mjs', 'outputs/mini_rule_input.json', 'outputs/mini_rule_output.json'],
                cwd=str(ROOT), check=True, capture_output=True, timeout=10,
            )
            rules = json.loads((ROOT / 'outputs' / 'mini_rule_output.json').read_text(encoding='utf-8'))
            legal = [candidate for candidate in rules['candidates'] if candidate['id'] != 'pass']
            if not legal:
                self.log({'state': 'lead_no_candidates'})
                return
            if self.role is None:
                self.role = 'landlord' if observation['count'] >= 19 else 'farmer'
            decision = None
            candidate = None
            source = None
            shortlist = []
            try:
                result = self.bot_decision('playing', rules_hand, role=self.role, can_pass=False, mode='candidates')
                shortlist = [item for item in result.get('candidates', []) if item['id'] != 'pass']
                if not shortlist:
                    raise ValueError('local engine returned no candidates')
                eligible = self.eligible_candidates(shortlist, result.get('recommended'))
                brief = ' / '.join(f"{item['id']}:{' '.join(item['ranks'])}" for item in eligible)
                self.notify(f"【可改选】{brief}（本地推荐 {result.get('recommended')}）", 'dim')
                try:
                    decision = self.deepseek_pick(rules_hand, self.role, eligible,
                                                 counts=dict(self.counts), recommended=result.get('recommended'))
                except Exception as error:
                    self.log({'state': 'deepseek_failed', 'error': str(error)})
                if decision:
                    candidate = next(item for item in eligible if item['id'] == decision['candidate_id'])
                    source = 'deepseek_limited'
                else:
                    recommended = result.get('recommended')
                    candidate = next((item for item in shortlist if item['id'] == recommended), shortlist[0])
                    source = 'local_recommended'
                    decision = {'reason': '本地引擎推荐', 'utterance': ''}
                chosen = [index_map[position] for position in candidate['indices']]
            except Exception as error:
                candidate = pick_lead(rules_hand, legal)
                source = 'local_fallback'
                decision = {'reason': f'本地兜底：{error}', 'utterance': ''}
                chosen = [index_map[position] for position in candidate['source_indices']]
            if not (decision.get('utterance') or '').strip():
                narration = self.narrate_final(rules_hand, self.role, None, candidate)
                decision = {**decision, **narration}
            self.lead_cache_key = (key, self.role)
            self.lead_cache = {'chosen': chosen, 'candidate': candidate, 'decision': decision}
            self.log({'state': 'lead_decision', 'source': source, 'candidate': candidate.get('id'),
                      'ranks': candidate.get('ranks'), 'reason': decision.get('reason'), 'utterance': decision.get('utterance')})
            self.notify(f"【决策】{' '.join(candidate.get('ranks') or [])}｜{decision.get('reason')}｜口播：{decision.get('utterance')}", 'decision')

        decision_event = self.write_decision_event({
            'phase': 'lead',
            'action': 'play',
            'role': self.role,
            'hand': rules_hand,
            'target': None,
            'chosen': candidate.get('id'),
            'chosen_ranks': candidate.get('ranks'),
            'selected_indices': chosen,
            'reason': decision.get('reason'),
            'utterance': decision.get('utterance'),
            'source': source,
            'execution': 'decided',
            'speak': source != 'cached_decision',
        })

        if self.check_tuoguan(hwnd, bbox, scale_x, scale_y):
            self.write_execution_result(decision_event, 'failed', '检测到托管接管操作，本次出牌暂停，需要先取消托管。')
            return
        if self.set_selection(hwnd, bbox, observation, chosen, scale_x, scale_y,
                              limit=self.selection_batch) is False:
            self.write_execution_result(decision_event, 'failed', '目标牌选择操作未完成，本次出牌未执行，准备重试。')
            return
        time.sleep(0.45)
        if self.check_tuoguan(hwnd, bbox, scale_x, scale_y):
            self.write_execution_result(decision_event, 'failed', '检测到托管接管操作，本次出牌暂停，需要先取消托管。')
            return
        rgb2, _ = self.capture(hwnd)
        confirmed, rgb2 = self.confirm_selection(hwnd, bbox, rgb2, chosen, scale_x, scale_y, before=observation)
        if not confirmed:
            if candidate.get('id') == 'selection_fallback':
                self.lead_blocked_key = key
                self.write_execution_result(decision_event, 'failed', '选牌连续两次未能选全，本手停止自动出牌，等待手牌变化。')
                self.notify('【停止】选牌连续失败，等待手牌/界面变化', 'error')
                return
            residual = (self.last_selection_check or {}).get('selected')
            if residual is None or residual:
                self.lead_blocked_key = key
                self.log({'state': 'lead_selection_residual_blocked', 'selected': residual})
                self.write_execution_result(decision_event, 'failed', '原选牌仍有残余或状态不明，本手停止出牌，不换牌提交。')
                self.notify('【停止】残余选牌未清空，本手不再换牌提交', 'error')
                return
            alternative = self.pick_lead_selection_fallback(rules_hand, index_map)
            if alternative is None:
                self.write_execution_result(decision_event, 'failed', '选牌点击未全部生效，本次不提交出牌，等待重新识别。')
                return
            self.lead_cache_key = (key, self.role)
            self.lead_cache = alternative
            self.write_execution_result(
                decision_event,
                'failed',
                f'原选牌未确认，改出{" ".join(alternative["candidate"].get("ranks") or [])}。',
            )
            self.notify(f"【选牌】原牌未选全，下一轮改出 {' '.join(alternative['candidate'].get('ranks') or [])}", 'warn')
            return
        _, _, ui2 = classify(rgb2, scale_x, scale_y)
        if ui2 and ui2['orange']:
            point = ui2['orange'][0]['center']
        else:
            point = ui['orange'][0]['center']
        clicked = self.click(hwnd, bbox, point, scale_x, scale_y, physical=True)
        self.last_action = time.perf_counter()
        if not clicked:
            self.write_execution_result(decision_event, 'failed', '出牌按钮未响应点击，本次出牌未执行，准备重试。')
            return
        time.sleep(0.65)
        rgb3, _ = self.capture(hwnd)
        state3, _, _ = classify(rgb3, scale_x, scale_y)
        hand_before = observation['count']
        hand_after = count_hand_in_frame(rgb3)
        expected_remaining = hand_before - len(candidate.get('ranks') or [])
        execution = evaluate_play_execution(state3, 'lead', hand_before, hand_after, expected_remaining)
        if execution == 'confirmed':
            self.lead_blocked_key = None
            self.lead_failure_counts.pop(key, None)
            self.played.extend(candidate.get('ranks') or [])
            self.last_play = None
            self.write_execution_result(decision_event, execution)
            self.notify(f"【执行】已点 {len(chosen)} 张并出牌")
        elif execution == 'rejected':
            self.save_action_frame(rgb3, 'lead_rejected')
            failures = self.lead_failure_counts.get(key, 0) + 1
            self.lead_failure_counts[key] = failures
            if failures >= 2:
                self.lead_blocked_key = key
                correction = '同一牌型第二次提交仍未生效，停止重复提交，等待状态变化。'
            else:
                correction = '牌型首次提交未生效，下一次按相同牌型重试一次。'
            self.write_execution_result(decision_event, execution, correction)
            self.notify('【失误】出牌未生效，已补纠正口播', 'error')
        else:
            self.save_action_frame(rgb3, 'lead_uncertain')
            self.write_execution_result(decision_event, execution)
            self.notify('【执行状态】界面已变化，但手牌证据不足，等待确认', 'warn')
        self.log({'state': 'lead_submit', 'source': source, 'indices': chosen, 'click': point,
                  'execution': execution, 'hand_before': hand_before, 'hand_after': hand_after,
                  'expected_remaining': expected_remaining, 'state_after': state3})

    def check_tuoguan(self, hwnd, bbox, scale_x, scale_y):
        rgb, _ = self.capture(hwnd)
        score = self.tuoguan_score(rgb, scale_x, scale_y)
        if score > 0.7:
            if self.ensure_foreground(hwnd):
                self.click(hwnd, bbox, TUOGUAN_POINT, scale_x, scale_y, physical=True)
                self.log({'state': 'cancel_tuoguan', 'click': TUOGUAN_POINT, 'match': round(score, 3), 'phase': 'handler'})
                self.notify('【托管】处理中检测到取消托管，已点掉', 'error')
                self.last_action = time.perf_counter()
                time.sleep(0.5)
            return True
        return False

    def update_role_from_table(self, rgb):
        if self.role is not None:
            return
        seat = mini_perceive.detect_landlord(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if seat is not None and not self.landlord_count_adjusted:
            # 徽章是更准的来源：有已确认的读数就用它，没有才按 20 - 已出张数回推。
            if self.badge_counts[seat] is not None:
                self.counts[seat] = self.badge_counts[seat]
            elif self.has_played[seat]:
                self.counts[seat] = max(0, 20 - self.played_total[seat])
            else:
                self.counts[seat] = 20
            self.landlord_count_adjusted = True
            self.log({'state': 'landlord_count_initialized', 'seat': seat, 'counts': dict(self.counts),
                      'played_total': dict(self.played_total), 'badge': self.badge_counts[seat]})
        self.landlord_seat = seat
        self.role = 'landlord' if seat is None else 'farmer'

    def apply_play_count(self, seat, amount, seen_at=None):
        seen_at = seen_at or time.perf_counter()
        already_counted = self.badge_at[seat] >= seen_at
        self.play_at[seat] = seen_at
        if already_counted:
            return False
        self.counts[seat] = max(0, self.counts[seat] - amount)
        self.played_total[seat] += amount
        return True

    def track_plays(self, rgb):
        now = time.perf_counter()
        if now - self.last_play_probe < 0.25:
            return
        self.last_play_probe = now
        try:
            played = mini_perceive.detect_played(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        except Exception:
            return
        play_changed = False
        for seat in ('left', 'right'):
            info = played[seat]
            ranks = tuple(info['ranks'])
            if ranks and not info.get('reliable', True):
                if ranks != self.play_unreliable_candidate[seat]:
                    self.play_unreliable_candidate[seat] = ranks
                    self.unreliable_since[seat] = now
                    continue
                if now - self.unreliable_since.get(seat, now) < 0.6:
                    continue
                if ranks != self.play_unreliable[seat]:
                    self.log({'state': 'target_unreliable', 'seat': seat, 'ranks': info['ranks'],
                              'count_est': info.get('count_est')})
                    self.play_unreliable[seat] = ranks
                    if ranks != self.play_state[seat]:
                        self.last_play = None
                        self.follow_retry_key = None
                        self.follow_cache_key = None
                        self.follow_cache = None
                    estimate = info.get('count_est') or len(ranks)
                    counted = self.apply_play_count(seat, estimate, self.unreliable_since[seat])
                    self.has_played[seat] = True
                    play_changed = True
                    self.log({'state': 'play_count_estimated', 'seat': seat, 'ranks': list(ranks),
                              'estimate': estimate, 'counted': counted, 'counts': dict(self.counts)})
                continue
            self.play_unreliable[seat] = ()
            self.play_unreliable_candidate[seat] = ()
            self.unreliable_since[seat] = 0.0
            if ranks != self.play_candidate[seat]:
                self.play_candidate[seat] = ranks
                self.play_candidate_at[seat] = now
                continue
            if ranks != self.play_state[seat]:
                self.play_state[seat] = ranks
                if ranks:
                    counted = self.apply_play_count(seat, len(ranks), self.play_candidate_at[seat])
                    self.has_played[seat] = True
                    play_changed = True
                    self.played.extend(ranks)
                    self.last_play = {'seat': seat, 'ranks': list(ranks), 'time': now}
                    self.follow_retry_key = None
                    self.follow_cache_key = None
                    self.follow_cache = None
                    self.log({'state': 'play_tracked', 'seat': seat, 'ranks': list(ranks),
                              'counted': counted, 'counts': dict(self.counts)})
        scale_x = rgb.shape[1] / BASE_WIDTH
        scale_y = rgb.shape[0] / BASE_HEIGHT
        self.update_counts_from_badges(rgb, scale_x, scale_y, force=play_changed)

    def handle_turn(self, hwnd, bbox, scale_x, scale_y, ui):
        self.hand_active = True
        if self.check_tuoguan(hwnd, bbox, scale_x, scale_y):
            return
        orange = ui['orange'][0]['center']
        pass_point = [orange[0] - 328, orange[1]]
        target_seat = None
        target = []
        target_off_by_one = False
        target_from_detection = False
        try:
            for attempt in range(3):
                rgb, _ = self.capture(hwnd)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                if self.last_play and time.perf_counter() - self.last_play['time'] < 15:
                    target_seat = self.last_play['seat']
                    target = self.last_play['ranks']
                    break
                played = mini_perceive.detect_played(bgr)
                for seat in ('left', 'right'):
                    info = played[seat]
                    if info['ranks'] and info.get('reliable', True):
                        target_seat = seat
                        target = info['ranks']
                        target_from_detection = True
                        break
                if target:
                    break
                seat, ranks = off_by_one_target(played)
                if seat is not None:
                    target_seat, target, target_off_by_one = seat, ranks, True
                    target_from_detection = True
                    self.log({'state': 'target_off_by_one', 'seat': seat, 'ranks': ranks})
                    break
                if attempt < 2:
                    self.log({'state': 'target_retry', 'attempt': attempt + 1})
                    time.sleep(0.3)
        except ValueError as error:
            self.log({'state': 'follow_unreadable', 'error': str(error)})
            return self.fallback_hint(hwnd, bbox, scale_x, scale_y, ui)
        rgb, observation = self.read_hand_with_retry(hwnd)
        if observation is None:
            self.log({'state': 'follow_unreadable'})
            return self.fallback_hint(hwnd, bbox, scale_x, scale_y, ui)
        if not target:
            self.log({'state': 'follow_no_target'})
            return self.fallback_hint(hwnd, bbox, scale_x, scale_y, ui)
        known = [(index, rank) for index, rank in enumerate(observation['my_hand']) if rank != 'UNKNOWN']
        rules_hand = [rank for _, rank in known]
        index_map = [index for index, _ in known]
        if len(rules_hand) < 2 or len(observation['unknown_indices']) > 1:
            self.log({'state': 'follow_hand_unreliable', 'target_seat': target_seat, 'target': target,
                      'known': rules_hand, 'unknown_indices': observation['unknown_indices']})
            return self.fallback_hint(hwnd, bbox, scale_x, scale_y, ui)
        if self.role is None:
            self.role = 'landlord' if observation['count'] >= 19 else 'farmer'
        counts_for_engine = counts_for_target(self.counts, target_seat, target, target_from_detection)
        if target_from_detection and target_seat in self.counts:
            self.log({'state': 'target_count_adjusted', 'seat': target_seat, 'target': target,
                      'counts': counts_for_engine})
        follow_key = (tuple(rules_hand), target_seat, tuple(target), self.role)
        cached = self.follow_cache if self.follow_cache_key == follow_key else None
        if cached:
            chosen = cached['chosen']
            candidate = cached['candidate']
            decision = cached['decision']
            source = 'cached_decision'
            is_pass = candidate.get('id') == 'pass' or candidate.get('type') == 'pass'
            self.log({'state': 'follow_cache_hit', 'target': target, 'candidate': candidate.get('id'),
                      'ranks': candidate.get('ranks')})
        else:
            request = {'my_hand': rules_hand, 'selected_ranks': [], 'selected_indices': [], 'last_play': {'ranks': target}}
            (ROOT / 'outputs' / 'mini_rule_input.json').write_text(json.dumps(request, ensure_ascii=False), encoding='utf-8')
            try:
                subprocess.run(
                    ['node', 'rules_cli.mjs', 'outputs/mini_rule_input.json', 'outputs/mini_rule_output.json'],
                    cwd=str(ROOT), check=True, capture_output=True, timeout=10,
                )
                rules = json.loads((ROOT / 'outputs' / 'mini_rule_output.json').read_text(encoding='utf-8'))
            except Exception as error:
                self.log({'state': 'follow_target_invalid', 'target_seat': target_seat, 'target': target,
                          'counts': dict(self.counts), 'badge_counts': dict(self.badge_counts), 'error': str(error)})
                self.save_action_frame(rgb, 'follow_target_invalid')
                self.notify(f"【识别】目标 {' '.join(target)} 无法校验，已保存现场，改用提示", 'error')
                return self.fallback_hint(hwnd, bbox, scale_x, scale_y, ui)
            legal = rules['candidates']
            if not legal:
                return self.fallback_hint(hwnd, bbox, scale_x, scale_y, ui)
            if target_off_by_one:
                self.log({'state': 'target_off_by_one_accepted', 'seat': target_seat, 'ranks': target})
                self.notify(f"【目标】{' '.join(target)} 牌堆估计多 1 张，但组合合法，按识别牌型处理", 'warn')
            try:
                result = self.bot_decision('playing', rules_hand, role=self.role, target=target,
                                           target_seat=target_seat, can_pass=True, counts=counts_for_engine,
                                           mode='candidates')
                shortlist = follow_shortlist(result)
            except Exception as error:
                self.log({'state': 'follow_engine_failed', 'error': str(error)})
                return self.fallback_hint(hwnd, bbox, scale_x, scale_y, ui)
            eligible = self.eligible_candidates(shortlist, result.get('recommended'))
            brief = ' / '.join(f"{item['id']}:{' '.join(item['ranks']) or 'pass'}" for item in eligible)
            self.notify(f"【跟牌可改选】目标 {' '.join(target)}｜{brief}（本地推荐 {result.get('recommended')}）", 'dim')
            decision = None
            try:
                decision = self.deepseek_pick(rules_hand, self.role, eligible, target_info={'seat': target_seat, 'ranks': target},
                                              counts={**counts_for_engine, 'leftLandlord': self.landlord_seat == 'left',
                                                      'rightLandlord': self.landlord_seat == 'right'},
                                              recommended=result.get('recommended'))
            except Exception as error:
                self.log({'state': 'deepseek_failed', 'error': str(error)})
            if decision:
                candidate = next(item for item in eligible if item['id'] == decision['candidate_id'])
                source = 'deepseek_limited'
            else:
                recommended = result.get('recommended')
                candidate = next((item for item in shortlist if item['id'] == recommended), None)
                if candidate is None:
                    candidate = shortlist[-1] if shortlist[-1]['id'] == 'pass' else shortlist[0]
                source = 'local_recommended'
                decision = {'reason': 'DeepSeek 超时/异常，采用本地正式推荐', 'utterance': ''}
            if not (decision.get('utterance') or '').strip():
                narration = self.narrate_final(rules_hand, self.role, {'seat': target_seat, 'ranks': target}, candidate,
                                               counts=counts_for_engine)
                decision = {**decision, **narration}
            self.log({'state': 'follow_decision', 'target_seat': target_seat, 'target': target,
                      'source': source, 'candidate': candidate.get('id'), 'ranks': candidate.get('ranks'),
                      'reason': decision.get('reason'), 'utterance': decision.get('utterance')})
            self.notify(f"【跟牌决策】{'过' if candidate.get('id') == 'pass' else ' '.join(candidate.get('ranks') or [])}"
                        f"｜{decision.get('reason')}｜口播：{decision.get('utterance')}", 'decision')
            is_pass = candidate.get('id') == 'pass' or candidate.get('type') == 'pass'
            chosen = [] if is_pass else [index_map[position] for position in candidate['indices']]
            self.follow_cache_key = follow_key
            self.follow_cache = {'chosen': chosen, 'candidate': candidate, 'decision': decision}
        decision_event = self.write_decision_event({
            'phase': 'follow',
            'action': 'pass' if is_pass else 'play',
            'role': self.role,
            'hand': rules_hand,
            'target': {'seat': target_seat, 'ranks': target},
            'chosen': candidate.get('id'),
            'chosen_ranks': candidate.get('ranks'),
            'selected_indices': chosen,
            'reason': decision.get('reason'),
            'utterance': decision.get('utterance'),
            'source': source,
            'execution': 'decided',
            'speak': source != 'cached_decision',
        })

        if is_pass:
            clicked = self.click(hwnd, bbox, pass_point, scale_x, scale_y, physical=True)
            self.write_execution_result(
                decision_event,
                'submitted' if clicked else 'failed',
                '不出按钮未响应点击，本次过牌未执行，准备重试。',
            )
            self.last_action = time.perf_counter()
            self.log({'state': 'follow_submit', 'action': 'pass', 'click': pass_point})
            self.notify('【执行】不出', 'warn')
            return
        if self.check_tuoguan(hwnd, bbox, scale_x, scale_y):
            self.write_execution_result(decision_event, 'failed', '检测到托管接管操作，本次跟牌暂停，需要先取消托管。')
            return
        if self.set_selection(hwnd, bbox, observation, chosen, scale_x, scale_y,
                              limit=self.selection_batch) is False:
            self.write_execution_result(decision_event, 'failed', '目标牌选择操作未完成，本次跟牌未执行，准备重新识别。')
            return
        time.sleep(0.45)
        if self.check_tuoguan(hwnd, bbox, scale_x, scale_y):
            self.write_execution_result(decision_event, 'failed', '检测到托管接管操作，本次跟牌暂停，需要先取消托管。')
            return
        rgb2, _ = self.capture(hwnd)
        confirmed, rgb2 = self.confirm_selection(hwnd, bbox, rgb2, chosen, scale_x, scale_y, before=observation)
        if not confirmed:
            self.write_execution_result(decision_event, 'failed', '选牌点击未全部生效，改用游戏提示。')
            self.notify('【选牌】跟牌未选全，改用提示', 'warn')
            return self.fallback_hint(hwnd, bbox, scale_x, scale_y, ui, announce=False,
                                      target_seat=target_seat)
        _, _, ui2 = classify(rgb2, scale_x, scale_y)
        play_point = ui2['orange'][0]['center'] if ui2 and ui2['orange'] else orange
        clicked = self.click(hwnd, bbox, play_point, scale_x, scale_y, physical=True)
        self.last_action = time.perf_counter()
        if not clicked:
            self.write_execution_result(decision_event, 'failed', '出牌按钮未响应点击，本次跟牌未执行，准备重试。')
            return
        self.log({'state': 'follow_submit', 'action': 'play', 'indices': chosen, 'click': play_point})
        time.sleep(0.6)
        rgb3, _ = self.capture(hwnd)
        state3, _, _ = classify(rgb3, scale_x, scale_y)
        hand_before = observation['count']
        hand_after = count_hand_in_frame(rgb3)
        expected_remaining = hand_before - len(candidate.get('ranks') or [])
        execution = evaluate_play_execution(state3, 'turn', hand_before, hand_after, expected_remaining)
        self.log({'state': 'follow_execution', 'execution': execution, 'hand_before': hand_before,
                  'hand_after': hand_after, 'expected_remaining': expected_remaining, 'state_after': state3})
        if execution == 'confirmed':
            self.played.extend(candidate.get('ranks') or [])
            self.last_play = None
            self.write_execution_result(decision_event, execution)
            self.notify(f"【执行】跟牌 {' '.join(candidate.get('ranks') or [])}")
            return
        if execution == 'uncertain':
            self.save_action_frame(rgb3, 'follow_uncertain')
            self.write_execution_result(decision_event, execution)
            self.notify('【执行状态】界面已变化，但手牌证据不足，等待确认', 'warn')
            return
        self.save_action_frame(rgb3, 'follow_rejected')
        retry_key = (target_seat, tuple(target), tuple(candidate.get('ranks') or []))
        if self.follow_retry_key != retry_key:
            self.follow_retry_key = retry_key
            self.notify('【执行状态】出牌提交未确认，同手牌原样重试一次', 'warn')
            retry_rgb, _ = self.capture(hwnd)
            try:
                retry_observation = mini_perceive.read_hand(cv2.cvtColor(retry_rgb, cv2.COLOR_RGB2BGR))
            except ValueError:
                retry_observation = None
            resubmitted = False
            if retry_observation is not None and self.set_selection(
                    hwnd, bbox, retry_observation, chosen, scale_x, scale_y, limit=self.selection_batch):
                resubmitted = self.click(hwnd, bbox, orange, scale_x, scale_y, physical=True)
            if resubmitted:
                self.last_action = time.perf_counter()
                time.sleep(0.65)
                retry_after, _ = self.capture(hwnd)
                retry_state, _, _ = classify(retry_after, scale_x, scale_y)
                execution = evaluate_play_execution(
                    retry_state, 'turn', hand_before, count_hand_in_frame(retry_after), expected_remaining)
                self.log({'state': 'follow_retry', 'execution': execution, 'state_after': retry_state,
                          'candidate': candidate.get('id'), 'ranks': candidate.get('ranks')})
                if execution == 'confirmed':
                    self.follow_retry_key = None
                    self.played.extend(candidate.get('ranks') or [])
                    self.last_play = None
                    self.write_execution_result(decision_event, execution)
                    self.notify(f"【执行】重试后跟牌 {' '.join(candidate.get('ranks') or [])}")
                    return
                if execution == 'uncertain':
                    self.save_action_frame(retry_after, 'follow_uncertain')
                    self.write_execution_result(decision_event, execution)
                    self.notify('【执行状态】重试后界面已变化，但手牌证据不足，等待确认', 'warn')
                    return
                execution = 'rejected'
        self.write_execution_result(
            decision_event,
            execution,
            '跟牌仍未确认生效，可能因上家牌型识别不完整，改用游戏提示。',
        )
        self.log({'state': 'follow_rejected', 'target': target, 'candidate': candidate.get('id')})
        self.notify('【执行状态】跟牌未确认生效，已补充失败原因，改用提示', 'error')
        return self.fallback_hint(hwnd, bbox, scale_x, scale_y, ui, announce=False,
                                  target_seat=target_seat)

    def fallback_hint(self, hwnd, bbox, scale_x, scale_y, ui, announce=True, target_seat=None):
        def decide(action, utterance, speak=True):
            return self.write_decision_event({
                'phase': 'follow',
                'action': action,
                'role': self.role,
                'hand': [],
                'target': None,
                'chosen': 'game_hint' if action == 'play' else action,
                'chosen_ranks': [],
                'reason': '目标或手牌识别不可靠，使用游戏内提示',
                'utterance': utterance,
                'source': 'local_fallback',
                'execution': 'decided',
                'speak': speak,
            })

        orange = ui['orange'][0]['center']
        hint = [orange[0] - 111, orange[1]]
        hint_source = 'legacy_offset'
        try:
            hint_rgb, _ = self.capture(hwnd)
            hint_button = self.find_hint_button(hint_rgb, scale_x, scale_y)
            if hint_button is not None:
                hint = hint_button['center']
                hint_source = 'rapidocr_button'
        except Exception as error:
            self.log({'state': 'hint_ocr_failed', 'error': str(error)})
        self.log({'state': 'hint_located', 'point_source': hint_source, 'click': hint})
        hint_event = decide('play', '目标或手牌识别不可靠，改用游戏提示。' if announce else '', announce)
        if not self.click(hwnd, bbox, hint, scale_x, scale_y, physical=True):
            self.write_execution_result(hint_event, 'failed', '游戏提示按钮未响应点击，准备重新识别当前回合。')
            return
        self.log({'state': 'turn', 'action': 'hint', 'click': hint, 'point_source': hint_source})
        time.sleep(0.45)
        if not self.click(hwnd, bbox, orange, scale_x, scale_y, physical=True):
            self.write_execution_result(hint_event, 'failed', '游戏提示按钮未响应点击，准备重新识别当前回合。')
            return
        self.log({'state': 'turn', 'action': 'submit', 'click': orange})
        self.last_action = time.perf_counter()
        time.sleep(0.5)
        rgb2, _ = self.capture(hwnd)
        state2, _, ui2 = classify(rgb2, scale_x, scale_y)
        if state2 == 'turn' and ui2 and ui2['orange']:
            submit_point = ui2['orange'][0]['center']
            self.log({'state': 'turn', 'action': 'submit_retry', 'click': submit_point})
            if self.click(hwnd, bbox, submit_point, scale_x, scale_y, physical=True):
                self.last_action = time.perf_counter()
                time.sleep(0.5)
                rgb3, _ = self.capture(hwnd)
                state3, _, ui3 = classify(rgb3, scale_x, scale_y)
                if not (state3 == 'turn' and ui3 and ui3['orange']):
                    self.write_execution_result(hint_event, 'confirmed')
                    return
            pass_point = [submit_point[0] - 328, submit_point[1]]
            pass_event = decide('pass', '游戏提示未生成可出的牌，选择过牌。')
            clicked = self.click(hwnd, bbox, pass_point, scale_x, scale_y, physical=True)
            self.write_execution_result(
                pass_event,
                'submitted' if clicked else 'failed',
                '过牌按钮未响应点击，本次过牌未执行，准备重试。',
            )
            self.log({'state': 'turn_fallback_pass', 'click': pass_point})
            self.last_action = time.perf_counter()
        else:
            self.write_execution_result(hint_event, 'confirmed')

    def handle_result(self, hwnd, bbox, rgb, scale_x, scale_y):
        outcome = detect_result_outcome(rgb, scale_x, scale_y)
        outcome_label = {'win': '胜利', 'loss': '失败'}.get(outcome)
        self.bid_done = False
        self.bid_complete = False
        self.bid_last = None
        self.bid_clicks = {}
        self.lead_failure_counts = {}
        self.lead_blocked_key = None
        self.hand_id = None
        self.role = None
        self.landlord_seat = None
        self.landlord_count_adjusted = False
        self.lead_cache_key = None
        self.lead_cache = None
        self.follow_cache_key = None
        self.follow_cache = None
        self.counts = {'left': 17, 'right': 17}
        self.has_played = {'left': False, 'right': False}
        self.badge_at = {'left': 0.0, 'right': 0.0}
        self.play_at = {'left': 0.0, 'right': 0.0}
        self.badge_counts = {'left': None, 'right': None}
        self.played_total = {'left': 0, 'right': 0}
        self.badge_last_read = {'left': None, 'right': None}
        self.played = []
        self.play_state = {'left': (), 'right': ()}
        self.play_candidate = {'left': (), 'right': ()}
        self.play_candidate_at = {'left': 0.0, 'right': 0.0}
        self.play_unreliable = {'left': (), 'right': ()}
        self.play_unreliable_candidate = {'left': (), 'right': ()}
        self.unreliable_since = {'left': 0.0, 'right': 0.0}
        self.last_play = None
        self.follow_retry_key = None
        self.last_hand_count = None
        played = self.hand_active
        self.hand_active = False
        if played:
            self.hands_done += 1
        if played and self.hand_limit and self.hands_done >= self.hand_limit:
            result_text = f'本局结果：{outcome_label}，' if outcome_label else '本局结束，'
            self.scene_event({
                'phase': 'result', 'action': f'game_{outcome}' if outcome else 'hand_limit_reached',
                'utterance': result_text + '已达到试玩局数，停在结算页。',
                'reason': '对局结算；已达到指定试玩局数',
                'outcome': outcome, 'source': 'scene_vision',
            }, f'对局结果{outcome_label or "未知"}', 'finished')
            self.log({'state': 'hand_limit_reached', 'hands': self.hands_done, 'outcome': outcome})
            return True
        result_text = f'本局结果：{outcome_label}，' if outcome_label else '本局结束，'
        rgb2, _ = self.capture(hwnd)
        state2, _, _ = classify(rgb2, scale_x, scale_y)
        if state2 not in {'result', 'result_candidate'}:
            self.log({'state': 'result_page_changed', 'state_after': state2})
            return False
        point = find_continue_point(rgb2, scale_x, scale_y) or STATIC_POINTS['continue']
        self.perform_scene_action(hwnd, bbox, scale_x, scale_y, {
            'phase': 'result', 'action': 'continue_game',
            'utterance': result_text + '结算后继续游戏。',
            'reason': '对局结算后继续游戏', 'point': point,
            'outcome': outcome, 'source': 'scene_vision',
        }, f'对局结果{outcome_label or "未知"}')
        time.sleep(0.8)
        return False

    def recheck_pass_only(self, hwnd, scale_x, scale_y, attempts=2):
        for attempt in range(attempts):
            time.sleep(0.3)
            rgb, _ = self.capture(hwnd)
            state, _, ui = classify(rgb, scale_x, scale_y)
            self.log({'state': 'pass_only_retry', 'attempt': attempt + 1, 'state_after': state})
            if state != 'pass_only':
                return None
        return ui

    def run(self, seconds):
        deadline = time.perf_counter() + seconds
        while time.perf_counter() < deadline:
            self.check_takeover_hotkey()
            if self.overlay is not None:
                self.overlay.update()
            if self.paused:
                time.sleep(0.1)
                continue
            hwnd = self.window()
            if hwnd is None:
                time.sleep(0.5)
                continue
            if self.overlay is not None:
                self.overlay.update()
            if self.user32.IsIconic(hwnd):
                self.user32.ShowWindow(hwnd, 4)
                time.sleep(0.35)
            try:
                rgb, bbox = self.capture(hwnd)
            except CaptureError:
                time.sleep(0.3)
                continue
            self.sync_overlay(hwnd)
            if rgb.shape[1] < 400 or rgb.shape[0] < 300:
                time.sleep(0.3)
                continue
            scale_x = rgb.shape[1] / BASE_WIDTH
            scale_y = rgb.shape[0] / BASE_HEIGHT
            size = [bbox[2] - bbox[0], bbox[3] - bbox[1]]
            self.last_client_size = size
            if size[0] < BASE_WIDTH or size[1] < BASE_HEIGHT:
                if not self.small_client_warned:
                    self.small_client_warned = True
                    self.log({'state': 'client_too_small', 'size': size, 'minimum': [BASE_WIDTH, BASE_HEIGHT]})
                    self.notify('【窗口】客户区小于 933×580，游戏画面会被裁切，已停手；请把窗口放大', 'error')
                time.sleep(1.0)
                continue
            self.small_client_warned = False
            try:
                self.track_plays(rgb)
                state, values, ui = classify(rgb, scale_x, scale_y)
                if state in {'lead', 'turn'}:
                    self.update_role_from_table(rgb)
                if state in {'warning_dialog', 'reward', 'result', 'room'} and not self.ensure_foreground(hwnd):
                    self.log({'state': state, 'skipped': 'foreground_failed'})
                    time.sleep(0.3)
                    continue
                tuoguan = self.tuoguan_score(rgb, scale_x, scale_y)
                if tuoguan > 0.7:
                    if self.ensure_foreground(hwnd):
                        self.click(hwnd, bbox, TUOGUAN_POINT, scale_x, scale_y, physical=True)
                        self.log({'state': 'cancel_tuoguan', 'click': TUOGUAN_POINT, 'match': round(tuoguan, 3)})
                        self.notify(f'【托管】检测到取消托管，已点掉（匹配 {tuoguan:.2f}）', 'error')
                        self.last_action = time.perf_counter()
                        time.sleep(0.6)
                    self.state = None
                    continue
                now = time.perf_counter()
                if now - self.last_hand_probe > 1.0:
                    self.last_hand_probe = now
                    try:
                        detection = mini_perceive.detect_hand(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                    except ValueError:
                        detection = None
                    if detection:
                        count = detection['count']
                        if (self.hand_active and self.last_hand_count is not None
                                and count < self.last_hand_count and now - self.last_action > 5.0
                                and state in {'turn', 'lead', 'pass_only'}):
                            stamp = datetime.now().strftime('%H%M%S')
                            frame_path = ROOT / 'outputs' / f'tuoguan_auto_{stamp}.png'
                            frame = print_window_image(hwnd, bbox[2] - bbox[0], bbox[3] - bbox[1])
                            if frame is not None:
                                frame.save(frame_path)
                            self.log({'state': 'auto_play_detected', 'before': self.last_hand_count, 'after': count, 'frame': str(frame_path)})
                            self.notify('【托管】手牌被系统打掉，尝试取消托管', 'error')
                            if self.ensure_foreground(hwnd):
                                self.click(hwnd, bbox, TUOGUAN_POINT, scale_x, scale_y, physical=True)
                                self.log({'state': 'cancel_tuoguan_guess', 'click': TUOGUAN_POINT})
                                self.last_action = time.perf_counter()
                                time.sleep(0.6)
                        self.last_hand_count = count
                if state != self.state:
                    self.state = state
                    self.state_since = time.perf_counter()
                    time.sleep(self.interval)
                    continue
                if time.perf_counter() - self.state_since < 0.3:
                    time.sleep(0.08)
                    continue
                if time.perf_counter() - self.last_action < 0.55:
                    time.sleep(0.08)
                    continue
                if state == 'warning_dialog':
                    self.handle_dialog_scene(hwnd, bbox, rgb, scale_x, scale_y)
                elif state == 'reward':
                    self.perform_scene_action(hwnd, bbox, scale_x, scale_y, {
                        'phase': 'reward', 'action': 'claim_reward',
                        'utterance': '有奖励，先领一下。', 'reason': '领取对局奖励',
                        'point': STATIC_POINTS['claim'],
                    }, '领取奖励')
                elif state == 'result':
                    if self.handle_result(hwnd, bbox, rgb, scale_x, scale_y):
                        return
                elif state == 'result_candidate':
                    try:
                        text = self.read_scene_ocr(rgb).get('text', '')
                    except Exception as error:
                        text = ''
                        self.log({'state': 'scene_ocr_failed', 'error': str(error), 'scene': 'result_candidate'})
                    if confirm_result_scene(text):
                        if self.handle_result(hwnd, bbox, rgb, scale_x, scale_y):
                            return
                    else:
                        plan = dialog_scene_plan(text)
                        if plan['action'] != 'dismiss_dialog':
                            self.perform_scene_action(hwnd, bbox, scale_x, scale_y, plan, text)
                        elif not self.hand_active:
                            plan = idle_scene_plan(text)
                            if plan['action'] != 'unknown_scene':
                                self.perform_scene_action(hwnd, bbox, scale_x, scale_y, plan, text)
                            else:
                                plan = dict(plan, action='unknown_result_page',
                                            utterance='结算页信息还没看全，先不乱点，找老师问问怎么办。')
                                self.perform_scene_action(hwnd, bbox, scale_x, scale_y, plan, text)
                        else:
                            self.log({'state': 'result_candidate_unconfirmed', 'text': compact_scene_text(text)[:200]})
                elif state == 'turn':
                    if self.ensure_foreground(hwnd):
                        self.handle_turn(hwnd, bbox, scale_x, scale_y, ui)
                elif state == 'lead':
                    if self.ensure_foreground(hwnd):
                        self.handle_lead(hwnd, bbox, scale_x, scale_y, ui)
                elif state == 'pass_only':
                    confirmed_ui = self.recheck_pass_only(hwnd, scale_x, scale_y)
                    if confirmed_ui is None:
                        continue
                    if self.ensure_foreground(hwnd):
                        clock = confirmed_ui['clock'][0]['center']
                        point = [clock[0] + 105, clock[1]]
                        decision_event = self.write_decision_event({
                            'phase': 'follow',
                            'action': 'pass',
                            'role': self.role,
                            'hand': [],
                            'target': None,
                            'chosen': 'cant_beat_pass',
                            'chosen_ranks': [],
                            'reason': '当前没有能压过上家的牌',
                            'utterance': '当前没有能压过上家的牌，选择过牌。',
                            'source': 'local_engine',
                            'execution': 'decided',
                        })
                        clicked = self.click(hwnd, bbox, point, scale_x, scale_y, physical=True)
                        self.write_execution_result(
                            decision_event,
                            'submitted' if clicked else 'failed',
                            '过牌按钮未响应点击，本次过牌未执行，准备重试。',
                        )
                        self.log({'state': 'pass_only', 'action': 'cant_beat_pass', 'click': point})
                        self.notify('【要不起】没有能压的牌，已过', 'warn')
                        self.last_action = time.perf_counter()
                        time.sleep(0.5)
                elif state == 'bid':
                    if self.ensure_foreground(hwnd):
                        self.handle_bid(hwnd, bbox, scale_x, scale_y, ui)
                elif state == 'room':
                    self.bid_done = False
                    self.bid_complete = False
                    self.bid_last = None
                    self.bid_clicks = {}
                    self.bid_click_keys = {}
                    self.bid_button_cache = None
                    self.lead_failure_counts = {}
                    self.lead_blocked_key = None
                    self.hand_id = None
                    self.role = None
                    self.landlord_seat = None
                    self.landlord_count_adjusted = False
                    self.lead_cache_key = None
                    self.lead_cache = None
                    self.follow_cache_key = None
                    self.follow_cache = None
                    self.counts = {'left': 17, 'right': 17}
                    self.has_played = {'left': False, 'right': False}
                    self.badge_at = {'left': 0.0, 'right': 0.0}
                    self.play_at = {'left': 0.0, 'right': 0.0}
                    self.badge_counts = {'left': None, 'right': None}
                    self.badge_last_read = {'left': None, 'right': None}
                    self.played_total = {'left': 0, 'right': 0}
                    self.played = []
                    self.play_state = {'left': (), 'right': ()}
                    self.play_candidate = {'left': (), 'right': ()}
                    self.play_candidate_at = {'left': 0.0, 'right': 0.0}
                    self.play_unreliable = {'left': (), 'right': ()}
                    self.play_unreliable_candidate = {'left': (), 'right': ()}
                    self.unreliable_since = {'left': 0.0, 'right': 0.0}
                    self.last_play = None
                    self.follow_retry_key = None
                    self.last_hand_count = None
                    self.perform_scene_action(hwnd, bbox, scale_x, scale_y, {
                        'phase': 'room', 'action': 'start_game',
                        'utterance': '准备完毕，开始游戏。',
                        'reason': '准备房开始游戏', 'point': STATIC_POINTS['start'],
                    }, '开始游戏')
                elif state == 'room_select':
                    self.handle_idle_scene(hwnd, bbox, rgb, scale_x, scale_y)
                elif state == 'idle':
                    self.handle_idle_scene(hwnd, bbox, rgb, scale_x, scale_y)
                self.state = None
            except Exception as error:
                self.log({'state': 'loop_error', 'error': str(error)})
                self.notify(f'【异常】{error}', 'error')
                self.state = None
                time.sleep(0.5)
            time.sleep(self.interval)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds', type=float, default=1800)
    parser.add_argument('--method', choices=['message', 'physical'], default='message')
    parser.add_argument('--interval', type=float, default=0.2)
    parser.add_argument('--hands', type=int, default=0)
    parser.add_argument('--no-overlay', action='store_true')
    parser.add_argument('--trace-selection', action='store_true')
    args = parser.parse_args()
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        ctypes.windll.user32.SetProcessDPIAware()
    overlay = None
    if not args.no_overlay:
        from overlay import Overlay
        overlay = Overlay()
        try:
            _, (_, _, rect) = find_window('腾讯欢乐斗地主', 'Chrome_WidgetWin_0')
            overlay.place_outside((rect.left, rect.top, rect.right, rect.bottom))
        except RuntimeError:
            pass
    bot = Bot(args.method, args.interval, args.hands, overlay)
    bot.selection_trace = args.trace_selection
    deadline = time.perf_counter() + args.seconds
    while True:
        try:
            bot.run(max(0.0, deadline - time.perf_counter()))
            break
        except KeyboardInterrupt:
            break
        except Exception as error:
            print(json.dumps({'state': 'fatal_recovered', 'error': str(error)}, ensure_ascii=True), flush=True)
            if overlay is not None:
                overlay.push(f'【异常】{error}｜已自动恢复', 'error')
            if time.perf_counter() >= deadline:
                break
            time.sleep(1.0)


if __name__ == '__main__':
    main()
