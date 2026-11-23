import json
import subprocess
from pathlib import Path

import cv2

import mini_bot
import mini_perceive

ROOT = Path(__file__).resolve().parent
RESULTS = []


def check(name, condition, detail=''):
    RESULTS.append({'check': name, 'pass': bool(condition), 'detail': detail})


def fixture_path(path):
    candidate = ROOT / 'samples' / 'selftest' / Path(path).name
    if str(path).startswith('outputs/') and candidate.exists():
        return candidate
    return ROOT / path


def load_rgb(path):
    return cv2.cvtColor(mini_perceive.load_image(fixture_path(path)), cv2.COLOR_BGR2RGB)


def classify_cases():
    cases = [
        ('outputs/mini2_bid_after.png', 'turn'),
        ('outputs/mini2_deal1.png', 'bid'),
        ('outputs/mini_live_now12.png', 'lead'),
        ('outputs/mini_bot_check3.png', 'lead'),
        ('outputs/mini2_table_wait.png', 'room'),
        ('outputs/mini_live_now.png', 'result'),
        ('outputs/mini2_restored.png', 'reward'),
        ('outputs/mini2_after_post.png', 'room_select'),
        ('outputs/mini_live_now4.png', 'idle'),
        ('outputs/run_final2.png', 'pass_only'),
        ('outputs/hand10_state.png', 'room_select'),
        ('outputs/mini2_restored2.png', 'warning_dialog'),
        ('outputs/live_trial4_current.png', 'result_candidate'),
        ('outputs/live_bean_scene.png', 'result_candidate'),
        ('outputs/live_current_lobby.png', 'result_candidate'),
        ('outputs/result_audit.png', 'result_candidate'),
    ]
    for path, expected in cases:
        rgb = load_rgb(path)
        sx = rgb.shape[1] / mini_bot.BASE_WIDTH
        sy = rgb.shape[0] / mini_bot.BASE_HEIGHT
        state, _, ui = mini_bot.classify(rgb, sx, sy)
        check(f'classify {path}', state == expected, f'{state} != {expected}')
    rgb = load_rgb('outputs/mini2_bid_after.png')
    sx = rgb.shape[1] / mini_bot.BASE_WIDTH
    sy = rgb.shape[0] / mini_bot.BASE_HEIGHT
    _, _, ui = mini_bot.classify(rgb, sx, sy)
    orange = ui['orange'][0]['center']
    hint = [orange[0] - 111, orange[1]]
    guard = [orange[0] - 328, orange[1]]
    check('turn hint point', abs(hint[0] - 533) <= 20 and abs(hint[1] - 333) <= 15, str(hint))
    check('turn guard point', abs(guard[0] - 316) <= 20 and abs(guard[1] - 333) <= 15, str(guard))
    rgb = load_rgb('outputs/mini2_deal1.png')
    _, _, ui = mini_bot.classify(rgb, sx, sy)
    orange = ui['orange'][0]['center']
    check('bid pass point', abs(orange[0] + 230 - 604) <= 20, str(orange))


def recognition_cases():
    import numpy as np

    expected = {
        'outputs/mini2_bid_after.png': ['LJ', '2', '2', '2', 'K', 'K', 'K', 'K', 'Q', 'Q', 'J', '9', '9', '9', '9', '7', '7'],
        'outputs/mini_bot_check2.png': ['LJ', 'A', 'A', 'A', 'A', 'K', 'K', 'K', 'K', '7', '7', '7', '7'],
        'outputs/mini_bot_check3.png': ['LJ', 'A', 'A', 'A', 'A', 'K', 'K', 'K', 'K'],
        'outputs/mini_live_now2.png': ['BJ', 'LJ', '2', '2', 'A', 'Q', 'Q', 'Q', 'J', 'J', 'J', '10', '10', '10', '7', '7', '7', '6', '5', '5'],
        'outputs/mini_live_now6.png': ['BJ', 'LJ', '2', '2', 'K', 'K', '10', '8', '8', '7', '7', '6', '6', '5', '4', '4', '3'],
        'outputs/lead_uncertain_153343_114198.png': ['A', 'K', 'J', 'J', '10', '9', '7', '7', '6', '6', '3', '3'],
        'outputs/hand_unreliable_162641_810824.png': ['LJ', 'A', 'A', 'A', 'K', 'K', 'K', '10', '9', '6', '6', '6', '6', '5', '5', '4', '3'],
        'outputs/hand_unreliable_173516_883162.png': ['K', 'K', 'Q', 'Q', '10', '10', '7', '7', '7', '7', '5', '4', '4', '4', '3'],
    }
    templates = mini_perceive.load_templates()
    for path, ranks in expected.items():
        observation = mini_perceive.read_hand(mini_perceive.load_image(fixture_path(path)), templates)
        check(f'hand {path}', observation['my_hand'] == ranks, str(observation['my_hand']))
    shape = mini_perceive.glyph_shape(templates['K'])
    red_k = np.full_like(templates['K'], 255)
    red_k[shape > 0] = (0, 0, 255)
    red_score, red_rank = mini_perceive.match_glyph(red_k, templates)[0]
    check('ordinary ranks use color independent shapes',
          red_rank == 'K' and red_score > 0.99, f'{red_rank} {red_score}')
    check('joker matching retains color evidence',
          all(mini_perceive.match_glyph(templates[rank], templates)[0][1] == rank for rank in ('BJ', 'LJ')), '')


def hand_read_retry_cases():
    from unittest.mock import patch
    import numpy as np

    def observation(unknown):
        return {'my_hand': ['A', 'K', 'Q'], 'unknown_indices': unknown}

    first = np.zeros((580, 933, 3), dtype=np.uint8)
    second = np.full((580, 933, 3), 1, dtype=np.uint8)
    for name, readings, expected_second in [
        ('unreliable hand is captured once more', [observation([0, 1]), observation([])], True),
        ('hand read exception is retried', [ValueError('no glyphs'), observation([])], True),
        ('two unreliable reads keep the better observation', [observation([0, 1]), observation([0, 1, 2])], False),
    ]:
        bot = mini_bot.Bot()
        captures = iter([(first, None), (second, None)])
        bot.capture = lambda hwnd: next(captures)
        logs = []
        bot.log = logs.append
        with patch.object(mini_perceive, 'read_hand', side_effect=readings):
            rgb, actual = bot.read_hand_with_retry(None, delay=0)
        check(name, len(logs) >= 2 and (rgb is second) == expected_second
              and actual['unknown_indices'] == ([] if expected_second else [0, 1])
              and logs[0]['state'] == 'hand_read_retry', str(logs))


def rules_candidates(hand):
    request = {'my_hand': hand, 'selected_ranks': [], 'selected_indices': [], 'last_play': None}
    (ROOT / 'outputs' / 'selftest_rule_input.json').write_text(json.dumps(request, ensure_ascii=False), encoding='utf-8')
    subprocess.run(
        ['node', 'rules_cli.mjs', 'outputs/selftest_rule_input.json', 'outputs/selftest_rule_output.json'],
        cwd=str(ROOT), check=True, capture_output=True, timeout=10,
    )
    rules = json.loads((ROOT / 'outputs' / 'selftest_rule_output.json').read_text(encoding='utf-8'))
    return [candidate for candidate in rules['candidates'] if candidate['id'] != 'pass']


def strategy_cases():
    hand = ['3', '3', '3', '3', '4', '5', '6', '7', '8', 'BJ', 'LJ']
    candidate = mini_bot.pick_lead(hand, rules_candidates(hand))
    check('keep rocket and bomb', not ({'BJ', 'LJ'} & set(candidate['ranks'])) and candidate['move']['type'] in {'straight', 'bomb'}, str(candidate['ranks']))

    hand = ['3', '4', '5', '6', '7']
    candidate = mini_bot.pick_lead(hand, rules_candidates(hand))
    check('finish in one move', len(candidate['ranks']) == 5, str(candidate['ranks']))

    hand = ['2', '2', '9', '9', '4', '3']
    candidate = mini_bot.pick_lead(hand, rules_candidates(hand))
    check('do not lead pair of 2s early', '2' not in candidate['ranks'], str(candidate['ranks']))

    hand = ['BJ', 'LJ', '2', '2', 'A', 'A', 'A', 'A', '10', '10', '9', '7', '5', '5', '4', '4', '3']
    candidate = mini_bot.pick_lead(hand, rules_candidates(hand))
    check('lead from strong hand is legal', len(candidate['ranks']) > 0, str(candidate['ranks']))
    check('lead from strong hand keeps bombs', 'A' * 4 not in ''.join(candidate['ranks']), str(candidate['ranks']))


def bot_engine(phase, hand, role='farmer', can_pass=False, target=None, target_seat=None):
    request = {'phase': phase, 'difficulty': 'hell', 'hand': hand, 'role': role,
               'target': target or [], 'targetSeat': target_seat, 'canPass': can_pass,
               'handCounts': {'left': 17, 'right': 17}}
    (ROOT / 'outputs' / 'selftest_bot_in.json').write_text(json.dumps(request, ensure_ascii=False), encoding='utf-8')
    subprocess.run(
        ['node', 'bot_cli.mjs', 'outputs/selftest_bot_in.json', 'outputs/selftest_bot_out.json'],
        cwd=str(ROOT), check=True, capture_output=True, timeout=10,
    )
    return json.loads((ROOT / 'outputs' / 'selftest_bot_out.json').read_text(encoding='utf-8'))


def bot_candidates(hand, role='landlord'):
    request = {'phase': 'playing', 'mode': 'candidates', 'difficulty': 'hell', 'hand': hand, 'role': role,
               'target': [], 'targetSeat': None, 'canPass': False,
               'handCounts': {'left': 17, 'right': 17}, 'candidateLimit': 6}
    (ROOT / 'outputs' / 'selftest_cand_in.json').write_text(json.dumps(request, ensure_ascii=False), encoding='utf-8')
    subprocess.run(
        ['node', 'bot_cli.mjs', 'outputs/selftest_cand_in.json', 'outputs/selftest_cand_out.json'],
        cwd=str(ROOT), check=True, capture_output=True, timeout=10,
    )
    return json.loads((ROOT / 'outputs' / 'selftest_cand_out.json').read_text(encoding='utf-8'))


def candidate_cases():
    strong = ['BJ', 'LJ', '2', '2', 'A', 'A', 'A', 'A', '10', '10', '9', '7', '5', '5', '4', '4', '3']
    result = bot_candidates(strong)
    shortlist = result.get('candidates', [])
    check('candidate shortlist size', len(shortlist) == 6, str(len(shortlist)))
    check('candidate recommended is in list', result.get('recommended') in [item['id'] for item in shortlist], str(result.get('recommended')))
    check('local score shortlist keeps bombs but allows the official whole-bomb lead',
          not any(item.get('uses_bomb') for item in shortlist if item['id'] != result.get('recommended')),
          str([item['type'] for item in shortlist]))
    check('candidate reasons present', all(item.get('reasons') for item in shortlist), '')
    check('candidate indices valid', all(all(isinstance(i, int) and 0 <= i < len(strong) for i in item['indices']) for item in shortlist), '')


def bot_engine_cases():
    strong = ['BJ', 'LJ', '2', '2', 'A', 'A', 'A', 'A', '10', '10', '9', '7', '5', '5', '4', '4', '3']
    result = bot_engine('bidding', strong)
    check('vendor bid strong hand', result['action']['score'] >= 2, str(result))
    result = bot_engine('playing', strong, role='landlord')
    action = result['action']
    check('vendor lead is legal', action['type'] == 'play' and action['indices'], str(action.get('ranks')))
    check('vendor lead keeps rocket', not ({'BJ', 'LJ'} & set(action.get('ranks', []))), str(action.get('ranks')))
    finishing = ['3', '4', '5', '6', '7']
    result = bot_engine('playing', finishing, role='landlord')
    check('vendor finishes in one move', sorted(result['action'].get('ranks', [])) == finishing, str(result['action'].get('ranks')))
    farmer = ['2', '2', '9', '9', '4', '3']
    result = bot_engine('playing', farmer, role='farmer')
    check('vendor farmer avoids early 2s', '2' not in result['action'].get('ranks', []), str(result['action'].get('ranks')))
    multi_bomb = ['2', '2', 'A', 'A', 'A', 'A', 'Q', 'Q', 'Q', 'Q', '6', '6', '4', '4', '4', '4', '3', '3', '3', '3']
    result = bot_engine('playing', multi_bomb, role='landlord')
    check('lead uses the smallest whole bomb and a pair instead of breaking other bombs',
          result['action'].get('ranks') == ['3', '3', '3', '3', '6', '6'], str(result['action'].get('ranks')))
    result = bot_candidates(multi_bomb)
    recommended = next((item for item in result['candidates'] if item['id'] == result.get('recommended')), None)
    check('official lead remains in candidates even outside local score shortlist',
          result.get('recommended_source') == 'bot.js' and recommended is not None
          and recommended['ranks'] == ['3', '3', '3', '3', '6', '6'], str(result.get('recommended')))
    result = bot_engine('playing', strong, role='landlord')
    check('four with two pairs remains legal when it does not break another bomb',
          result['action'].get('ranks') == ['A', 'A', 'A', 'A', '4', '4', '5', '5']
          and result['action'].get('move', {}).get('type') == 'four_two_pairs', str(result['action'].get('ranks')))


def follow_bomb_wing_cases():
    hand = ['BJ', '2', '2', '2', 'K', 'K', 'K', 'Q', 'Q', 'Q', '9', '9', '5', '3', '3', '3', '3']
    bot = mini_bot.Bot()
    bot.role = 'farmer'
    bot.landlord_seat = 'left'
    bot.counts = {'left': 16, 'right': 17}
    bot.played = ['10', '10', '10', '7']
    result = bot.bot_decision('playing', hand, role='farmer', target=['10', '10', '10', '7'],
                              target_seat='left', can_pass=True, counts=bot.counts, mode='candidates')
    check('follow includes a wing that does not break the bomb',
          any(item['ranks'] == ['Q', 'Q', 'Q', '5'] for item in result['candidates']), str(result.get('recommended')))
    recommended = next((item for item in result['candidates'] if item['id'] == result.get('recommended')), None)
    check('follow penalty and exact action matching choose QQQ with the safe wing',
          result.get('recommended_source') == 'bot.js' and recommended is not None
          and recommended['ranks'] == ['Q', 'Q', 'Q', '5']
          and result['action'].get('ranks') == recommended['ranks'], str(result.get('recommended')))


def landlord_cases():
    check('landlord badge on right', mini_perceive.detect_landlord(mini_perceive.load_image(fixture_path('outputs/mini_bot_check2.png'))) == 'right', '')
    check('no landlord badge', mini_perceive.detect_landlord(mini_perceive.load_image(fixture_path('outputs/mini_live_now2.png'))) is None, '')


def play_tracking_cases():
    import time
    bot = mini_bot.Bot()

    def feed(path):
        rgb = load_rgb(path)
        bot.track_plays(rgb)
        time.sleep(0.3)
        bot.track_plays(rgb)
        time.sleep(0.3)

    feed('outputs/mini_bot_check2.png')
    check('track right play', bool(bot.last_play) and bot.last_play['seat'] == 'right' and bot.last_play['ranks'] == ['2', '2'], str(bot.last_play))
    feed('outputs/run_final2.png')
    check('track latest left play', bool(bot.last_play) and bot.last_play['seat'] == 'left' and bot.last_play['ranks'] == ['9'], str(bot.last_play))


def alignment_cases():
    image = mini_perceive.load_image(fixture_path('outputs/mini2_bid_after.png'))
    templates = mini_perceive.load_templates()
    baseline = mini_perceive.read_hand(image, templates)['my_hand']
    original = mini_perceive.glyph_boxes
    try:
        boxes = original(image)
        mini_perceive.glyph_boxes = lambda img: boxes[:6] + boxes[7:]
        dropped_middle = mini_perceive.read_hand(image, templates)['my_hand']
        check('alignment fixes missing middle glyph', dropped_middle[:6] == baseline[:6] and dropped_middle[7:] == baseline[7:] and dropped_middle[6] == 'UNKNOWN', str(dropped_middle[5:9]))
    finally:
        mini_perceive.glyph_boxes = original


def played_cases():
    cases = [
        ('outputs/run_final2.png', {'left': ['9'], 'right': []}),
        ('outputs/mini_bot_check2.png', {'left': [], 'right': ['2', '2']}),
        ('outputs/mini_live_now9.png', {'left': [], 'right': ['8', '8', '8', '10']}),
        ('outputs/review_now.png', {'left': [], 'right': ['2']}),
        ('outputs/mini_live_now12.png', {'left': [], 'right': []}),
        ('outputs/stuck_state.png', {'left': ['J', '10', '9', '8', '7', '6', '5'], 'right': []}),
        ('outputs/stuck_state2.png', {'left': ['8'], 'right': ['4']}),
        ('outputs/manual_select_test2.png', {'left': ['7', '7'], 'right': []}),
        ('outputs/follow_debug.png', {'left': ['K', 'K'], 'right': []}),
        ('outputs/pre_run2.png', {'left': [], 'right': []}),
    ]
    templates = mini_perceive.load_templates()
    for path, expected in cases:
        image = mini_perceive.load_image(fixture_path(path))
        played = mini_perceive.detect_played(image, templates)
        actual = {seat: played[seat]['ranks'] for seat in ('left', 'right')}
        check(f'played {path}', actual == expected, str(actual))
        check(f'played count {path}', all(info['reliable'] and info['count_est'] == len(info['ranks']) for info in played.values()),
              str({seat: info['count_est'] for seat, info in played.items()}))


def tuoguan_cases():
    bot = mini_bot.Bot()
    for path in ['outputs/mini_live_now8.png', 'outputs/mini_bot_check3.png', 'outputs/mini_live_now12.png', 'outputs/mini2_bid_after.png', 'outputs/run_final2.png']:
        rgb = load_rgb(path)
        sx = rgb.shape[1] / mini_bot.BASE_WIDTH
        sy = rgb.shape[0] / mini_bot.BASE_HEIGHT
        score = bot.tuoguan_score(rgb, sx, sy)
        check(f'no cancel button {path}', score < 0.5, str(round(score, 3)))
    rgb = load_rgb('outputs/run_now9.png')
    sx = rgb.shape[1] / mini_bot.BASE_WIDTH
    sy = rgb.shape[0] / mini_bot.BASE_HEIGHT
    check('cancel button detected', bot.tuoguan_score(rgb, sx, sy) > 0.7, str(round(bot.tuoguan_score(rgb, sx, sy), 3)))
    padded = cv2.copyMakeBorder(rgb, 0, 40, 0, 0, cv2.BORDER_CONSTANT)
    padded_score = bot.tuoguan_score(padded, 1.0, 620 / 580)
    check('cancel button is found relative to the table with extra window height',
          padded_score > 0.9, str(round(padded_score, 3)))


def strength_cases():
    check('strong hand calls', mini_bot.Bot.hand_strength(None, ['BJ', 'LJ', '2', '2', 'A', 'A', 'A', 'A', '10', '10', '9', '7', '5', '5', '4', '4', '3']) >= 6, '')
    check('weak hand passes', mini_bot.Bot.hand_strength(None, ['3', '4', '5', '6', '7', '9', '10', 'J', 'Q', 'K', '4', '6', '8', '9', '10', 'J', 'Q']) < 6, '')


def lead_execution_event_cases():
    import tempfile
    from streamer_events import StreamerEventStream, follow_events

    lead_rgb = load_rgb('outputs/mini_live_now12.png')
    result_rgb = load_rgb('outputs/mini_live_now.png')
    observation = mini_perceive.read_hand(cv2.cvtColor(lead_rgb, cv2.COLOR_RGB2BGR))
    index = next(i for i, rank in enumerate(observation['my_hand']) if rank != 'UNKNOWN')
    rank = observation['my_hand'][index]
    scale_x = lead_rgb.shape[1] / mini_bot.BASE_WIDTH
    scale_y = lead_rgb.shape[0] / mini_bot.BASE_HEIGHT
    _, _, ui = mini_bot.classify(lead_rgb, scale_x, scale_y)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        bot = mini_bot.Bot()
        bot.events = StreamerEventStream(root / 'events.jsonl', root / 'latest.json', session_id='selftest')
        bot.role = 'farmer'
        bot.lead_cache_key = (tuple(observation['my_hand']), bot.role)
        bot.lead_cache = {
            'chosen': [index],
            'candidate': {'id': 'cached', 'type': 'single', 'ranks': [rank], 'indices': [index]},
            'decision': {'reason': '缓存决策', 'utterance': f'出一张{rank}。'},
        }
        frames = iter([(lead_rgb, None), (lead_rgb, None), (result_rgb, None)])
        bot.capture = lambda hwnd: next(frames)
        bot.check_tuoguan = lambda *args: False
        bot.set_selection = lambda *args, **kwargs: True
        bot.confirm_selection = lambda hwnd, bbox, rgb, chosen, sx, sy, **kwargs: (True, rgb)
        events_at_click = []

        def click(*args, **kwargs):
            events_at_click.extend(follow_events(root / 'events.jsonl', from_start=True, speech_only=False, once=True))
            return True

        bot.click = click
        bot.bot_decision = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('cache missed'))
        bot.narrate_final = lambda *args, **kwargs: {'reason': '不应重新解说', 'utterance': '不应重新解说'}
        bot.last_play = {'seat': 'right', 'ranks': ['K'], 'time': 0.0}
        bot.handle_lead(None, None, scale_x, scale_y, ui)
        events = list(follow_events(root / 'events.jsonl', from_start=True, speech_only=False, once=True))
        check('lead decision is published before click', len(events_at_click) == 1
              and events_at_click[0]['execution'] == 'decided' and events_at_click[0]['source'] == 'cached_decision', str(events_at_click))
        check('cached lead emits decision and confirmed result without repeated speech', len(events) == 2
              and events[0]['speak'] is False and events[1]['event_kind'] == 'execution_result'
              and events[1]['execution'] == 'confirmed' and events[1]['result_for'] == events[0]['event_id']
              and events[1]['speak'] is False and bot.last_play is None, str(events))


def streamer_event_cases():
    import tempfile
    from streamer_events import StreamerEventStream, follow_events

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        stream = StreamerEventStream(root / 'events.jsonl', root / 'latest.json', session_id='selftest')
        decided = stream.publish({
            'phase': 'lead', 'action': 'play', 'chosen_ranks': ['Q'],
            'utterance': '单张Q。', 'execution': 'decided',
        })
        rejected = stream.publish({
            'phase': 'follow', 'action': 'play', 'chosen_ranks': ['J'],
            'utterance': '单张J。', 'execution': 'rejected',
        })
        latest = json.loads((root / 'latest.json').read_text(encoding='utf-8'))
        spoken = list(follow_events(root / 'events.jsonl', from_start=True, once=True))
        all_events = list(follow_events(root / 'events.jsonl', from_start=True, speech_only=False, once=True))
        check('streamer event identity', decided['event_id'] != rejected['event_id'] and decided['session_id'] == 'selftest', '')
        check('streamer decided event speaks immediately', decided['speak'] is True and decided['schema_version'] == 2, '')
        check('streamer rejected event muted', rejected['speak'] is False, '')
        check('streamer latest keeps latest speakable event', latest['event_id'] == decided['event_id'], '')
        check('streamer consumer filters rejected speech', [event['event_id'] for event in spoken] == [decided['event_id']], str(spoken))
        check('streamer diagnostic replay', [event['sequence'] for event in all_events] == [1, 2], str(all_events))


def correction_event_cases():
    import tempfile
    from streamer_events import StreamerEventStream, follow_events

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        bot = mini_bot.Bot()
        bot.events = StreamerEventStream(root / 'events.jsonl', root / 'latest.json', session_id='selftest')
        decision = bot.write_decision_event({
            'phase': 'follow', 'action': 'play', 'chosen_ranks': ['A', 'A'],
            'utterance': '对A，压住。', 'execution': 'decided',
        })
        failure_text = '跟牌未确认生效，可能因目标识别不完整，改用游戏提示。'
        correction = bot.write_execution_result(decision, 'rejected', failure_text)
        next_decision = bot.write_decision_event({
            'phase': 'lead', 'action': 'play', 'chosen_ranks': ['5'],
            'utterance': '先出一张5。', 'execution': 'decided',
        })
        success = bot.write_execution_result(next_decision, 'confirmed', '成功时不应朗读')
        spoken = list(follow_events(root / 'events.jsonl', from_start=True, once=True))
        check('failed action adds speakable correction', correction['speak'] is True
              and correction['event_kind'] == 'execution_result' and correction['execution'] == 'rejected'
              and correction['reason'] == failure_text, str(correction))
        check('correction references original decision', correction['action_id'] == decision['action_id']
              and correction['result_for'] == decision['event_id'], '')
        check('successful action result is silent', success['speak'] is False and success['utterance'] == '', '')
        check('live consumer receives decisions and correction only', [event['event_id'] for event in spoken]
              == [decision['event_id'], correction['event_id'], next_decision['event_id']], str(spoken))


def scene_plan_cases():
    kicked = mini_bot.dialog_scene_plan('哎呀！太久没操作，地主要请你离开房间了！确定')
    free_beans = mini_bot.dialog_scene_plan('欢乐豆不足，可以免费领取救济金')
    paid_beans = mini_bot.dialog_scene_plan('欢乐豆不足，是否前往购买并充值6元')
    bean_notice = mini_bot.dialog_scene_plan('欢乐豆不足，确定')
    lobby = mini_bot.idle_scene_plan('斗地主合集 百变八雀牌')
    loading = mini_bot.idle_scene_plan('斗地主合集 正在加载98%')
    unknown = mini_bot.idle_scene_plan('一个从未见过的页面')
    video = mini_bot.dialog_scene_plan('观看视频，输豆返还暴击得欢乐豆！欢乐豆×323 暴击领豆')
    friend_auth = mini_bot.dialog_scene_plan('申请获取你的好友信息，用于展示好友列表 取消 确定')
    friend_room = mini_bot.idle_scene_plan('好友房仅供娱乐，勿做他用 加入房间 开设房间 人数选择 玩法选择')
    friend_auth_idle = mini_bot.idle_scene_plan('申请获取你的好友信息，用于展示好友列表 取消 确定')
    check('kicked room dialog plan', kicked['action'] == 'acknowledge_kick' and kicked['point'] == mini_bot.STATIC_POINTS['confirm'], str(kicked))
    check('free beans dialog plan', free_beans['action'] == 'claim_free_beans' and not free_beans.get('blocked'), str(free_beans))
    check('paid beans dialog blocks automation', paid_beans['action'] == 'purchase_required' and paid_beans['blocked'], str(paid_beans))
    check('bean notice can be dismissed safely', bean_notice['action'] == 'dismiss_bean_notice' and not bean_notice.get('blocked'), str(bean_notice))
    check('main lobby opens collection', lobby['action'] == 'open_collection' and lobby['point'] == mini_bot.STATIC_POINTS['lobby_collection'], str(lobby))
    check('main lobby tolerates OCR 王/主 typo', mini_bot.idle_scene_plan('斗地王合集 百变八雀牌 斗仙牌')['action'] == 'open_collection', '')
    check('loading scene waits instead of clicking', loading['action'] == 'wait_loading' and loading['wait'], str(loading))
    check('unknown scene does not blind click', unknown['action'] == 'unknown_scene' and unknown['wait'], str(unknown))
    check('unknown scene asks the teacher instead of guessing', '找老师' in unknown['utterance'], str(unknown))
    check('free video bean reward is never clicked',
          video['action'] == 'skip_video_beans' and video.get('blocked')
          and 'point' not in video, str(video))
    check('free video bean variants are also left alone',
          all(mini_bot.dialog_scene_plan(text)['action'] == 'skip_video_beans'
              for text in ('免费视频返豆', '看视频领欢乐豆', '观看视频返还欢乐豆 暴击')),
          '')
    check('result candidate requires scoreboard text', mini_bot.confirm_result_scene('昵称 底分 倍数 欢乐豆'), '')
    check('result confirmation tolerates missing nickname', mini_bot.confirm_result_scene('底分80 倍数120 继续游戏'), '')
    check('result confirmation tolerates OCR missing table headers',
          mini_bot.confirm_result_scene('昵称玩家甲 底分151515 360180 欢乐10032006 继续游戏'), '')
    check('continue text alone is not confirmed result', not mini_bot.confirm_result_scene('新手场 继续游戏'), '')
    check('video reward is not confirmed result', not mini_bot.confirm_result_scene('观看视频 欢乐豆 暴击领豆'), '')
    check('friend authorization dialog is cancelled, not accepted',
          friend_auth['action'] == 'deny_friend_authorization'
          and friend_auth['point'] == mini_bot.STATIC_POINTS['friend_cancel'], str(friend_auth))
    check('friend room returns to the lobby',
          friend_room['action'] == 'leave_friend_room' and friend_room['point'] == mini_bot.STATIC_POINTS['back'], str(friend_room))
    check('friend authorization is cancelled from the idle path too',
          friend_auth_idle['action'] == 'deny_friend_authorization'
          and friend_auth_idle['point'] == mini_bot.STATIC_POINTS['friend_cancel'], str(friend_auth_idle))


def purchase_scene_cases():
    import tempfile
    from streamer_events import StreamerEventStream, follow_events

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        bot = mini_bot.Bot()
        bot.events = StreamerEventStream(root / 'events.jsonl', root / 'latest.json', session_id='selftest')
        bot.read_scene_ocr = lambda rgb: {'text': '欢乐豆不足，前往购买并充值6元'}
        bot.click = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('must not click purchase'))
        bot.handle_dialog_scene(None, None, None, 1, 1)
        bot.handle_dialog_scene(None, None, None, 1, 1)
        events = list(follow_events(root / 'events.jsonl', from_start=True, speech_only=False, once=True))
        check('purchase scene emits one blocked speech event', len(events) == 1 and events[0]['action'] == 'purchase_required'
              and events[0]['execution'] == 'blocked' and events[0]['speak'] is True, str(events))
        check('purchase scene keeps hand id empty', events[0]['hand_id'] is None, str(events[0]))


def post_trial_guard_cases():
    import tempfile
    from streamer_events import StreamerEventStream, follow_events

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        bot = mini_bot.Bot()
        bot.events = StreamerEventStream(root / 'scene_events.jsonl', root / 'scene_latest.json', session_id='selftest')
        bot.ensure_foreground = lambda hwnd: True
        bot.click = lambda *args, **kwargs: True
        plan = {
            'phase': 'lobby', 'action': 'quick_start', 'utterance': '快速开始。',
            'reason': '测试冷却', 'point': mini_bot.STATIC_POINTS['quick_start'],
        }
        bot.perform_scene_action(None, None, 1, 1, plan, '快速开始')
        bot.perform_scene_action(None, None, 1, 1, plan, '快速开始')
        scene_events = list(follow_events(root / 'scene_events.jsonl', from_start=True, speech_only=False, once=True))
        check('scene action cooldown prevents duplicate click and speech', len(scene_events) == 2, str(scene_events))
        bot.bid_complete = True
        bot.capture = lambda hwnd: (_ for _ in ()).throw(AssertionError('completed bid must not recapture'))
        bot.handle_bid(None, None, 1, 1, {'orange': []})
        check('completed bid stage does not repeat', True, '')

        lead_rgb = load_rgb('outputs/mini_live_now12.png')
        bot.last_scene_ocr = 0.0
        bot.read_scene_ocr = lambda rgb: (_ for _ in ()).throw(AssertionError('table transition must not OCR'))
        bot.handle_idle_scene(None, None, lead_rgb, 1, 1)
        check('table transition with hand is not unknown lobby', True, '')
        bot.hand_active = True
        bot.handle_idle_scene(None, None, None, 1, 1)
        check('active hand never enters lobby OCR', True, '')

        observation = mini_perceive.read_hand(cv2.cvtColor(lead_rgb, cv2.COLOR_RGB2BGR))
        scale_x = lead_rgb.shape[1] / mini_bot.BASE_WIDTH
        scale_y = lead_rgb.shape[0] / mini_bot.BASE_HEIGHT
        _, _, ui = mini_bot.classify(lead_rgb, scale_x, scale_y)
        retry_bot = mini_bot.Bot()
        retry_bot.events = StreamerEventStream(root / 'retry_events.jsonl', root / 'retry_latest.json', session_id='selftest')
        retry_bot.role = 'farmer'
        key = tuple(observation['my_hand'])
        chosen = [17, 16, 14, 13, 11, 10, 9]
        ranks = ['3', '4', '5', '6', '7', '8', '9']
        retry_bot.lead_cache_key = (key, retry_bot.role)
        retry_bot.lead_cache = {
            'chosen': chosen,
            'candidate': {'id': 'straight', 'type': 'straight', 'ranks': ranks, 'indices': chosen},
            'decision': {'reason': '固定顺子', 'utterance': '出3到9的顺子。'},
        }
        frames = iter([(lead_rgb, None)] * 6)
        retry_bot.capture = lambda hwnd: next(frames)
        retry_bot.check_tuoguan = lambda *args: False
        retry_bot.save_action_frame = lambda *args: None
        selected = []
        retry_bot.set_selection = lambda hwnd, bbox, obs, indices, sx, sy, limit=None: selected.extend(indices) or True
        retry_bot.confirm_selection = lambda hwnd, bbox, rgb, chosen, sx, sy, **kwargs: (True, rgb)
        retry_bot.click = lambda *args, **kwargs: True
        retry_bot.bot_decision = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('retry must not ask engine'))
        retry_bot.handle_lead(None, None, scale_x, scale_y, ui)
        retry_bot.handle_lead(None, None, scale_x, scale_y, ui)
        retry_events = list(follow_events(root / 'retry_events.jsonl', from_start=True, speech_only=False, once=True))
        check('failed lead retries the identical move without strategy changes', selected == chosen + chosen
              and all(event['chosen_ranks'] == ranks for event in retry_events), str(retry_events))
        check('second identical failure stops further attempts', len(retry_events) == 4
              and retry_bot.lead_blocked_key == key
              and retry_events[1]['speak'] is True and retry_events[3]['speak'] is True, str(retry_events))


def result_outcome_cases():
    import tempfile
    from streamer_events import StreamerEventStream

    cases = [
        ('outputs/live_trial_result.png', 'win'),
        ('outputs/hand_result.png', 'win'),
        ('outputs/mini_live_now.png', 'win'),
        ('outputs/live_trial4_current.png', 'win'),
        ('outputs/hand10_result.png', 'loss'),
        ('outputs/hand9_result.png', 'loss'),
        ('outputs/hand4_result.png', 'loss'),
        ('outputs/result_audit.png', 'loss'),
    ]
    for path, expected in cases:
        rgb = load_rgb(path)
        sx = rgb.shape[1] / mini_bot.BASE_WIDTH
        sy = rgb.shape[0] / mini_bot.BASE_HEIGHT
        actual = mini_bot.detect_result_outcome(rgb, sx, sy)
        check(f'result outcome {path}', actual == expected, f'{actual} != {expected}')

    rgb = load_rgb('outputs/result_audit.png')
    sx = rgb.shape[1] / mini_bot.BASE_WIDTH
    sy = rgb.shape[0] / mini_bot.BASE_HEIGHT
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        bot = mini_bot.Bot(hand_limit=1)
        bot.events = StreamerEventStream(root / 'events.jsonl', root / 'latest.json', session_id='selftest')
        bot.hand_active = True
        finished = bot.handle_result(None, None, rgb, sx, sy)
        check('one hand limit exits on a partially recognized OCR result page',
              finished and bot.hands_done == 1 and not bot.hand_active, str(bot.hands_done))


def continue_point_cases():
    cases = [
        ('outputs/live_trial6_result.png', 579, 521),
        ('outputs/live_trial5_result.png', 579, 521),
        ('outputs/mini_live_now.png', 579, 472),
    ]
    for path, x, y in cases:
        rgb = load_rgb(path)
        sx = rgb.shape[1] / mini_bot.BASE_WIDTH
        sy = rgb.shape[0] / mini_bot.BASE_HEIGHT
        point = mini_bot.find_continue_point(rgb, sx, sy)
        check(f'continue point {path}', point is not None
              and abs(point[0] - x) <= 6 and abs(point[1] - y) <= 6, str(point))


def scaled_selection_cases():
    image = mini_perceive.load_image(fixture_path('outputs/mini_live_now12.png'))
    base = mini_perceive.detect_hand(image)
    padded = cv2.copyMakeBorder(image, 0, 40, 0, 0, cv2.BORDER_CONSTANT)
    padded_hand = mini_perceive.detect_hand(padded)
    check('extra window height does not mark every card as selected',
          padded_hand['count'] == base['count']
          and padded_hand['selected_indices'] == base['selected_indices'], str(padded_hand['selected_indices']))
    scale = 0.8
    scaled_image = cv2.resize(image, (746, 464), interpolation=cv2.INTER_AREA)
    scaled = mini_perceive.detect_hand(scaled_image)
    bot = mini_bot.Bot()
    indices = [1, 2, 3, 4, 5]
    base_points = [bot.card_point(base, index) for index in indices]
    scaled_points = [bot.card_point(scaled, index, scale, scale) for index in indices]
    difference = max(abs(left - right) for left_point, right_point in zip(base_points, scaled_points)
                     for left, right in zip(left_point, right_point))
    check('scaled card points stay in base coordinates', difference <= 3.0, str(difference))
    clicked = []
    logs = []
    bot.click = lambda *args, **kwargs: clicked.append(args[2]) or True
    bot.log = logs.append
    bbox = (100, 50, 846, 514)
    bot.set_selection(None, bbox, scaled, [3, 4, 5, 1, 2], scale, scale)
    expected_screen = [[100 + round(point[0] * scale), 50 + round(point[1] * scale)] for point in base_points]
    actual_screen = [[100 + round(point[0] * scale), 50 + round(point[1] * scale)] for point in clicked]
    check('scaled selection emits correct screen coordinates', len(clicked) == 5
          and all(abs(actual - expected) <= 2 for actual_point, expected_point in zip(actual_screen, expected_screen)
                  for actual, expected in zip(actual_point, expected_point))
          and len(logs) == 5 and all(entry['scale'] == [0.8, 0.8] for entry in logs), str(actual_screen))
    cluttered = mini_perceive.detect_hand(
        mini_perceive.load_image(fixture_path('outputs/selection_unstable_113049_426018.png')))
    check('table decoration left of the hand is not counted as a card',
          cluttered['count'] == 6 and cluttered['selected_indices'] == [0, 1, 2], str(cluttered['count']))


def role_lock_cases():
    rgb = load_rgb('outputs/run_final2.png')
    original = mini_perceive.detect_landlord
    try:
        seats = iter([None, 'right'])
        mini_perceive.detect_landlord = lambda image: next(seats)
        bot = mini_bot.Bot()
        bot.update_role_from_table(rgb)
        bot.update_role_from_table(rgb)
        check('self landlord role stays locked', bot.role == 'landlord' and bot.landlord_seat is None
              and bot.counts == {'left': 17, 'right': 17}, str(bot.counts))

        seats = iter(['right', None])
        mini_perceive.detect_landlord = lambda image: next(seats)
        bot = mini_bot.Bot()
        bot.update_role_from_table(rgb)
        bot.update_role_from_table(rgb)
        check('farmer role and landlord count stay locked', bot.role == 'farmer' and bot.landlord_seat == 'right'
              and bot.counts == {'left': 17, 'right': 20}, str(bot.counts))
    finally:
        mini_perceive.detect_landlord = original


def bad_play_regression_cases():
    bot = mini_bot.Bot()
    known = [(0, 'BJ'), (1, 'Q'), (2, '10'), (3, '10'), (4, '8')]
    picked = bot.pick_safe_single(known)
    check('fallback single does not split known pair', picked == (4, '8'), str(picked))
    result = bot_engine('playing', ['Q', '10', '8'], role='landlord', can_pass=True, target=['8'], target_seat='left')
    check('landlord can beat single 8', result['action']['type'] == 'play'
          and len(result['action'].get('ranks', [])) == 1
          and mini_bot.RANK_VALUE[result['action']['ranks'][0]] > 8, str(result['action']))


def role_prompt_cases():
    import model_decision

    states = []
    runtime_root = ROOT / 'outputs' / 'selftest_runtime'
    runtime_root.mkdir(parents=True, exist_ok=True)
    (runtime_root / 'api_config.json').write_text(json.dumps({
        'base_url': 'https://example.invalid/v1',
        'model': 'offline-test',
        'api_key_env': 'DOUDIZHU_API_KEY',
        'timeout_seconds': 1,
    }), encoding='utf-8')
    original_root = mini_bot.ROOT
    original_key = model_decision.load_api_key
    original_post = model_decision.post_chat
    mini_bot.ROOT = runtime_root
    try:
        model_decision.load_api_key = lambda config: 'offline-test'
        def post_chat(config, api_key, messages):
            states.append(json.loads(messages[1]['content']))
            content = json.dumps({'candidate_id': 'c000', 'reason': '测试理由', 'utterance': '测试素材'})
            return {'choices': [{'message': {'content': content}}]}, 200, 1
        model_decision.post_chat = post_chat
        bot = mini_bot.Bot()
        candidate = {'id': 'c000', 'type': 'single', 'ranks': ['Q'], 'indices': [0], 'reasons': ['测试']}
        bot.deepseek_pick(['Q', '10', '8'], 'landlord', [candidate], recommended='c000')
        bot.narrate_final(['Q', '10', '8'], 'landlord', None, candidate)
        check('decision prompt contains locked landlord identity', states[0]['role'] == 'landlord'
              and states[0]['landlord_seat'] == 'self' and states[0]['seat_roles']['self'] == 'landlord'
              and states[0]['seat_roles']['left'] == 'farmer' and states[0]['role_locked'] is True
              and states[0]['card_names'] == {'LJ': '小王', 'BJ': '大王'}, str(states[0]))
        check('narration prompt reuses same locked identity', states[1]['role'] == states[0]['role']
              and states[1]['landlord_seat'] == states[0]['landlord_seat']
              and states[1]['seat_roles'] == states[0]['seat_roles']
              and states[1]['identity_rule'] == states[0]['identity_rule']
              and states[1]['my_hand'] == ['Q', '10', '8']
              and states[1]['card_names'] == states[0]['card_names'], str(states[1]))
    finally:
        mini_bot.ROOT = original_root
        model_decision.load_api_key = original_key
        model_decision.post_chat = original_post


def execution_evidence_cases():
    check('four Q with idle and unchanged hand is not success',
          mini_bot.evaluate_play_execution('idle', 'lead', 4, 4, 0) == 'uncertain', '')
    check('unchanged lead is rejected',
          mini_bot.evaluate_play_execution('lead', 'lead', 4, 4, 0) == 'rejected', '')
    check('exact hand decrease confirms play',
          mini_bot.evaluate_play_execution('idle', 'lead', 5, 1, 1) == 'confirmed', '')
    check('result screen confirms final play without glyphs',
          mini_bot.evaluate_play_execution('result', 'lead', 4, None, 0) == 'confirmed', '')


def teammate_fallback_cases():
    import tempfile
    import numpy as np
    from streamer_events import StreamerEventStream, follow_events

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        bot = mini_bot.Bot()
        bot.events = StreamerEventStream(root / 'events.jsonl', root / 'latest.json', session_id='selftest')
        bot.role = 'farmer'
        bot.landlord_seat = 'right'
        bot.capture = lambda hwnd: (np.zeros((580, 933, 3), dtype=np.uint8), (0, 0, 933, 580))
        bot.find_hint_button = lambda *args: None
        clicked = []
        bot.click = lambda hwnd, bbox, point, *args, **kwargs: clicked.append(point) or True
        ocr_bot = mini_bot.Bot()
        ocr_bot.events = StreamerEventStream(root / 'ocr_events.jsonl', root / 'ocr_latest.json', session_id='selftest')
        ocr_bot.log = lambda record: None
        ocr_bot.notify = lambda *args, **kwargs: None
        ocr_bot.capture = bot.capture
        ocr_bot.find_hint_button = lambda *args: {'text': '提示', 'center': [526, 330], 'source': 'rapidocr'}
        ocr_clicked = []
        ocr_bot.click = lambda hwnd, bbox, point, *args, **kwargs: ocr_clicked.append(point) or True
        original_classify = mini_bot.classify
        mini_bot.classify = lambda rgb, sx, sy: ('idle', {}, None)
        try:
            bot.fallback_hint(None, None, 1, 1, {'orange': [{'center': [644, 331]}]}, target_seat='left')
            ocr_bot.fallback_hint(None, None, 1, 1, {'orange': [{'center': [644, 331]}]}, target_seat='left')
        finally:
            mini_bot.classify = original_classify
        events = list(follow_events(root / 'events.jsonl', from_start=True, speech_only=False, once=True))
        check('unreadable follow uses the game hint even when the seat is the teammate',
              clicked[:2] == [[533, 331], [644, 331]] and len(events) == 2
              and events[0]['action'] == 'play' and events[0]['chosen'] == 'game_hint', str((clicked, events)))
        check('hint OCR point is used before the legacy offset',
              ocr_clicked[:2] == [[526, 330], [644, 331]], str(ocr_clicked))


def hint_button_cases():
    bot = mini_bot.Bot()
    rgb = load_rgb('outputs/mini2_bid_after.png')
    sx = rgb.shape[1] / mini_bot.BASE_WIDTH
    sy = rgb.shape[0] / mini_bot.BASE_HEIGHT
    button = bot.find_hint_button(rgb, sx, sy)
    check('hint button is located by OCR in a historical frame',
          button is not None and button['text'] == '提示' and button['source'] == 'rapidocr'
          and abs(button['center'][0] - 526) <= 3 and abs(button['center'][1] - 330) <= 3, str(button))


def pass_only_retry_cases():
    import numpy as np

    bot = mini_bot.Bot()
    bot.log = lambda record: None
    bot.capture = lambda hwnd: (np.zeros((580, 933, 3), dtype=np.uint8), (0, 0, 933, 580))
    original_classify = mini_bot.classify
    try:
        states = iter([('turn', {}, {'orange': [{'center': [644, 331]}]})])
        mini_bot.classify = lambda rgb, sx, sy: next(states)
        check('pass-only recheck aborts when a normal turn reappears',
              bot.recheck_pass_only(None, 1, 1) is None, '')

        ui = {'clock': [{'center': [474, 328]}]}
        states = iter([('pass_only', {}, ui), ('pass_only', {}, ui)])
        mini_bot.classify = lambda rgb, sx, sy: next(states)
        check('pass-only continues only after two matching rechecks',
              bot.recheck_pass_only(None, 1, 1) is ui, '')
    finally:
        mini_bot.classify = original_classify


def bid_button_cases():
    bot = mini_bot.Bot()
    for path in ['outputs/run_now7.png', 'outputs/mini2_deal1.png']:
        image = mini_perceive.load_image(fixture_path(path))
        rgb = image[:, :, ::-1]
        sx, sy = mini_perceive.scale_for(image)
        buttons = bot.read_bid_buttons(rgb, sx, sy)
        texts = {item['text'] for item in buttons}
        check(f'bid button templates {path}', {'叫地主', '不叫'} <= texts
              and all(item['source'] == 'template' for item in buttons)
              and all(280 <= item['center'][1] <= 370 for item in buttons), str(buttons))


def bid_recognition_cases():
    from unittest.mock import patch
    import numpy as np

    bot = mini_bot.Bot()
    bot.read_hand_with_retry = lambda hwnd: (np.zeros((580, 933, 3), dtype=np.uint8), {
        'my_hand': ['A', 'A'], 'unknown_indices': [],
    })
    bot.bot_decision = lambda *args, **kwargs: {'power': 1, 'action': {'score': 1}}
    bot.read_bid_buttons = lambda *args: [{'text': '不叫', 'source': 'template', 'center': [580, 333]}]
    logs = []
    bot.log = logs.append
    bot.click = lambda *args, **kwargs: True
    bot.handle_bid(None, None, 1, 1, {'orange': [{'center': [374, 333]}], 'blue': []})
    check('bid log reports the actual template source',
          any(item.get('state') == 'bid_decision' and item.get('point_source') == 'template' for item in logs), str(logs))

    cached_bot = mini_bot.Bot()
    calls = []
    cached_bot.read_ui_buttons = lambda *args: calls.append(1) or [
        {'text': '不叫', 'source': 'rapidocr', 'center': [580, 333]},
    ]
    rgb = np.zeros((580, 933, 3), dtype=np.uint8)
    original_dir = mini_bot.BID_TEMPLATE_DIR
    mini_bot.BID_TEMPLATE_DIR = mini_bot.ROOT / 'outputs' / '__missing_bid_templates__'
    try:
        cached_bot.read_bid_buttons(rgb, 1, 1)
        cached_bot.read_bid_buttons(rgb, 1, 1)
        rgb[300, 300, 0] = 1
        cached_bot.read_bid_buttons(rgb, 1, 1)
    finally:
        mini_bot.BID_TEMPLATE_DIR = original_dir
    check('identical bid frame reuses OCR result but changed frame is reread',
          len(calls) == 2, str(len(calls)))

    deduped_bot = mini_bot.Bot()
    key = (True, 'pass_bid', ('A', 'A'))
    deduped_bot.bid_last = (key, mini_bot.time.perf_counter())
    deduped_bot.bid_click_keys = {key: ('bid', '不叫')}
    deduped_bot.bid_clicks = {('bid', '不叫'): 2}
    deduped_bot.read_hand_with_retry = lambda hwnd: (_ for _ in ()).throw(
        AssertionError('deduped bid must not recapture or run OCR'))
    deduped_bot.log = logs.append
    deduped_bot.handle_bid(None, None, 1, 1, {'orange': [], 'blue': []})
    check('bid click limit is enforced before another OCR pass',
          logs[-1]['state'] == 'bid_click_deduped', str(logs[-1]))


def click_guard_cases():
    import ctypes
    import pydirectinput
    from capture_window import RECT

    class FakeUser32:
        def GetForegroundWindow(self):
            return 1
        def GetClientRect(self, hwnd, pointer):
            rect = ctypes.cast(pointer, ctypes.POINTER(RECT)).contents
            rect.left = 0
            rect.top = 0
            rect.right = 933
            rect.bottom = 580
            return True
        def ClientToScreen(self, hwnd, pointer):
            return True
        def SetCursorPos(self, x, y):
            return True
        def GetCursorPos(self, pointer):
            actual = ctypes.cast(pointer, ctypes.POINTER(mini_bot.POINT)).contents
            actual.x = 999
            actual.y = 999
            return True

    class ResizedUser32(FakeUser32):
        def GetClientRect(self, hwnd, pointer):
            rect = ctypes.cast(pointer, ctypes.POINTER(RECT)).contents
            rect.left = 0
            rect.top = 0
            rect.right = 800
            rect.bottom = 496
            return True

    class NearUser32(FakeUser32):
        def __init__(self):
            self.position = (0, 0)
        def SetCursorPos(self, x, y):
            self.position = (x, y)
            return True
        def GetCursorPos(self, pointer):
            actual = ctypes.cast(pointer, ctypes.POINTER(mini_bot.POINT)).contents
            actual.x = self.position[0] + 4
            actual.y = self.position[1]
            return True

    bot = mini_bot.Bot()
    bot.user32 = FakeUser32()
    original = pydirectinput.mouseDown
    pydirectinput.mouseDown = lambda: (_ for _ in ()).throw(AssertionError('must not press at wrong cursor'))
    try:
        result = bot.click(1, (0, 0, 933, 580), [100, 100], 1, 1, physical=True)
    finally:
        pydirectinput.mouseDown = original
    check('click is blocked when cursor cannot reach target', result is False, '')

    resized = mini_bot.Bot()
    resized.user32 = ResizedUser32()
    check('click is blocked when the live window size differs from the capture',
          resized.click(1, (0, 0, 933, 580), [100, 100], 1, 1, physical=True) is False, '')

    near = mini_bot.Bot()
    near.user32 = NearUser32()
    pressed = []
    original_down = pydirectinput.mouseDown
    original_up = pydirectinput.mouseUp
    pydirectinput.mouseDown = lambda: pressed.append('down')
    pydirectinput.mouseUp = lambda: pressed.append('up')
    try:
        result = near.click(1, (0, 0, 933, 580), [100, 100], 1, 1, physical=True)
    finally:
        pydirectinput.mouseDown = original_down
        pydirectinput.mouseUp = original_up
    check('small cursor rounding offset within six pixels is accepted',
          result is True and pressed == ['down', 'up'], str((result, pressed)))


def takeover_cases():
    import tempfile
    from streamer_events import StreamerEventStream, follow_events

    class OverlayStub:
        def __init__(self):
            self.on_takeover = None
            self.paused = None
        def set_paused(self, value):
            self.paused = value
        def show_speech(self, *args):
            return None

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        overlay = OverlayStub()
        bot = mini_bot.Bot(overlay=overlay)
        bot.events = StreamerEventStream(root / 'events.jsonl', root / 'latest.json', session_id='selftest')
        bot.toggle_pause()
        click_result = bot.click(None, None, [1, 2], 1, 1, physical=False)
        foreground_result = bot.ensure_foreground(1)
        check('manual takeover blocks clicks and foreground changes', bot.paused is True
              and overlay.paused is True and click_result is False and foreground_result is False, '')
        bot.toggle_pause()
        events = list(follow_events(root / 'events.jsonl', from_start=True, speech_only=False, once=True))
        check('takeover and resume require separate user actions', bot.paused is False
              and [event['action'] for event in events] == ['pause', 'resume'], str(events))


def tuoguan_template_failclosed_cases():
    logs = []
    rgb = load_rgb('outputs/run_final2.png')
    original = mini_bot.TUOGUAN_TEMPLATE
    mini_bot.TUOGUAN_TEMPLATE = ROOT / 'outputs' / '__missing_tuoguan__.png'
    try:
        bot = mini_bot.Bot()
        bot.log = logs.append
        bot.notify = lambda *args, **kwargs: None
        score = bot.tuoguan_score(rgb, rgb.shape[1] / mini_bot.BASE_WIDTH, rgb.shape[0] / mini_bot.BASE_HEIGHT)
    finally:
        mini_bot.TUOGUAN_TEMPLATE = original
    check('missing tuoguan template cannot report auto-play active', score == 0.0, str(score))
    check('missing tuoguan template is logged once', [entry['state'] for entry in logs] == ['tuoguan_template_missing'], str(logs))


def bid_stage_cases():
    import tempfile
    import numpy as np
    from streamer_events import StreamerEventStream, follow_events

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        bot = mini_bot.Bot()
        bot.events = StreamerEventStream(root / 'events.jsonl', root / 'latest.json', session_id='selftest')
        bot.log = lambda record: None
        bot.notify = lambda *args, **kwargs: None
        bot.capture = lambda hwnd: (np.zeros((580, 933, 3), dtype=np.uint8), (0, 0, 933, 580))
        bot.bot_decision = lambda *args, **kwargs: {'power': 12, 'action': {'score': 2}}
        bot.bid_done = True
        bot.read_bid_buttons = lambda rgb, sx, sy: [
            {'text': '叫地主', 'center': [374, 333], 'confidence': 0.99, 'source': 'template'},
            {'text': '不叫', 'center': [604, 335], 'confidence': 0.99, 'source': 'template'},
        ]
        clicked = []
        bot.click = lambda *args, **kwargs: clicked.append(args[2]) or True
        bot.handle_bid(None, None, 1, 1, {'orange': [{'center': [374, 333]}]})
        check('a repeated first stage after bidding is treated as grab and declined',
              clicked == [[604, 333]], str(clicked))
        bot.handle_bid(None, None, 1, 1, {'orange': [{'center': [374, 333]}]})
        events = [event for event in follow_events(root / 'events.jsonl', from_start=True, speech_only=False, once=True)
                  if event.get('event_kind') == 'decision']
        check('repeated bid decision does not speak twice',
              len(events) == 2 and events[0]['speak'] is True and events[1]['speak'] is False, str(events))

        def double_bot(buttons, power, clicks):
            b = mini_bot.Bot()
            b.events = StreamerEventStream(root / 'double_events.jsonl', root / 'double_latest.json', session_id='selftest')
            b.log = lambda record: None
            b.notify = lambda *args, **kwargs: None
            b.capture = lambda hwnd: (np.zeros((580, 933, 3), dtype=np.uint8), (0, 0, 933, 580))
            b.bot_decision = lambda *args, **kwargs: {'power': power, 'action': {'score': 3}}
            b.bid_done = True
            b.read_bid_buttons = lambda rgb, sx, sy: buttons
            b.click = lambda *args, **kwargs: clicks.append(args[2]) or True
            return b

        clicks = []
        super_and_substring = [
            {'text': '超级加倍', 'center': [265, 308], 'confidence': 0.9, 'source': 'rapidocr'},
            {'text': '加倍', 'center': [265, 308], 'confidence': 0.9, 'source': 'rapidocr'},
            {'text': '不加倍', 'center': [661, 307], 'confidence': 0.9, 'source': 'rapidocr'},
        ]
        bot = double_bot(super_and_substring, 24, clicks)
        bot.handle_bid(None, None, 1, 1, {'orange': [{'center': [374, 333]}]})
        check('super double is never clicked when only its substring matches double',
              clicks == [[661, 307]], str(clicks))

        clicks = []
        real_double = [
            {'text': '超级加倍', 'center': [265, 308], 'confidence': 0.9, 'source': 'rapidocr'},
            {'text': '加倍', 'center': [463, 307], 'confidence': 0.9, 'source': 'rapidocr'},
            {'text': '不加倍', 'center': [661, 307], 'confidence': 0.9, 'source': 'rapidocr'},
        ]
        bot = double_bot(real_double, 24, clicks)
        bot.handle_bid(None, None, 1, 1, {'orange': [{'center': [374, 333]}]})
        check('real double button is clicked away from the super double',
              clicks == [[463, 307]], str(clicks))

        def grab_bot(power, clicks, called=False):
            b = mini_bot.Bot()
            b.events = StreamerEventStream(root / 'grab_events.jsonl', root / 'grab_latest.json', session_id='selftest')
            b.log = lambda record: None
            b.notify = lambda *args, **kwargs: None
            b.capture = lambda hwnd: (np.zeros((580, 933, 3), dtype=np.uint8), (0, 0, 933, 580))
            b.bot_decision = lambda *args, **kwargs: {'power': power, 'action': {'score': 3}}
            b.bid_done = called
            b.read_bid_buttons = lambda rgb, sx, sy: [
                {'text': '抢地主', 'center': [357, 311], 'confidence': 0.9, 'source': 'rapidocr'},
                {'text': '不抢', 'center': [604, 335], 'confidence': 0.9, 'source': 'rapidocr'},
            ]
            b.click = lambda *args, **kwargs: clicks.append(args[2]) or True
            return b

        clicks = []
        bot = grab_bot(16, clicks)
        bot.handle_bid(None, None, 1, 1, {'orange': [{'center': [357, 311]}], 'blue': [{'center': [604, 335]}]})
        check('grab is declined below twenty power', clicks == [[604, 335]], str(clicks))

        clicks = []
        bot = grab_bot(24, clicks)
        bot.handle_bid(None, None, 1, 1, {'orange': [{'center': [357, 311]}], 'blue': [{'center': [604, 335]}]})
        check('grab is taken at twenty or more power when the bot has not bid', clicks == [[357, 311]], str(clicks))

        clicks = []
        bot = grab_bot(26, clicks, called=True)
        bot.handle_bid(None, None, 1, 1, {'orange': [{'center': [357, 311]}], 'blue': [{'center': [604, 335]}]})
        check('grab is taken again at twenty or more power after the bot called', clicks == [[357, 311]], str(clicks))

        clawed = []
        bot = grab_bot(24, clawed)
        for _ in range(3):
            bot.handle_bid(None, None, 1, 1, {'orange': [{'center': [357, 311]}], 'blue': [{'center': [604, 335]}]})
        check('the same bid click is sent at most twice', len(clawed) == 2, str(clawed))


def follow_retry_cases():
    import tempfile
    import time
    from streamer_events import StreamerEventStream, follow_events

    lead_rgb = load_rgb('outputs/mini_live_now12.png')
    result_rgb = load_rgb('outputs/mini_live_now.png')
    scale_x = lead_rgb.shape[1] / mini_bot.BASE_WIDTH
    scale_y = lead_rgb.shape[0] / mini_bot.BASE_HEIGHT
    ui = {'orange': [{'center': [644, 331]}]}
    states = iter([('turn', {}, ui), ('turn', {}, ui), ('result', {}, None)])
    original_classify = mini_bot.classify
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        bot = mini_bot.Bot()
        bot.events = StreamerEventStream(root / 'events.jsonl', root / 'latest.json', session_id='selftest')
        bot.log = lambda record: None
        bot.notify = lambda *args, **kwargs: None
        bot.role = 'landlord'
        bot.last_play = {'seat': 'left', 'ranks': ['7'], 'time': time.perf_counter()}
        frames = iter([(lead_rgb, None)] * 5 + [(result_rgb, None)])
        bot.capture = lambda hwnd: next(frames)
        bot.check_tuoguan = lambda *args: False
        bot.save_action_frame = lambda *args: None
        bot.bot_decision = lambda *args, **kwargs: {
            'candidates': [{'id': 'c000', 'type': 'single', 'ranks': ['9'], 'indices': [12], 'reasons': ['测试']}],
            'recommended': 'c000', 'can_pass': True,
        }
        bot.deepseek_pick = lambda *args, **kwargs: {'candidate_id': 'c000', 'reason': '测试', 'utterance': '出单9。'}
        selected = []
        bot.set_selection = lambda hwnd, bbox, obs, indices, sx, sy, limit=None: selected.append(list(indices)) or True
        bot.confirm_selection = lambda hwnd, bbox, rgb, chosen, sx, sy, **kwargs: (True, rgb)
        clicks = []

        def click(*args, **kwargs):
            clicks.append({'point': args[2], 'physical': kwargs.get('physical')})
            return True

        bot.click = click
        try:
            mini_bot.classify = lambda rgb, sx, sy: next(states)
            bot.handle_turn(None, (0, 0, 933, 580), scale_x, scale_y, ui)
        finally:
            mini_bot.classify = original_classify
        events = list(follow_events(root / 'events.jsonl', from_start=True, speech_only=False, once=True))
    check('failed follow retries the identical move once',
          selected == [[12], [12]] and [item['point'] for item in clicks] == [[644, 331], [644, 331]], str(clicks))
    check('follow retry uses physical clicks for the play button',
          all(item['physical'] is True for item in clicks), str(clicks))
    check('successful follow retry is silent and confirmed',
          [event['execution'] for event in events] == ['decided', 'confirmed']
          and events[1]['speak'] is False, str(events))


def hint_submit_retry_cases():
    import tempfile
    from streamer_events import StreamerEventStream, follow_events

    result_rgb = load_rgb('outputs/mini_live_now.png')
    ui = {'orange': [{'center': [644, 331]}]}
    states = iter([('turn', {}, ui), ('result', {}, None)])
    original_classify = mini_bot.classify
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        bot = mini_bot.Bot()
        bot.events = StreamerEventStream(root / 'events.jsonl', root / 'latest.json', session_id='selftest')
        bot.log = lambda record: None
        bot.notify = lambda *args, **kwargs: None
        bot.role = 'landlord'
        bot.capture = lambda hwnd: (result_rgb, None)
        bot.find_hint_button = lambda *args: None
        clicks = []

        def click(*args, **kwargs):
            clicks.append(args[2])
            return True

        bot.click = click
        try:
            mini_bot.classify = lambda rgb, sx, sy: next(states)
            bot.fallback_hint(None, (0, 0, 933, 580), 1, 1, ui, announce=False, target_seat='left')
        finally:
            mini_bot.classify = original_classify
        events = list(follow_events(root / 'events.jsonl', from_start=True, speech_only=False, once=True))
    check('hint submit retries the play button before passing',
          clicks == [[533, 331], [644, 331], [644, 331]], str(clicks))
    check('hint submit retry does not speak a rejection', [event['execution'] for event in events] == ['decided', 'confirmed'], str(events))


def room_select_cases():
    words = [
        {'text': '新', 'x': 235, 'y': 211, 'w': 35, 'h': 34},
        {'text': '手', 'x': 272, 'y': 212, 'w': 34, 'h': 33},
        {'text': '场', 'x': 308, 'y': 211, 'w': 33, 'h': 34},
        {'text': '普', 'x': 493, 'y': 211, 'w': 34, 'h': 34},
        {'text': '通', 'x': 529, 'y': 212, 'w': 34, 'h': 33},
        {'text': '场', 'x': 565, 'y': 211, 'w': 33, 'h': 34},
        {'text': '高', 'x': 750, 'y': 211, 'w': 34, 'h': 34},
        {'text': '级', 'x': 786, 'y': 212, 'w': 34, 'h': 33},
        {'text': '场', 'x': 822, 'y': 211, 'w': 33, 'h': 34},
    ]
    point = mini_bot.room_select_point(words, 800 / mini_bot.BASE_WIDTH, 496 / mini_bot.BASE_HEIGHT)
    plan = mini_bot.idle_scene_plan('欢乐经典 新手场 普通场 高级场 底分15')
    table_header = mini_bot.idle_scene_plan('欢乐经典新手场 底分15 倍数60')
    check('room select page recognized', plan['action'] == 'select_novice_room' and plan['phase'] == 'lobby', str(plan))
    check('novice room point lands on the novice card',
          point is not None and abs(point[0] - 336) <= 2 and abs(point[1] - 267) <= 2, str(point))
    check('table header alone does not trigger room select', table_header['action'] == 'unknown_scene', str(table_header))


def selection_confirm_cases():
    import numpy as np

    def obs(selected, tops=None, count=6):
        tops = tops or [0] * count
        glyphs = [[0, top, 1, 1] for top in tops]
        return {'selected_indices': selected, 'glyphs': glyphs, 'card_lefts': [0] * count,
                'pitch': 46, 'count': count, 'my_hand': ['3'] * count, 'unknown_indices': []}

    original = mini_perceive.read_hand
    bot = mini_bot.Bot()
    bot.log = lambda record: None
    bot.capture = lambda hwnd: (np.zeros((580, 933, 3), dtype=np.uint8), (0, 0, 933, 580))
    calls = []
    bot.set_selection = lambda hwnd, bbox, observation, indices, sx, sy, limit=None: calls.append(list(indices)) or True
    try:
        mini_perceive.read_hand = lambda image: obs([2, 3, 4])
        ok, _ = bot.confirm_selection(None, (0, 0, 933, 580), np.zeros((580, 933, 3), np.uint8), [2, 3, 4], 1, 1)
        check('correct selection needs no corrective click', ok and calls == [], str(calls))

        reads = [obs([2, 3]), obs([2, 3, 4])]
        mini_perceive.read_hand = lambda image: reads.pop(0) if reads else obs([2, 3, 4])
        calls.clear()
        ok, _ = bot.confirm_selection(None, (0, 0, 933, 580), np.zeros((580, 933, 3), np.uint8), [2, 3, 4], 1, 1)
        check('missed card is retried once and confirmed', ok and calls == [[2, 3, 4]], str(calls))

        reads = [obs([2, 3]), obs([2, 3]), obs([2, 3])]
        mini_perceive.read_hand = lambda image: reads.pop(0) if reads else obs([2, 3])
        calls.clear()
        ok, _ = bot.confirm_selection(None, (0, 0, 933, 580), np.zeros((580, 933, 3), np.uint8), [2, 3, 4], 1, 1)
        check('selection gives up after two corrections', not ok and calls == [[2, 3, 4], [2, 3, 4]], str(calls))

        reads = [obs([2, 3]), obs([2, 3]), obs([2, 3, 4])]
        mini_perceive.read_hand = lambda image: reads.pop(0) if reads else obs([2, 3, 4])
        calls.clear()
        ok, _ = bot.confirm_selection(None, (0, 0, 933, 580), np.zeros((580, 933, 3), np.uint8), [2, 3, 4], 1, 1)
        check('selection confirms after the second correction', ok and calls == [[2, 3, 4], [2, 3, 4]], str(calls))

        before = obs([], tops=[400, 400, 400, 400, 400, 400])
        mini_perceive.read_hand = lambda image: obs([], tops=[395, 400, 400, 400, 400, 400])
        calls.clear()
        ok, _ = bot.confirm_selection(None, (0, 0, 933, 580), np.zeros((580, 933, 3), np.uint8), [0], 1, 1, before=before)
        check('card lifted relative to the pre-click frame counts as selected', ok and calls == [], str(calls))

        mini_perceive.read_hand = lambda image: obs([2, 3, 9])
        check_record = bot.selection_check(np.zeros((580, 933, 3), np.uint8), [2, 3, 4])
        check('selection check reports missing and extra', check_record is not None
              and check_record['missing'] == [4] and check_record['extra'] == [9], str(check_record))

        before = obs([], tops=[400] * 7)
        bot.save_action_frame = lambda *args, **kwargs: None
        mini_perceive.read_hand = lambda image: obs([2, 3], tops=[424] * 12, count=12)
        calls.clear()
        ok, _ = bot.confirm_selection(None, (0, 0, 933, 580), np.zeros((580, 933, 3), np.uint8), [2, 3, 4], 1, 1, before=before)
        check('unstable hand read is retried like a missed selection', not ok and len(calls) == 2, str(calls))

        traced = []
        bot.log = traced.append
        bot.selection_trace = True
        mini_perceive.read_hand = lambda image: obs([2, 3])
        bot.sample_selection(None, [2, 3, 4], seconds=0.12, step=0.05)
        check('selection trace samples the hand over time',
              traced and all(item['state'] == 'selection_trace' and item['missing'] == [4] for item in traced),
              str(traced))
        bot.selection_trace = False

        batch_bot = mini_bot.Bot()
        batch_bot.log = lambda record: None
        clicked = []
        batch_bot.click = lambda *args, **kwargs: clicked.append(args[2]) or True
        ok = batch_bot.set_selection(None, (0, 0, 933, 580), obs([]), [0, 1, 2, 3, 4, 5], 1, 1, limit=3)
        check('long selection clicks at most three cards per batch', ok and len(clicked) == 3, str(clicked))

        bot.log = lambda record: None
        mini_perceive.read_hand = lambda image: obs([2, 3])
        calls.clear()
        ok, _ = bot.confirm_selection(None, (0, 0, 933, 580), np.zeros((580, 933, 3), np.uint8),
                                      [2, 3, 4, 5], 1, 1, before=obs([], count=6))
        check('selection stops after three batches', not ok and len(calls) == 2, str(calls))

        calls.clear()
        mini_perceive.read_hand = lambda image: obs([2, 3], count=12)
        ok, _ = bot.confirm_selection(None, (0, 0, 933, 580), np.zeros((580, 933, 3), np.uint8),
                                      list(range(12)), 1, 1, before=obs([], count=12))
        check('a twelve card play gets extra batches', not ok and len(calls) == 3, str(calls))
    finally:
        mini_perceive.read_hand = original


def selection_batch_progress_cases():
    from unittest.mock import patch
    import numpy as np

    def observation(selected, lifted):
        return {
            'count': 5,
            'pitch': 46,
            'card_lefts': [0, 46, 92, 138, 184],
            'glyphs': [[11 + index * 46, 380 if index in lifted else 400, 20, 30] for index in range(5)],
            'selected_indices': list(selected),
            'my_hand': ['6', '6', '6', '3', '3'],
            'unknown_indices': [],
        }

    before = observation([], [])
    partial = observation([], [0, 1, 2])
    complete = observation([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])
    bot = mini_bot.Bot()
    bot.log = lambda record: None
    rgb = np.zeros((580, 933, 3), dtype=np.uint8)
    bot.capture = lambda hwnd: (rgb, None)
    clicks = []
    bot.click = lambda *args, **kwargs: clicks.append(args[2]) or True
    with patch.object(mini_perceive, 'read_hand', side_effect=[partial, complete]):
        confirmed, _ = bot.confirm_selection(None, (0, 0, 933, 580), rgb, [0, 1, 2, 3, 4], 1, 1, before=before)
    expected = [bot.card_point(partial, index) for index in (3, 4)]
    check('next selection batch clicks only the remaining pair',
          confirmed and clicks == expected and partial['selected_indices'] == [0, 1, 2], str(clicks))


def selection_lift_accumulation_cases():
    def observation(selected, lifted):
        return {
            'count': 2,
            'pitch': 46,
            'card_lefts': [0, 46],
            'glyphs': [[11 + index * 46, 380 if index in lifted else 400, 20, 30] for index in range(2)],
            'selected_indices': list(selected),
            'my_hand': ['8', '8'],
            'unknown_indices': [],
        }

    def check_for(observation):
        return {'observation': observation, 'selected': list(observation['selected_indices']),
                'missing': [], 'extra': []}

    bot = mini_bot.Bot()
    initial = observation([], [])
    first = observation([], [0])
    first_check = bot.apply_lift(check_for(first), initial, [0, 1], 1)
    second = observation([], [0, 1])
    second_check = bot.apply_lift(check_for(second), first, [0, 1], 1)
    check('selection lift keeps cards selected by earlier batches',
          first_check['selected'] == [0] and second_check['selected'] == [0, 1]
          and second['selected_indices'] == [0, 1], str(second_check))
    lowered = observation([], [1])
    swap = bot.apply_lift(check_for(lowered), first, [0, 1], 1)
    check('selection lift removes a card that was actually lowered',
          swap['selected'] == [1] and lowered['selected_indices'] == [1], str(swap))


def capture_cases():
    from types import SimpleNamespace
    from unittest.mock import patch
    from PIL import Image

    bbox = (10, 20, 943, 600)
    rect = SimpleNamespace(left=1, top=2, right=952, bottom=609)
    valid = Image.new('RGB', (933, 580), (20, 30, 40))
    bot = mini_bot.Bot()
    bot.user32 = SimpleNamespace(GetForegroundWindow=lambda: 1, GetDpiForWindow=lambda hwnd: 120)
    logs = []
    bot.log = logs.append
    common = {
        'client_bbox': patch.object(mini_bot, 'client_bbox', return_value=bbox),
        'window_rect': patch.object(mini_bot, 'window_rect', return_value=rect),
    }
    with common['client_bbox'], common['window_rect']:
        with patch.object(mini_bot, 'print_window_image', return_value=valid):
            rgb, actual_bbox = bot.capture(1)
        check('PrintWindow capture records separate geometry diagnostics',
              actual_bbox == bbox and rgb.shape == (580, 933, 3)
              and logs[-1]['state'] == 'client_size' and logs[-1]['client_size'] == [933, 580]
              and logs[-1]['image_size'] == [933, 580] and logs[-1]['window_rect'] == [1, 2, 952, 609]
              and logs[-1]['dpi'] == 120 and logs[-1]['capture_method'] == 'PrintWindow', str(logs))

        logs.clear()
        wrong = Image.new('RGB', (650, 620))
        with patch.object(mini_bot, 'print_window_image', return_value=wrong), \
                patch.object(mini_bot.ImageGrab, 'grab', return_value=valid) as grab:
            rgb, _ = bot.capture(1)
        check('PrintWindow size mismatch falls back to ImageGrab',
              grab.call_count == 1 and rgb.shape == (580, 933, 3)
              and any(item['state'] == 'capture_size_mismatch' and item['capture_method'] == 'PrintWindow'
                      and item['client_size'] == [933, 580] and item['image_size'] == [650, 620] for item in logs)
              and logs[-1]['capture_method'] == 'ImageGrab', str(logs))

        logs.clear()
        with patch.object(mini_bot, 'print_window_image', return_value=None), \
                patch.object(mini_bot.ImageGrab, 'grab', return_value=wrong):
            try:
                bot.capture(1)
                error = None
            except mini_bot.CaptureError as caught:
                error = str(caught)
        mismatches = [item for item in logs if item['state'] == 'capture_size_mismatch']
        check('ImageGrab size mismatch fails closed after a fresh retry',
              error == 'capture_size_mismatch' and len(mismatches) == 2
              and all(item['capture_method'] == 'ImageGrab' for item in mismatches)
              and logs[-1]['state'] == 'capture_failed', str(logs))

        logs.clear()
        bot.user32.GetForegroundWindow = lambda: False
        with patch.object(mini_bot, 'print_window_image', return_value=None), \
                patch.object(mini_bot.ImageGrab, 'grab') as grab:
            try:
                bot.capture(1)
                error = None
            except mini_bot.CaptureError as caught:
                error = str(caught)
        check('occluded capture failure never returns a black frame',
              error == 'printwindow_failed_and_not_foreground' and grab.call_count == 0
              and logs[-1]['state'] == 'capture_failed'
              and logs[-1]['reason'] == 'printwindow_failed_and_not_foreground', str(logs))


def overlay_geometry_cases():
    from types import SimpleNamespace
    from unittest.mock import patch
    import overlay

    beside = overlay.overlay_geometry((426, 111, 1241, 614), 1920, 1080, 600, 330)
    below = overlay.overlay_geometry((426, 111, 1241, 614), 1600, 1080, 600, 330)
    check('overlay is placed to the right of the game window', beside == (1249, 111), str(beside))
    check('overlay falls below when the right side is too narrow', below == (426, 622), str(below))
    check('overlay hides instead of covering a game that fills the screen',
          overlay.overlay_geometry((0, 0, 1000, 700), 1000, 700, 600, 330) is None, '')

    class OverlayStub:
        def __init__(self):
            self.rects = []
            self.on_takeover = None

        def place_outside(self, game_rect):
            self.rects.append(game_rect)

    overlay_stub = OverlayStub()
    bot = mini_bot.Bot(overlay=overlay_stub)
    first = SimpleNamespace(left=10, top=20, right=952, bottom=609)
    moved = SimpleNamespace(left=30, top=40, right=972, bottom=629)
    with patch.object(mini_bot, 'window_rect', side_effect=[first, first, moved]):
        bot.sync_overlay(1)
        bot.sync_overlay(1)
        bot.sync_overlay(1)
    check('overlay follows actual window geometry changes only',
          overlay_stub.rects == [(10, 20, 952, 609), (30, 40, 972, 629)], str(overlay_stub.rects))


def lead_selection_fallback_cases():
    import tempfile
    from streamer_events import StreamerEventStream

    lead_rgb = load_rgb('outputs/mini_live_now12.png')
    observation = mini_perceive.read_hand(cv2.cvtColor(lead_rgb, cv2.COLOR_RGB2BGR))
    scale_x = lead_rgb.shape[1] / mini_bot.BASE_WIDTH
    scale_y = lead_rgb.shape[0] / mini_bot.BASE_HEIGHT
    _, _, ui = mini_bot.classify(lead_rgb, scale_x, scale_y)
    key = tuple(observation['my_hand'])
    index = next(i for i, rank in enumerate(observation['my_hand']) if rank != 'UNKNOWN')
    rank = observation['my_hand'][index]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        bot = mini_bot.Bot()
        bot.events = StreamerEventStream(root / 'events.jsonl', root / 'latest.json', session_id='selftest')
        bot.log = lambda record: None
        bot.notify = lambda *args, **kwargs: None
        bot.role = 'farmer'
        bot.lead_cache_key = (key, bot.role)
        bot.lead_cache = {
            'chosen': [index],
            'candidate': {'id': 'cached', 'type': 'single', 'ranks': [rank], 'indices': [index]},
            'decision': {'reason': '缓存', 'utterance': f'出{rank}。'},
        }
        bot.capture = lambda hwnd: (lead_rgb, None)
        bot.check_tuoguan = lambda *args: False
        bot.set_selection = lambda *args, **kwargs: True
        bot.confirm_selection = lambda *args, **kwargs: (False, lead_rgb)
        fallback = {
            'chosen': [index],
            'candidate': {'id': 'selection_fallback', 'type': 'single', 'ranks': ['3'], 'indices': [index]},
            'decision': {'reason': '换牌', 'utterance': '改出3。'},
        }
        bot.pick_lead_selection_fallback = lambda *args: fallback
        clicks = []
        bot.click = lambda *args, **kwargs: clicks.append(args[2]) or True
        bot.last_selection_check = {'selected': [0, 1, 2]}
        bot.handle_lead(None, None, scale_x, scale_y, ui)
        check('residual selection blocks fallback instead of playing a mixed move',
              bot.lead_blocked_key == key and bot.lead_cache != fallback and clicks == [], str(bot.lead_cache))
        bot.lead_blocked_key = None
        bot.last_selection_check = {'selected': []}
        bot.handle_lead(None, None, scale_x, scale_y, ui)
        check('failed lead selection swaps to another legal move', bot.lead_cache == fallback, str(bot.lead_cache))
        check('failed lead selection does not press play', clicks == [], str(clicks))
        bot.handle_lead(None, None, scale_x, scale_y, ui)
        check('second selection failure blocks the hand', bot.lead_blocked_key == key, str(bot.lead_blocked_key))

        helper = mini_bot.Bot()
        helper.log = lambda record: None
        helper.bot_decision = lambda *args, **kwargs: {'candidates': [
            {'id': 'c1', 'type': 'triple', 'ranks': ['K', 'K', 'K'], 'indices': [0, 1, 2]},
            {'id': 'c2', 'type': 'single', 'ranks': ['9'], 'indices': [3]},
        ]}
        alternative = helper.pick_lead_selection_fallback(['K', 'K', 'K', '9'], [0, 1, 2, 3])
        check('lead selection fallback prefers a single',
              alternative is not None and alternative['candidate']['ranks'] == ['9']
              and alternative['chosen'] == [3], str(alternative))


def table_glyph_cases():
    templates = mini_perceive.load_templates()
    image = mini_perceive.load_image(fixture_path('outputs/stuck_state.png'))
    for scale in (1.0, 0.9, 0.8):
        img = image if scale == 1.0 else cv2.resize(
            image, (round(image.shape[1] * scale), round(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        played = mini_perceive.detect_played(img, templates)
        check(f'table straight keeps 5 and 6 apart at {scale}',
              played['left']['ranks'] == ['J', '10', '9', '8', '7', '6', '5'] and played['left']['reliable'], str(played['left']))
    table = mini_perceive.load_table_templates()
    check('table templates cover all single-digit ranks',
          {'2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'K', 'A'} <= set(table), str(sorted(table)))
    six = mini_perceive.resolve_played_rank([(0.60, '5'), (0.58, '6')], table['6'])
    five = mini_perceive.resolve_played_rank([(0.60, '6'), (0.58, '5')], table['5'])
    check('ambiguous 5/6 tie-breaks on the closed loop', six == (0.58, '6'), str(six))
    check('ambiguous 5/6 tie-breaks on the open hook', five == (0.58, '5'), str(five))


def follow_cache_cases():
    import tempfile
    import time
    from streamer_events import StreamerEventStream, follow_events

    lead_rgb = load_rgb('outputs/mini_live_now12.png')
    observation = mini_perceive.read_hand(cv2.cvtColor(lead_rgb, cv2.COLOR_RGB2BGR))
    rules_hand = [rank for rank in observation['my_hand'] if rank != 'UNKNOWN']
    scale_x = lead_rgb.shape[1] / mini_bot.BASE_WIDTH
    scale_y = lead_rgb.shape[0] / mini_bot.BASE_HEIGHT
    ui = {'orange': [{'center': [644, 331]}]}
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        bot = mini_bot.Bot()
        bot.events = StreamerEventStream(root / 'events.jsonl', root / 'latest.json', session_id='selftest')
        bot.log = lambda record: None
        bot.notify = lambda *args, **kwargs: None
        bot.role = 'landlord'
        bot.last_play = {'seat': 'left', 'ranks': ['7'], 'time': time.perf_counter()}
        bot.capture = lambda hwnd: (lead_rgb, None)
        bot.check_tuoguan = lambda *args: False
        bot.follow_cache_key = (tuple(rules_hand), 'left', ('7',), 'landlord')
        bot.follow_cache = {
            'chosen': [],
            'candidate': {'id': 'pass', 'type': 'pass', 'ranks': [], 'indices': [], 'reasons': ['放弃这一手']},
            'decision': {'reason': '缓存过牌', 'utterance': '选择不出。'},
        }
        bot.bot_decision = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('cache must skip engine'))
        bot.deepseek_pick = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('cache must skip deepseek'))
        clicks = []
        bot.click = lambda *args, **kwargs: clicks.append(args[2]) or True
        bot.handle_turn(None, (0, 0, 933, 580), scale_x, scale_y, ui)
        events = list(follow_events(root / 'events.jsonl', from_start=True, speech_only=False, once=True))
    check('follow cache reuses the pass without re-deciding', clicks == [[316, 331]], str(clicks))
    check('follow cache hit is silent',
          bool(events) and events[0]['source'] == 'cached_decision' and events[0]['speak'] is False, str(events))


def watchdog_reset_cases():
    bot = mini_bot.Bot()
    bot.last_hand_count = 10
    bot.click(None, None, [1, 2], 1, 1, physical=False)
    check('click resets the auto-play watchdog baseline', bot.last_hand_count is None, str(bot.last_hand_count))


def off_by_one_target_cases():
    played = {
        'left': {'ranks': ['6', '6', '6', '5', '5', '5'], 'count_est': 7, 'reliable': False},
        'right': {'ranks': [], 'count_est': 0, 'reliable': True},
    }
    seat, ranks = mini_bot.off_by_one_target(played)
    check('off-by-one pile is offered for legality validation',
          seat == 'left' and ranks == ['6', '6', '6', '5', '5', '5'], str((seat, ranks)))
    played['left'] = {'ranks': ['K', 'K', '9', '5', '3'], 'count_est': 6, 'reliable': False}
    seat, _ = mini_bot.off_by_one_target(played)
    check('illegal-looking pile is still passed to the rules engine', seat == 'left', str(seat))
    played['left'] = {'ranks': ['6', '6', '6', '5', '5', '5'], 'count_est': 6, 'reliable': True}
    seat, _ = mini_bot.off_by_one_target(played)
    check('reliable pile is not treated as off-by-one', seat is None, str(seat))


def unreliable_target_cases():
    import time
    original = mini_perceive.detect_played
    try:
        pile = {'ranks': ['8', '8', '8', '10'], 'cards': [], 'count_est': 5, 'reliable': False}
        empty = {'ranks': [], 'cards': [], 'count_est': 0, 'reliable': True}
        mini_perceive.detect_played = lambda image: {'left': empty, 'right': pile}
        bot = mini_bot.Bot()
        bot.log = lambda record: None
        bot.update_counts_from_badges = lambda *args, **kwargs: None
        bot.last_play = {'seat': 'left', 'ranks': ['9'], 'time': 0}
        rgb = load_rgb('outputs/mini_live_now9.png')
        for _ in range(3):
            bot.last_play_probe = 0.0
            bot.track_plays(rgb)
            time.sleep(0.35)
        check('new unreliable pile clears the stale target', bot.last_play is None, str(bot.last_play))
        check('unreliable pile still subtracts its estimated count', bot.counts['right'] == 12, str(bot.counts))
        check('unreliable pile marks the seat as played', bot.has_played['right'] is True, str(bot.has_played))
    finally:
        mini_perceive.detect_played = original


def count_badge_cases():
    class FakeOCR:
        def __init__(self, text):
            self.text = text
            self.calls = 0
        def __call__(self, image):
            self.calls += 1
            return [([[0, 0], [1, 0], [1, 1], [0, 1]], self.text, 0.9)], 0.1

    bot = mini_bot.Bot()
    bot.log = lambda record: None
    rgb = load_rgb('outputs/mini_live_now12.png')
    sx = rgb.shape[1] / mini_bot.BASE_WIDTH
    sy = rgb.shape[0] / mini_bot.BASE_HEIGHT
    bot._rapidocr = FakeOCR('17')
    check('badge read parses the two-digit count', bot.read_badge_count(rgb, 'left', sx, sy) == 17, '')

    bot._rapidocr = FakeOCR('9')
    check('badge read rejects a partial single digit', bot.read_badge_count(rgb, 'left', sx, sy) is None, '')

    single_samples = [
        ('outputs/follow_rejected_092035_518235.png', 'left', 9),
        ('outputs/follow_uncertain_085042_917630.png', 'right', 4),
        ('outputs/selection_unstable_113049_426018.png', 'left', 1),
        ('outputs/selection_unstable_113049_426018.png', 'right', 1),
        ('outputs/follow_uncertain_105033_082218.png', 'left', 6),
        ('outputs/follow_uncertain_111917_906464.png', 'right', 8),
        ('outputs/lead_rejected_080403_834098.png', 'right', 3),
    ]
    for path, seat, expected in single_samples:
        sample_rgb = load_rgb(path)
        sample_sx = sample_rgb.shape[1] / mini_bot.BASE_WIDTH
        sample_sy = sample_rgb.shape[0] / mini_bot.BASE_HEIGHT
        bot._rapidocr = FakeOCR('')
        actual = bot.read_badge_count(sample_rgb, seat, sample_sx, sample_sy)
        check(f'badge template reads {expected} for {seat} in {path}', actual == expected, str(actual))

    single_bot = mini_bot.Bot()
    single_bot.log = lambda record: None
    single_bot._rapidocr = FakeOCR('')
    single_bot.counts = {'left': 12, 'right': 17}
    single_rgb = load_rgb('outputs/follow_rejected_092035_518235.png')
    single_sx = single_rgb.shape[1] / mini_bot.BASE_WIDTH
    single_sy = single_rgb.shape[0] / mini_bot.BASE_HEIGHT
    single_bot.last_count_probe = 0.0
    single_bot.update_counts_from_badges(single_rgb, single_sx, single_sy)
    pending = single_bot.counts['left']
    single_bot.last_count_probe = 0.0
    single_bot.update_counts_from_badges(single_rgb, single_sx, single_sy)
    check('single-digit badge needs two matching reads',
          pending == 12 and single_bot.counts['left'] == 9, str((pending, single_bot.counts)))

    bot._rapidocr = FakeOCR('17')
    bot.counts = {'left': 15, 'right': 9}
    bot.last_count_probe = 0.0
    bot.update_counts_from_badges(rgb, sx, sy)
    check('badge reads below ten but holds a sudden jump for confirmation',
          bot.counts == {'left': 17, 'right': 9} and bot._rapidocr.calls == 2
          and bot.badge_last_read['right'] == 17,
          str(bot.counts) + ' calls=' + str(bot._rapidocr.calls))

    bot._rapidocr = FakeOCR('17')
    bot.counts = {'left': 11, 'right': 17}
    bot.has_played = {'left': True, 'right': False}
    bot.last_count_probe = 0.0
    bot.update_counts_from_badges(rgb, sx, sy)
    check('stale large badge increase is rejected', bot.counts['left'] == 11, str(bot.counts))

    bot._rapidocr = FakeOCR('12')
    bot.counts = {'left': 11, 'right': 17}
    bot.last_count_probe = 0.0
    bot.update_counts_from_badges(rgb, sx, sy)
    check('small badge correction is accepted', bot.counts['left'] == 12, str(bot.counts))

    bot._rapidocr = FakeOCR('12')
    bot.counts = {'left': 17, 'right': 17}
    bot.has_played = {'left': True, 'right': False}
    bot.badge_last_read = {'left': None, 'right': None}
    bot.last_count_probe = 0.0
    bot.update_counts_from_badges(rgb, sx, sy)
    pending = bot.counts['left']
    bot.last_count_probe = 0.0
    bot.update_counts_from_badges(rgb, sx, sy)
    check('a sudden badge drop needs two matching reads',
          pending == 17 and bot.counts['left'] == 12, str((pending, bot.counts)))

    bot.counts = {'left': 17, 'right': 17}
    bot.badge_last_read = {'left': None, 'right': None}
    bot._rapidocr = FakeOCR('12')
    bot.last_count_probe = 0.0
    bot.update_counts_from_badges(rgb, sx, sy)
    bot._rapidocr = FakeOCR('14')
    bot.last_count_probe = 0.0
    bot.update_counts_from_badges(rgb, sx, sy)
    bot._rapidocr = FakeOCR('11')
    bot.last_count_probe = 0.0
    bot.update_counts_from_badges(rgb, sx, sy)
    check('a flickering badge never overrides the count', bot.counts['left'] == 17, str(bot.counts))

    bot = mini_bot.Bot()
    bot.log = lambda record: None
    bot.counts = {'left': 12, 'right': 17}
    bot.badge_counts = {'left': 12, 'right': None}
    bot.badge_at = {'left': 5.0, 'right': 0.0}
    bot.play_at = {'left': 0.0, 'right': 0.0}
    counted = bot.apply_play_count('left', 8)
    check('a play the badge already reflects is not subtracted twice',
          counted is False and bot.counts['left'] == 12, str(bot.counts))
    counted = bot.apply_play_count('left', 3)
    check('a later play is subtracted after the badge read',
          counted is True and bot.counts['left'] == 9, str(bot.counts))

    original_landlord = mini_perceive.detect_landlord
    try:
        mini_perceive.detect_landlord = lambda image: 'left'
        bot = mini_bot.Bot()
        bot.log = lambda record: None
        bot.counts = {'left': 4, 'right': 17}
        bot.has_played = {'left': True, 'right': False}
        bot.played_total = {'left': 8, 'right': 0}
        bot.update_role_from_table(load_rgb('outputs/mini_live_now12.png'))
        check('late landlord detection rebuilds the count from twenty',
              bot.counts['left'] == 12, str(bot.counts))
        bot = mini_bot.Bot()
        bot.log = lambda record: None
        bot.counts = {'left': 15, 'right': 17}
        bot.has_played = {'left': True, 'right': False}
        bot.played_total = {'left': 8, 'right': 0}
        bot.badge_counts = {'left': 15, 'right': None}
        bot.update_role_from_table(load_rgb('outputs/mini_live_now12.png'))
        check('late landlord count prefers the confirmed badge',
              bot.counts['left'] == 15, str(bot.counts))
    finally:
        mini_perceive.detect_landlord = original_landlord


def unreliable_stability_cases():
    import time
    original = mini_perceive.detect_played
    logs = []
    bot = mini_bot.Bot()
    bot.log = logs.append
    info = {'ranks': ['9', '4', '9', '9'], 'cards': [], 'count_est': 6, 'reliable': False}
    empty = {'ranks': [], 'cards': [], 'count_est': 0, 'reliable': True}
    mini_perceive.detect_played = lambda image: {'left': info, 'right': empty}
    bot.update_counts_from_badges = lambda *args, **kwargs: None
    rgb = load_rgb('outputs/mini_live_now12.png')

    def feed():
        bot.last_play_probe = 0.0
        bot.track_plays(rgb)

    try:
        feed()
        check('a first unreliable read is not acted on', bot.counts['left'] == 17, str(bot.counts))
        time.sleep(0.3)
        feed()
        check('a repeated unreliable read still waits for stability', bot.counts['left'] == 17, str(bot.counts))
        time.sleep(0.4)
        feed()
        check('a stable unreliable read is accepted once',
              bot.counts['left'] == 11 and len([log for log in logs if log.get('state') == 'play_count_estimated']) == 1,
              str(bot.counts))
    finally:
        mini_perceive.detect_played = original


def eligible_deepseek_cases():
    bot = mini_bot.Bot()
    shortlist = [
        {'id': 'c000', 'local_score': 67.5, 'ranks': ['A']},
        {'id': 'c001', 'local_score': 65.5, 'ranks': ['2']},
        {'id': 'c002', 'local_score': 65.5, 'ranks': ['LJ']},
        {'id': 'pass', 'local_score': 50, 'type': 'pass', 'ranks': []},
    ]
    eligible = bot.eligible_candidates(shortlist, 'c000')
    check('recommendation uses the two nearest alternatives without forcing pass',
          [item['id'] for item in eligible] == ['c000', 'c001', 'c002'],
          str([item['id'] for item in eligible]))
    eligible = bot.eligible_candidates(shortlist, 'pass')
    check('recommended pass has two normal alternatives',
          [item['id'] for item in eligible] == ['pass', 'c001', 'c002'],
          str([item['id'] for item in eligible]))
    shortlist[0]['local_score'] = 52.0
    eligible = bot.eligible_candidates(shortlist, 'c000')
    check('pass is included naturally when its real score is close',
          [item['id'] for item in eligible] == ['c000', 'pass', 'c001'],
          str([item['id'] for item in eligible]))
    result = {'recommended': 'c000', 'can_pass': True, 'candidates': [shortlist[0], shortlist[-1]]}
    check('follow shortlist removes pass when a real play is recommended',
          [item['id'] for item in mini_bot.follow_shortlist(result)] == ['c000'], '')
    no_moves = {'recommended': 'pass', 'can_pass': True, 'candidates': []}
    check('only pass remains when no legal play exists',
          [item['id'] for item in mini_bot.follow_shortlist(no_moves)] == ['pass'], '')
    actual = bot.bot_decision('playing', ['K', 'K'], role='farmer', target=['J', 'J'],
                              target_seat='left', can_pass=True, mode='candidates')
    check('engine output omits pass when the official action is a legal play',
          actual.get('recommended_source') == 'bot.js' and actual['action'].get('type') == 'play'
          and not any(item['id'] == 'pass' for item in actual['candidates']), str(actual.get('recommended')))
    passive = bot.bot_decision('playing', ['3', '3', '3', '3', '4', '5'], role='farmer', target=['2'],
                               target_seat='left', can_pass=True, mode='candidates')
    check('engine output keeps pass when pass is the official action',
          passive.get('recommended') == 'pass'
          and any(item['id'] == 'pass' for item in passive['candidates']), str(passive.get('recommended')))


def four_two_wing_cases():
    bot = mini_bot.Bot()
    hand = ['LJ', '2', 'A', 'A', 'A', '10', '8', '7', '7', '7', '7', '6', '6', '6', '6', '5', '5']
    result = bot.bot_decision('playing', hand, role='farmer', target=['3', '3', '3', '3', 'J', '4'],
                              target_seat='right', can_pass=True, mode='candidates')
    best = result['candidates'][0]
    four_two = [item for item in result['candidates'] if item['type'] == 'four_two' and not item['breaks_bomb']]
    check('four with a pair of wings is offered without breaking the other bomb',
          four_two and four_two[0]['ranks'] == ['6', '6', '6', '6', '5', '5'], str(best))

    hand = ['LJ', '2', 'A', 'A', 'A', '10', '8', '7', '7', '7', '5']
    result = bot.bot_decision('playing', hand, role='farmer', target=['K'],
                              target_seat='right', can_pass=True, mode='candidates')
    check('a single two is offered as a candidate for a single K',
          any(item['ranks'] == ['2'] for item in result['candidates']), str(result['candidates']))


def endgame_count_cases():
    counts = {'left': 5, 'right': 15}
    adjusted = mini_bot.counts_for_target(counts, 'left', ['J', 'J', 'J', '8'], True)
    kept = mini_bot.counts_for_target(counts, 'left', ['J', 'J', 'J', '8'], False)
    check('target seat count is reduced by the current play',
          adjusted['left'] == 1 and kept['left'] == 5, str((adjusted, kept)))

    bot = mini_bot.Bot()
    bot.role = 'landlord'
    bomb = bot.bot_decision('playing', ['LJ', '2', 'A', 'A', 'A', 'A'], role='landlord',
                            target=['J', 'J', 'J', '8'], target_seat='left', can_pass=True,
                            counts={'left': 1, 'right': 15}, mode='candidates')
    passive = bot.bot_decision('playing', ['LJ', '2', 'A', 'A', 'A', 'A'], role='landlord',
                               target=['J', 'J', 'J', '8'], target_seat='left', can_pass=True,
                               counts={'left': 5, 'right': 15}, mode='candidates')
    check('engine bombs when the target seat has one card',
          (bomb.get('action') or {}).get('type') == 'play'
          and bomb['action'].get('ranks') == ['A', 'A', 'A', 'A'], str(bomb.get('action')))
    check('engine passes when the target seat still has five cards',
          (passive.get('action') or {}).get('type') == 'pass', str(passive.get('action')))


def main():
    (ROOT / 'outputs').mkdir(exist_ok=True)
    classify_cases()
    recognition_cases()
    hand_read_retry_cases()
    strategy_cases()
    bot_engine_cases()
    follow_bomb_wing_cases()
    candidate_cases()
    played_cases()
    alignment_cases()
    play_tracking_cases()
    landlord_cases()
    tuoguan_cases()
    strength_cases()
    lead_execution_event_cases()
    streamer_event_cases()
    correction_event_cases()
    scene_plan_cases()
    purchase_scene_cases()
    post_trial_guard_cases()
    result_outcome_cases()
    continue_point_cases()
    scaled_selection_cases()
    role_lock_cases()
    bad_play_regression_cases()
    role_prompt_cases()
    execution_evidence_cases()
    teammate_fallback_cases()
    hint_button_cases()
    pass_only_retry_cases()
    bid_button_cases()
    bid_recognition_cases()
    click_guard_cases()
    takeover_cases()
    tuoguan_template_failclosed_cases()
    bid_stage_cases()
    follow_retry_cases()
    hint_submit_retry_cases()
    room_select_cases()
    table_glyph_cases()
    selection_confirm_cases()
    selection_batch_progress_cases()
    selection_lift_accumulation_cases()
    capture_cases()
    overlay_geometry_cases()
    lead_selection_fallback_cases()
    follow_cache_cases()
    watchdog_reset_cases()
    off_by_one_target_cases()
    unreliable_target_cases()
    count_badge_cases()
    unreliable_stability_cases()
    eligible_deepseek_cases()
    four_two_wing_cases()
    endgame_count_cases()
    failed = [result for result in RESULTS if not result['pass']]
    print(json.dumps({'total': len(RESULTS), 'failed': len(failed), 'failures': failed}, ensure_ascii=True, indent=2))
    (ROOT / 'outputs' / 'mini_selftest.json').write_text(
        json.dumps({'total': len(RESULTS), 'failed': len(failed), 'failures': failed}, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()
