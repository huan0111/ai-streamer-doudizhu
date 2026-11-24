import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
TEMPLATE_DIR = ROOT / 'templates' / 'mini_skin'
TABLE_TEMPLATE_DIR = ROOT / 'templates' / 'table_skin'
_TABLE_TEMPLATES = None
BASE_WIDTH = 933
BASE_HEIGHT = 580
PITCH = 46
GLYPH_LEFT_OFFSET = 11
LIFT_TOP = 393
THRESHOLD = 0.7
GLYPH_SIZE = (40, 40)

RANK_ORDER = ['3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A', '2', 'LJ', 'BJ']

AVATAR_ROIS = {
    'left': (0, 170, 170, 320),
    'right': (760, 170, 933, 320),
}

TABLE_ROIS = {
    'left': (140, 160, 460, 300),
    'right': (540, 160, 835, 300),
}


def load_image(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'cannot read image: {path}')
    return image


def save_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f'cannot encode image: {path}')
    data.tofile(str(path))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def scale_for(image):
    scale = min(image.shape[1] / BASE_WIDTH, image.shape[0] / BASE_HEIGHT)
    return scale, scale


def glyph_boxes(image):
    scale_x, scale_y = scale_for(image)
    x0 = round(2 * scale_x)
    x1 = round(933 * scale_x)
    y0 = round(366 * scale_y)
    y1 = round(460 * scale_y)
    roi = image[y0:y1, x0:x1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 2] < 170) | (hsv[:, :, 1] > 60)).astype(np.uint8) * 255
    white = (np.all(roi > 175, axis=2)).astype(np.uint8)
    white = cv2.dilate(white, np.ones((5, 5), np.uint8))
    mask = cv2.bitwise_and(mask, white * 255)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if not (8 * scale_x <= width <= 45 * scale_x and 22 * scale_y <= height <= 50 * scale_y):
            continue
        area = cv2.contourArea(contour)
        if area < 40:
            continue
        candidates.append((x, y, width, height, area / (width * height)))
    boxes = []
    candidates.sort(key=lambda item: item[0])
    for index, item in enumerate(candidates):
        x, y, width, height, density = item
        if density < 0.18:
            continue
        polluted = index == 0 and density < 0.25 and height >= 40 * scale_y and any(
            other is not item
            and 0 <= other[0] - (x + width) < 10 * scale_x
            and abs(other[1] - y) < 10 * scale_y
            and other[3] < height
            and other[4] > density * 1.5
            for other in candidates
        )
        if not polluted:
            boxes.append([x + x0, y + y0, width, height])
    boxes.sort(key=lambda box: box[0])
    merged = []
    for box in boxes:
        if merged:
            last = merged[-1]
            gap = box[0] - (last[0] + last[2])
            if gap < 10 * scale_x and abs(box[1] - last[1]) < 10 * scale_y:
                left = min(last[0], box[0])
                top = min(last[1], box[1])
                right = max(last[0] + last[2], box[0] + box[2])
                bottom = max(last[1] + last[3], box[1] + box[3])
                merged[-1] = [left, top, right - left, bottom - top]
                continue
        merged.append(box)
    rank_boxes = []
    for box in merged:
        center_x = box[0] + box[2] / 2
        has_rank_above = any(
            other is not box
            and 12 * scale_y <= box[1] - other[1] <= 55 * scale_y
            and abs(center_x - (other[0] + other[2] / 2)) <= 30 * scale_x
            for other in merged
        )
        if not has_rank_above:
            rank_boxes.append(box)
    return rank_boxes


def crop_glyph(image, box):
    scale_x, scale_y = scale_for(image)
    x, y, width, height = box[:4]
    if width == 0 or height == 0:
        raise ValueError('empty glyph')
    margin = round(2 * scale_x)
    top = max(0, y - margin)
    left = max(0, x - margin)
    bottom = min(image.shape[0], y + height + margin)
    right = min(image.shape[1], x + width + margin)
    roi = image[top:bottom, left:right]
    return cv2.resize(roi, GLYPH_SIZE, interpolation=cv2.INTER_AREA)


def cluster_row(boxes):
    boxes = sorted(boxes, key=lambda box: box[0])
    if len(boxes) > 2:
        gaps = [boxes[index + 1][0] - boxes[index][0] for index in range(len(boxes) - 1)]
        typical_gap = float(np.median(gaps))
        splits = [0] + [index + 1 for index, gap in enumerate(gaps) if gap > typical_gap * 2.2] + [len(boxes)]
        segments = [boxes[start:end] for start, end in zip(splits, splits[1:])]
        boxes = max(segments, key=len)
    while len(boxes) > 2:
        xs = [box[0] for box in boxes]
        nearest = [min(abs(xs[index] - xs[other]) for other in range(len(xs)) if other != index)
                   for index in range(len(xs))]
        median = float(np.median(nearest))
        loneliest = max(range(len(boxes)), key=lambda index: nearest[index])
        if nearest[loneliest] <= median * 2.2:
            break
        boxes.pop(loneliest)
    return boxes


def drop_isolated_glyphs(boxes):
    if len(boxes) < 3:
        return boxes
    gaps = [boxes[index + 1][0] - boxes[index][0] for index in range(len(boxes) - 1)]
    typical_gap = float(np.median(gaps))
    typical_top = float(np.median([box[1] for box in boxes[:-1]]))
    while len(boxes) >= 3:
        last = boxes[-1]
        if last[0] - boxes[-2][0] > typical_gap * 1.5 and last[1] < typical_top - 8:
            boxes = boxes[:-1]
        else:
            break
    return boxes


def detect_hand(image):
    boxes = drop_isolated_glyphs(cluster_row(glyph_boxes(image)))
    if not boxes:
        raise ValueError('no card glyphs found')
    scale_x, scale_y = scale_for(image)
    xs = [box[0] for box in boxes]
    ys = [box[1] for box in boxes]
    pitch = float(np.median([xs[index + 1] - xs[index] for index in range(len(xs) - 1)])) if len(xs) > 1 else PITCH * scale_x
    baseline = int(np.median(ys))
    first = xs[0]
    last = xs[-1]
    row_left = round(first - GLYPH_LEFT_OFFSET * scale_x)
    count = max(1, min(20, int(round((last - first) / pitch)) + 1))
    slots = []
    used = set()
    for index in range(count):
        slot_left = round(row_left + index * pitch)
        target = slot_left + GLYPH_LEFT_OFFSET * scale_x
        nearest = None
        nearest_distance = pitch * 0.5
        for box_index, box in enumerate(boxes):
            if box_index in used:
                continue
            distance = abs(box[0] - target)
            if distance < nearest_distance:
                nearest_distance = distance
                nearest = box_index
        if nearest is None:
            slots.append([target, baseline, 0, 0])
        else:
            used.add(nearest)
            slots.append(list(boxes[nearest]))
    while slots and slots[-1][2] == 0:
        slots.pop()
    if not slots:
        raise ValueError('no card slots found')
    lift_top = round(LIFT_TOP * scale_y)
    selected = [index for index, box in enumerate(slots) if box[2] and box[1] <= lift_top]
    return {
        'count': len(slots),
        'first_left': row_left,
        'pitch': round(pitch, 2),
        'card_lefts': [round(row_left + index * pitch) for index in range(len(slots))],
        'glyphs': slots,
        'selected_indices': selected,
    }


def build_templates(image, ranks, only_missing=False):
    detection = detect_hand(image)
    if len(ranks) != detection['count']:
        raise ValueError(f'label count {len(ranks)} does not match detected {detection["count"]}')
    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    sources = {}
    for rank, box in zip(ranks, detection['glyphs']):
        if box[2] == 0 or box[3] == 0:
            continue
        path = TEMPLATE_DIR / f'{rank}.png'
        if only_missing and path.exists():
            continue
        save_image(path, crop_glyph(image, box))
        sources[rank] = box
    return sources


def load_templates():
    templates = {}
    for path in TEMPLATE_DIR.glob('*.png'):
        templates[path.stem] = load_image(path)
    return templates


def glyph_shape(image):
    if image.ndim == 2:
        return image
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return ((hsv[:, :, 2] < 170) | (hsv[:, :, 1] > 60)).astype(np.uint8) * 255


def match_glyph(glyph, templates):
    scores = []
    shape = glyph_shape(glyph)
    for rank, template in templates.items():
        if rank in {'BJ', 'LJ'}:
            source, candidate = glyph, template
        else:
            source, candidate = shape, glyph_shape(template)
        result = cv2.matchTemplate(source, candidate, cv2.TM_CCOEFF_NORMED)
        _, score, _, _ = cv2.minMaxLoc(result)
        scores.append((float(score), rank))
    scores.sort(reverse=True)
    return scores


def load_badge_templates():
    templates = {}
    for path in sorted((TEMPLATE_DIR.parent / 'mini_badges').glob('*.png')):
        data = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            templates.setdefault(path.stem.split('_')[0], []).append(image)
    return templates


def badge_glyph_mask(image):
    crop = cv2.resize(image, (44, 44), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 1] < 110) & (hsv[:, :, 2] > 145)).astype(np.uint8) * 255
    inner = np.zeros_like(mask)
    inner[10:34, 3:44] = mask[10:34, 3:44]
    return inner


def normalize_badge_glyph(mask, box):
    x, y, width, height = box
    glyph = mask[y:y + height, x:x + width]
    normalized_width = max(1, round(width * 24 / height))
    resized = cv2.resize(glyph, (normalized_width, 24), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((32, 32), dtype=np.uint8)
    left = (32 - normalized_width) // 2
    canvas[4:28, left:left + normalized_width] = resized
    return canvas


def match_badge_digit(image, templates=None):
    templates = load_badge_templates() if templates is None else templates
    mask = badge_glyph_mask(image)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    glyphs = []
    for contour in contours:
        box = cv2.boundingRect(contour)
        x, y, width, height = box
        if 10 <= height <= 20 and 2 <= width <= 20 and cv2.contourArea(contour) >= 8:
            glyphs.append((cv2.contourArea(contour), box))
    if len(glyphs) != 1:
        return []
    glyph = normalize_badge_glyph(mask, glyphs[0][1])
    best = {}
    for digit, variants in templates.items():
        for template in variants:
            score = float(cv2.matchTemplate(glyph, template, cv2.TM_CCOEFF_NORMED)[0, 0])
            best[digit] = max(best.get(digit, -1.0), score)
    return sorted(((score, digit) for digit, score in best.items()), reverse=True)


def load_table_templates():
    global _TABLE_TEMPLATES
    if _TABLE_TEMPLATES is None:
        templates = {}
        for path in TABLE_TEMPLATE_DIR.glob('*.png'):
            data = np.fromfile(str(path), dtype=np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                templates[path.stem] = image
        _TABLE_TEMPLATES = templates
    return _TABLE_TEMPLATES


def glyph_holes(gray):
    mask = (gray < 150).astype(np.uint8) * 255
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return 0
    return sum(1 for item in hierarchy[0] if item[3] != -1 and cv2.contourArea(contours[item[3]]) > 4)


def match_table_glyph(crop, hand_templates, table_templates=None):
    table_templates = load_table_templates() if table_templates is None else table_templates
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    normalized = cv2.resize(gray, GLYPH_SIZE, interpolation=cv2.INTER_AREA)
    best = {}
    for rank, template in hand_templates.items():
        hand_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if template.ndim == 3 else template
        hand_gray = cv2.resize(hand_gray, GLYPH_SIZE, interpolation=cv2.INTER_AREA)
        score = float(cv2.matchTemplate(normalized, hand_gray, cv2.TM_CCOEFF_NORMED)[0][0])
        best[rank] = max(best.get(rank, -1.0), score)
    for rank, template in table_templates.items():
        resized = cv2.resize(template, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_AREA)
        score = float(cv2.matchTemplate(gray, resized, cv2.TM_CCOEFF_NORMED)[0][0])
        best[rank] = max(best.get(rank, -1.0), score)
    return sorted(((score, rank) for rank, score in best.items()), reverse=True)


def resolve_played_rank(scores, crop):
    score, rank = scores[0]
    second_score, second_rank = scores[1] if len(scores) > 1 else (-1.0, None)
    if {rank, second_rank} == {'5', '6'} and score - second_score <= 0.12:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        holes = glyph_holes(gray)
        if holes == 1:
            rank = '6'
        elif holes == 0:
            rank = '5'
        else:
            return None, None
        score = next(item_score for item_score, item_rank in scores if item_rank == rank)
    return score, rank


def read_hand(image, templates=None):
    templates = templates or load_templates()
    if not templates:
        raise ValueError('no mini-skin templates')
    detection = detect_hand(image)
    cards = []
    for index, box in enumerate(detection['glyphs']):
        if box[2] == 0 or box[3] == 0:
            cards.append({'index': index, 'rank': 'UNKNOWN', 'score': 0.0, 'selected': False})
            continue
        glyph = crop_glyph(image, box)
        scores = match_glyph(glyph, templates)
        score, rank = scores[0]
        cards.append({
            'index': index,
            'rank': rank if score >= THRESHOLD else 'UNKNOWN',
            'score': round(score, 4),
            'selected': index in detection['selected_indices'],
        })
    detection['cards'] = cards
    detection['my_hand'] = [card['rank'] for card in cards]
    detection['selected_ranks'] = [card['rank'] for card in cards if card['selected'] and card['rank'] != 'UNKNOWN']
    detection['unknown_indices'] = [card['index'] for card in cards if card['rank'] == 'UNKNOWN']
    return detection


def detect_landlord(image):
    scale_x, scale_y = scale_for(image)
    for seat, (x0, y0, x1, y1) in AVATAR_ROIS.items():
        roi = image[round(y0 * scale_y):round(y1 * scale_y), round(x0 * scale_x):round(x1 * scale_x)]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = (((hsv[:, :, 0] <= 15) | (hsv[:, :, 0] >= 170)) & (hsv[:, :, 1] > 150) & (hsv[:, :, 2] > 150)).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            if 14 * scale_x <= width <= 45 * scale_x and 30 * scale_y <= height <= 75 * scale_y and cv2.contourArea(contour) > 350 * scale_x * scale_y:
                return seat
    return None


def _split_rank(roi, component):
    x, y, width, height = component
    block = roi[y:y + height, x:x + width]
    if height <= 62:
        return block
    hsv = cv2.cvtColor(block, cv2.COLOR_BGR2HSV)
    dark = ((hsv[:, :, 2] < 170) | (hsv[:, :, 1] > 60))
    rows = dark.any(axis=1)
    segments = []
    start = None
    for index, value in enumerate(rows):
        if value and start is None:
            start = index
        elif not value and start is not None:
            segments.append((start, index))
            start = None
    if start is not None:
        segments.append((start, len(rows)))
    segments = [segment for segment in segments if segment[1] - segment[0] >= 18]
    if len(segments) >= 2:
        top, bottom = segments[0]
        return block[top:bottom, :]
    return block[:max(1, int(height * 0.58)), :]


def detect_played(image, templates=None):
    templates = templates or load_templates()
    table_templates = load_table_templates()
    scale_x, scale_y = scale_for(image)
    result = {}
    for seat, (x0, y0, x1, y1) in TABLE_ROIS.items():
        left = round(x0 * scale_x)
        top = round(y0 * scale_y)
        roi = image[top:round(y1 * scale_y), left:round(x1 * scale_x)]
        white = (np.all(roi > 175, axis=2)).astype(np.uint8) * 255
        white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        groups, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cards = []
        for group in groups:
            gx, gy, gw, gh = cv2.boundingRect(group)
            if gw < 35 * scale_x or gh < 55 * scale_y or cv2.contourArea(group) < 1500 * scale_x * scale_y:
                continue
            sub = roi[gy:gy + gh, gx:gx + gw]
            nonwhite = (~(np.all(sub > 175, axis=2))).astype(np.uint8) * 255
            glyphs, _ = cv2.findContours(nonwhite, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            group_cards = []
            for glyph in glyphs:
                x, y, width, height = cv2.boundingRect(glyph)
                if not (10 * scale_x <= width <= 60 * scale_x and 22 * scale_y <= height <= 60 * scale_y
                        and cv2.contourArea(glyph) > 80 * scale_x * scale_y):
                    continue
                if y > gh * 0.5:
                    continue
                crop = _split_rank(sub, (x, y, width, height))
                if crop.size == 0:
                    continue
                score, rank = resolve_played_rank(match_table_glyph(crop, templates, table_templates), crop)
                if rank is None or score < 0.5:
                    continue
                group_cards.append({'x': gx + x + left, 'y': gy + y + top, 'rank': rank, 'score': round(float(score), 3)})
            if group_cards:
                cards.extend(group_cards)
        cards.sort(key=lambda card: card['x'])
        count_est = len(cards)
        if len(cards) >= 2:
            pitch = float(np.median([cards[index + 1]['x'] - cards[index]['x']
                                     for index in range(len(cards) - 1)]))
            if pitch > 0:
                count_est = int(round((cards[-1]['x'] - cards[0]['x']) / pitch)) + 1
        reliable = (not cards) or count_est == len(cards)
        result[seat] = {'ranks': [card['rank'] for card in cards], 'cards': cards,
                        'count_est': count_est, 'reliable': reliable}
    return result


LABELS = {
    'samples/selftest/mini2_bid_after.png': ['LJ', '2', '2', '2', 'K', 'K', 'K', 'K', 'Q', 'Q', 'J', '9', '9', '9', '9', '7', '7'],
    'samples/selftest/mini_bot_check2.png': ['LJ', 'A', 'A', 'A', 'A', 'K', 'K', 'K', 'K', '7', '7', '7', '7'],
    'samples/selftest/mini_bot_check3.png': ['LJ', 'A', 'A', 'A', 'A', 'K', 'K', 'K', 'K'],
    'samples/selftest/mini_live_now2.png': ['BJ', 'LJ', '2', '2', 'A', 'Q', 'Q', 'Q', 'J', 'J', 'J', '10', '10', '10', '7', '7', '7', '6', '5', '5'],
    'samples/selftest/mini_live_now6.png': ['BJ', 'LJ', '2', '2', 'K', 'K', '10', '8', '8', '7', '7', '6', '6', '5', '4', '4', '3'],
}


def calibrate():
    sources = {}
    for path, ranks in LABELS.items():
        image = load_image(ROOT / path)
        added = build_templates(image, ranks, only_missing=True)
        sources[path] = added
    write_json(TEMPLATE_DIR / 'manifest.json', {'sources': sources, 'threshold': THRESHOLD})
    missing = [rank for rank in RANK_ORDER if not (TEMPLATE_DIR / f'{rank}.png').exists()]
    print(json.dumps({'added': {k: sorted(v) for k, v in sources.items()}, 'missing': missing}, ensure_ascii=True))


def evaluate():
    templates = load_templates()
    report = {}
    for path, expected in LABELS.items():
        image = load_image(ROOT / path)
        result = read_hand(image, templates)
        actual = result['my_hand']
        report[path] = {
            'count': result['count'],
            'actual': actual,
            'correct': actual == expected,
            'unknown_indices': result['unknown_indices'],
            'selected_indices': result['selected_indices'],
            'min_score': round(min(card['score'] for card in result['cards']), 4),
        }
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'calibrate':
        calibrate()
    elif len(sys.argv) > 1 and sys.argv[1] == 'evaluate':
        evaluate()
    elif len(sys.argv) > 1 and sys.argv[1] == 'read':
        print(json.dumps(read_hand(load_image(sys.argv[2])), ensure_ascii=True))
