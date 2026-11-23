import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
LABELS_PATH = ROOT / 'samples' / 'labels.json'
TEMPLATE_DIR = ROOT / 'templates' / 'current_skin'
THRESHOLD = 0.85
PITCH = 62
LAST_CARD_WIDTH = 156
NORMAL_CARD_HEIGHT = 219
RANK_WIDTH = 52


def read_image(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'cannot read image: {path}')
    return image


def write_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f'cannot encode image: {path}')
    data.tofile(str(path))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def detect_hand(image):
    if image.shape[:2] != (1080, 1920):
        raise ValueError(f'expected 1920x1080, got {image.shape[1]}x{image.shape[0]}')
    x0, y0 = 250, 700
    roi = image[y0:1050, x0:1750]
    mask = (np.all(roi > 180, axis=2).astype(np.uint8) * 255)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if width > 100 and height > 150 and cv2.contourArea(contour) > 20000:
            regions.append((x + x0, y + y0, width, height))
    if len(regions) != 1:
        raise ValueError(f'expected one hand region, found {len(regions)}')
    x, y, width, height = regions[0]
    count = round((width - LAST_CARD_WIDTH) / PITCH) + 1
    expected_width = (count - 1) * PITCH + LAST_CARD_WIDTH
    if count < 1 or abs(width - expected_width) > 8:
        raise ValueError(f'inconsistent hand width: {width}, estimated count: {count}')
    baseline_top = y + height - NORMAL_CARD_HEIGHT
    return {
        'bbox': [x, y, width, height],
        'count': count,
        'baseline_top': baseline_top,
    }


def template_crop(image, detection, index, rank, selected=False):
    x = detection['bbox'][0] + index * PITCH + 5
    card_top = detection['baseline_top'] - (21 if selected else 0)
    y = card_top + 5
    height = 185 if rank in {'BJ', 'LJ'} else 65
    return image[y:y + height, x:x + RANK_WIDTH].copy()


def build_templates(image, detection, ranks, output_dir, only_missing=False, selected_indices=()):
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = {}
    selected_indices = set(selected_indices)
    for index, rank in enumerate(ranks):
        path = output_dir / f'{rank}.png'
        if only_missing and path.exists():
            continue
        if rank not in sources:
            write_image(path, template_crop(image, detection, index, rank, index in selected_indices))
            sources[rank] = index
    return sources


def load_templates(directory):
    templates = {}
    for path in directory.glob('*.png'):
        templates[path.stem] = read_image(path)
    if not templates:
        raise ValueError(f'no templates in {directory}')
    return templates


def recognize_hand(image, templates):
    started = time.perf_counter()
    original_size = [image.shape[1], image.shape[0]]
    if image.shape[:2] != (1080, 1920):
        image = cv2.resize(image, (1920, 1080), interpolation=cv2.INTER_LINEAR)
    detection = detect_hand(image)
    detection['original_size'] = original_size
    detection['normalized_size'] = [1920, 1080]
    baseline = detection['baseline_top']
    search_y = baseline - 37
    gray_templates = {
        rank: template if rank in {'BJ', 'LJ'} else cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        for rank, template in templates.items()
    }
    cards = []
    for index in range(detection['count']):
        x = detection['bbox'][0] + index * PITCH + 5
        roi = image[search_y:baseline + 213, x - 8:x + RANK_WIDTH + 8]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        matches = []
        for rank, template in gray_templates.items():
            current_roi = roi if rank in {'BJ', 'LJ'} else gray_roi
            result = cv2.matchTemplate(current_roi, template, cv2.TM_CCOEFF_NORMED)
            _, score, _, location = cv2.minMaxLoc(result)
            matches.append((float(score), rank, location[1] + search_y))
        score, rank, match_y = max(matches)
        recognized = rank if score >= THRESHOLD else 'UNKNOWN'
        cards.append({
            'index': index,
            'rank': recognized,
            'score': round(score, 6),
            'match_y': match_y,
            'selected': match_y < baseline - 2,
        })
    detection.update({
        'my_hand': [card['rank'] for card in cards],
        'selected_indices': [card['index'] for card in cards if card['selected']],
        'selected_ranks': [card['rank'] for card in cards if card['selected'] and card['rank'] != 'UNKNOWN'],
        'unknown_indices': [card['index'] for card in cards if card['rank'] == 'UNKNOWN'],
        'cards': cards,
        'elapsed_ms': round((time.perf_counter() - started) * 1000, 3),
    })
    return detection


def expected_selected_indices(ranks, selected_ranks):
    used = set()
    selected = []
    for rank in selected_ranks:
        index = next(i for i, candidate in enumerate(ranks) if candidate == rank and i not in used)
        used.add(index)
        selected.append(index)
    return sorted(selected)


def evaluate():
    labels = json.loads(LABELS_PATH.read_text(encoding='utf-8'))['samples']
    calibration, validation = labels[:2]
    eval_dir = ROOT / 'outputs' / 'evaluation_templates'
    eval_dir.mkdir(parents=True, exist_ok=True)
    for path in eval_dir.glob('*.png'):
        path.unlink()
    calibration_image = read_image(LABELS_PATH.parent / calibration['image'])
    calibration_detection = detect_hand(calibration_image)
    sources = build_templates(
        calibration_image,
        calibration_detection,
        calibration['my_hand'],
        eval_dir,
    )
    templates = load_templates(eval_dir)
    validation_image = read_image(LABELS_PATH.parent / validation['image'])
    result = recognize_hand(validation_image, templates)
    expected = validation['my_hand']
    actual = result['my_hand']
    known = [i for i, rank in enumerate(expected) if rank in sources]
    unseen = [i for i, rank in enumerate(expected) if rank not in sources]
    correct_known = [i for i in known if actual[i] == expected[i]]
    incorrect_known = [i for i in known if actual[i] != expected[i]]
    unseen_rejected = [i for i in unseen if actual[i] == 'UNKNOWN']
    false_positives = [i for i in unseen if actual[i] != 'UNKNOWN']
    expected_selected = validation.get('selected_indices') or expected_selected_indices(expected, validation['selected_ranks'])
    report = {
        'method': 'sample_001_templates_to_sample_002',
        'template_source': calibration['image'],
        'validation_image': validation['image'],
        'calibration_detection': {
            'expected_count': len(calibration['my_hand']),
            'detected_count': calibration_detection['count'],
            'correct': calibration_detection['count'] == len(calibration['my_hand']),
        },
        'validation': {
            'expected_count': len(expected),
            'detected_count': result['count'],
            'known_cards': len(known),
            'correct_known': len(correct_known),
            'incorrect_known_indices': incorrect_known,
            'unseen_cards': len(unseen),
            'unseen_rejected': len(unseen_rejected),
            'unseen_false_positive_indices': false_positives,
            'whole_hand_exact': actual == expected,
            'expected_selected_indices': expected_selected,
            'predicted_selected_indices': result['selected_indices'],
            'selected_state_exact': result['selected_indices'] == expected_selected,
            'elapsed_ms': result['elapsed_ms'],
        },
        'observation': result,
    }
    write_json(ROOT / 'outputs' / 'recognition_results.json', report)
    added = build_templates(
        validation_image,
        detect_hand(validation_image),
        validation['my_hand'],
        eval_dir,
        only_missing=True,
        selected_indices=validation['selected_indices'],
    )
    write_json(eval_dir / 'manifest.json', {
        'calibration_sources': sources,
        'augmented_sources': added,
        'validation_scope': 'Only sample_001 to sample_002 is cross-image validation; augmented templates are self-checked separately.',
        'threshold': THRESHOLD,
        'missing_ranks': ['9'],
    })
    print(json.dumps(report['validation'], ensure_ascii=True))


def augment(label_index):
    labels = json.loads(LABELS_PATH.read_text(encoding='utf-8'))['samples']
    sample = labels[label_index]
    image = read_image(LABELS_PATH.parent / sample['image'])
    detection = detect_hand(image)
    added = build_templates(
        image,
        detection,
        sample['my_hand'],
        TEMPLATE_DIR,
        only_missing=True,
        selected_indices=sample['selected_indices'],
    )
    manifest_path = TEMPLATE_DIR / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
    manifest.setdefault('additional_sources', {})[sample['image']] = added
    rank_order = ['3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A', '2', 'LJ', 'BJ']
    templates = load_templates(TEMPLATE_DIR)
    manifest['missing_ranks'] = [rank for rank in rank_order if rank not in templates]
    write_json(manifest_path, manifest)
    observation = recognize_hand(image, templates)
    write_json(ROOT / 'outputs' / f'observation_{label_index + 1:03d}.json', observation)
    print(json.dumps({
        'added_ranks': sorted(added),
        'missing_ranks': manifest['missing_ranks'],
        'my_hand': observation['my_hand'],
        'unknown_indices': observation['unknown_indices'],
    }, ensure_ascii=True))


def recognize(image_path, output_path):
    observation = recognize_hand(read_image(image_path), load_templates(TEMPLATE_DIR))
    write_json(output_path, observation)
    print(json.dumps({
        'count': observation['count'],
        'my_hand': observation['my_hand'],
        'selected_indices': observation['selected_indices'],
        'unknown_indices': observation['unknown_indices'],
        'elapsed_ms': observation['elapsed_ms'],
    }, ensure_ascii=True))


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', required=True)
    subparsers.add_parser('evaluate')
    recognize_parser = subparsers.add_parser('recognize')
    recognize_parser.add_argument('image', type=Path)
    recognize_parser.add_argument('output', type=Path)
    augment_parser = subparsers.add_parser('augment')
    augment_parser.add_argument('label_index', type=int)
    args = parser.parse_args()
    if args.command == 'evaluate':
        evaluate()
    elif args.command == 'augment':
        augment(args.label_index)
    else:
        recognize(args.image, args.output)


if __name__ == '__main__':
    main()
