import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from perceive import write_json

ROOT = Path(__file__).resolve().parent


def load_api_key(config):
    name = config['api_key_env']
    value = os.environ.get(name)
    if value:
        return value
    env_path = ROOT / '.env'
    if not env_path.exists():
        raise RuntimeError(f'missing environment variable {name} and file .env')
    for line in env_path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line and not line.startswith('#') and line.startswith(f'{name}='):
            value = line.split('=', 1)[1].strip().strip('"').strip("'")
            if value and value != 'replace_with_your_key':
                return value
    raise RuntimeError(f'{name} is not configured in .env')


def post_chat(config, api_key, messages):
    payload = {
        'model': config['model'],
        'messages': messages,
        'temperature': 0.2,
        'max_tokens': 300,
        'stream': False,
        'response_format': {'type': 'json_object'},
    }
    request = urllib.request.Request(
        f"{config['base_url'].rstrip('/')}/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
        headers={
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
            'User-Agent': 'doudizhu-offline-probe/0.1',
        },
        method='POST',
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=config['timeout_seconds']) as response:
        body = response.read().decode('utf-8')
        status = response.status
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    return json.loads(body), status, elapsed_ms


def parse_content(response):
    content = response['choices'][0]['message']['content'].strip()
    if content.startswith('```'):
        lines = content.splitlines()
        if lines[0].startswith('```'):
            lines = lines[1:]
        if lines and lines[-1].strip() == '```':
            lines = lines[:-1]
        content = '\n'.join(lines).strip()
    return json.loads(content), content


def main():
    config = json.loads((ROOT / 'api_config.json').read_text(encoding='utf-8'))
    labels = json.loads((ROOT / 'samples' / 'labels.json').read_text(encoding='utf-8'))['samples']
    sample = labels[1]
    observation = json.loads((ROOT / 'outputs' / 'observation_002.json').read_text(encoding='utf-8'))
    rules = json.loads((ROOT / 'outputs' / 'rule_output_002.json').read_text(encoding='utf-8'))
    candidates = [{
        'id': candidate['id'],
        'type': candidate['move']['type'],
        'ranks': candidate['ranks'],
    } for candidate in rules['candidates']]
    state = {
        'game': '欢乐斗地主',
        'role': 'landlord',
        'my_hand': observation['my_hand'],
        'last_play': sample['last_play'],
        'remaining_counts': sample['remaining_counts'],
        'instruction': '从 candidates 中选择这一整手最合理的一个合法动作。只能使用给定 candidate id，不要创造新牌型。输出 JSON：{"candidate_id":"c000","reason":"简短中文原因"}。last_play 为 null 表示你领出，不能过牌。',
        'candidates': candidates,
    }
    messages = [
        {
            'role': 'system',
            'content': '你是斗地主决策器。本地程序已生成全部合法候选，你只负责选择。必须只输出一个 JSON 对象，不要 Markdown，不要解释候选之外的动作。',
        },
        {'role': 'user', 'content': json.dumps(state, ensure_ascii=False)},
    ]
    api_key = load_api_key(config)
    attempts = []
    started = time.perf_counter()
    final = None
    for attempt_number in range(1, 3):
        attempt = {'attempt': attempt_number, 'messages': messages.copy()}
        try:
            response, http_status, elapsed_ms = post_chat(config, api_key, messages)
            decision, content = parse_content(response)
            candidate_id = decision.get('candidate_id')
            reason = decision.get('reason')
            valid = (
                isinstance(candidate_id, str)
                and any(candidate['id'] == candidate_id for candidate in candidates)
                and isinstance(reason, str)
                and bool(reason.strip())
            )
            attempt.update({
                'http_status': http_status,
                'elapsed_ms': elapsed_ms,
                'response': response,
                'parsed_decision': decision,
                'valid': valid,
            })
            attempts.append(attempt)
            if valid:
                final = {
                    'status': 'SUCCESS',
                    'candidate_id': candidate_id,
                    'reason': reason.strip(),
                    'attempts_used': attempt_number,
                }
                break
            error = f'候选编号无效或 reason 为空：{content}'
            if attempt_number == 2:
                final = {'status': 'FAILED', 'error': error, 'attempts_used': attempt_number}
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, KeyError, ValueError, json.JSONDecodeError) as error:
            detail = str(error)
            if isinstance(error, urllib.error.HTTPError):
                detail = f'HTTP {error.code}: {error.read().decode("utf-8", errors="replace")[:2000]}'
            attempt.update({'error': detail})
            attempts.append(attempt)
            final = {'status': 'FAILED', 'error': detail, 'attempts_used': attempt_number}
        if final or attempt_number == 2:
            break
        messages.extend([
            {'role': 'assistant', 'content': content},
            {'role': 'user', 'content': f'上一次输出未通过本地校验：{error}。请只返回一个有效 candidate_id 和简短 reason 的 JSON。'},
        ])
    total_elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    result = {
        **final,
        'model': config['model'],
        'base_url': config['base_url'],
        'total_elapsed_ms': total_elapsed_ms,
        'selected_candidate': next((candidate for candidate in rules['candidates'] if candidate['id'] == final.get('candidate_id')), None),
        'real_click_executed': False,
    }
    write_json(ROOT / 'outputs' / 'model_decision_attempts.json', {
        'state': state,
        'attempts': attempts,
        'total_elapsed_ms': total_elapsed_ms,
    })
    write_json(ROOT / 'outputs' / 'model_decision_result.json', result)
    print(json.dumps(result, ensure_ascii=True))
    if result['status'] != 'SUCCESS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
