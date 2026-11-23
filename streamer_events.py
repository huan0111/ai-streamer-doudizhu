import argparse
import json
import os
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EVENTS_PATH = ROOT / 'outputs' / 'decision_events.jsonl'
LATEST_PATH = ROOT / 'outputs' / 'streamer_latest.json'


class StreamerEventStream:
    def __init__(self, path=EVENTS_PATH, latest_path=LATEST_PATH, session_id=None):
        self.path = Path(path)
        self.latest_path = Path(latest_path)
        self.session_id = session_id or uuid.uuid4().hex
        self.sequence = 0
        self.lock = threading.Lock()

    def publish(self, event):
        with self.lock:
            self.sequence += 1
            event_id = uuid.uuid4().hex
            utterance = str(event.get('utterance') or '').strip()
            execution = event.get('execution', 'decided')
            default_speak = bool(utterance) and execution not in {'pending', 'failed', 'rejected'}
            speak = bool(event['speak']) and bool(utterance) if 'speak' in event else default_speak
            record = {
                'schema_version': 2,
                'event_id': event_id,
                'session_id': self.session_id,
                'sequence': self.sequence,
                'timestamp': datetime.now().astimezone().isoformat(),
                'type': 'doudizhu_action',
                'event_kind': 'decision',
                'action_id': event_id,
                'game': '欢乐斗地主',
                **event,
                'utterance': utterance,
                'execution': execution,
                'speak': speak,
            }
            encoded = json.dumps(record, ensure_ascii=False)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.latest_path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open('a', encoding='utf-8') as handle:
                handle.write(encoded + '\n')
            if speak:
                temporary = self.latest_path.with_suffix(self.latest_path.suffix + '.tmp')
                temporary.write_text(encoded + '\n', encoding='utf-8')
                os.replace(temporary, self.latest_path)
            return record


def follow_events(path=EVENTS_PATH, from_start=False, speech_only=True, poll_interval=0.2, once=False):
    path = Path(path)
    seen = set()
    while not path.exists():
        if once:
            return
        time.sleep(poll_interval)
    with path.open('rb') as handle:
        if not from_start:
            handle.seek(0, os.SEEK_END)
        while True:
            offset = handle.tell()
            raw = handle.readline()
            if not raw:
                if once:
                    return
                time.sleep(poll_interval)
                continue
            if not raw.endswith(b'\n'):
                handle.seek(offset)
                if once:
                    return
                time.sleep(poll_interval)
                continue
            try:
                event = json.loads(raw.decode('utf-8'))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            event_id = event.get('event_id')
            if event_id and event_id in seen:
                continue
            if event_id:
                seen.add(event_id)
            if speech_only and event.get('speak') is not True:
                continue
            yield event


def main():
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description='输出斗地主机器人产生的主播口播事件（JSON Lines）')
    parser.add_argument('--path', type=Path, default=EVENTS_PATH)
    parser.add_argument('--from-start', action='store_true', help='从文件开头读取；默认只监听新事件，避免重播历史口播')
    parser.add_argument('--include-non-speech', action='store_true', help='同时输出诊断和失败事件')
    parser.add_argument('--once', action='store_true', help='读完当前可用事件后退出')
    args = parser.parse_args()
    for event in follow_events(
        args.path,
        from_start=args.from_start,
        speech_only=not args.include_non_speech,
        once=args.once,
    ):
        print(json.dumps(event, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
