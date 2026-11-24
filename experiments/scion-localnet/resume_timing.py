"""Keep timing-rejected measurements and repeat each affected case at most once."""
import argparse
import json
from pathlib import Path
import time

import batch
import lab
import measured
import peer_exchange


def record_repeat(out, kind):
    if kind == 'peer':
        plan_path, runs = out / 'peer-plan.json', out / 'peers'
    else:
        plan_path, runs = out / 'plan.json', out / 'runs'
    plan = json.loads(plan_path.read_text())
    rejected = []
    for task in plan['tasks']:
        path = runs / task['id'] / 'summary.json'
        if path.is_file():
            summary = json.loads(path.read_text())
            if not summary['accepted']:
                rejected.append((task, summary))
    if len(rejected) != 1:
        return None
    task, summary = rejected[0]
    if task['id'].endswith('-attempt2') or summary['max_start_skew_ms'] <= 5:
        return None

    entry = dict(kind=kind, task=dict(task), summary=summary)
    if kind == 'peer':
        entry['reason'] = 'Predeclared 5 ms start gate exceeded; one identical repeat regardless of delivery.'
    else:
        entry['reason'] = 'Predeclared 5 ms sender-start tolerance exceeded; one identical repeat, regardless of delivery outcome.'
        entry['retry_code_sha256'] = lab.file_sha256(Path(__file__))
    history_path = out / 'rejected-runs.json'
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    history.append(entry)
    measured.write(history_path, history)
    archive = out / 'plan-history'
    archive.mkdir(exist_ok=True)
    measured.write(archive / f'{time.time_ns()}.json', plan)
    task['id'] += '-attempt2'
    measured.write(plan_path, plan)
    print('Retained timing rejection; repeating', task['id'], flush=True)
    return task


def run_peers(out):
    while True:
        try:
            return peer_exchange.run(out)
        except RuntimeError:
            if record_repeat(out, 'peer') is None:
                raise


def run(out):
    lab.select_profile('expanded')
    while True:
        try:
            return batch.run(out)
        except (RuntimeError, ValueError):
            task = record_repeat(out, 'payload')
            if task is None:
                raise
            if task['fault']:
                batch.recovery_check(task['fault'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args().out)
