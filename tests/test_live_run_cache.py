"""Disk monitoring tails append-only streams and replays replaced sources."""
import json

import pytest

from frontend.live_run import LiveRun, LiveRunCache


def metric(step, value):
    return json.dumps({'v': 1, 'type': 'metrics', 'step': step, 'data': {'loss': value}}) + '\n'


@pytest.mark.parametrize('pinned', [False, True])
def test_stream_cache_only_processes_appends_and_resets_replacements(tmp_path, monkeypatch, pinned):
    agent = tmp_path / 'g/e/a'
    version = agent / 'version_0'
    version.mkdir(parents=True)
    stream = version / 'events.jsonl'
    stream.write_text(''.join(metric(i, i) for i in range(1000)))
    cache = LiveRunCache()

    def load():
        return cache.load_version(agent, 'version_0') if pinned else cache.load(tmp_path, 'g', 'e', ['a'])['a']

    first = load()
    processed = []
    original = LiveRun.add

    def add(self, records, **kwargs):
        processed.append(len(records))
        return original(self, records, **kwargs)

    monkeypatch.setattr(LiveRun, 'add', add)
    assert load() is first
    assert processed == []
    with stream.open('a') as f:
        f.write(metric(1000, 7))
    assert load() is first
    assert processed == [1]
    assert len(first.rows) == 1001
    assert first.latest('loss') == 7
    # Atomic sync replacement can be longer than the original file.
    replacement = version / 'replacement'
    replacement.write_text(''.join(metric(i, 9) for i in range(1100)))
    replacement.replace(stream)
    second = load()
    assert second is not first
    assert len(second.rows) == 1100
    assert second.latest('loss') == 9
    stream.write_text(metric(0, 3))
    assert len(load().rows) == 1
    assert load().latest('loss') == 3


def test_partial_stream_and_new_version(tmp_path):
    agent = tmp_path / 'g/e/a'
    version = agent / 'version_0'
    version.mkdir(parents=True)
    stream = version / 'events.jsonl'
    stream.write_text(metric(0, 1))
    cache = LiveRunCache()
    load = lambda: cache.load(tmp_path, 'g', 'e', ['a'])['a']
    first = load()
    record = metric(1, 2)
    with stream.open('a') as f:
        f.write(record[:-1])
    assert load() is first and len(first.rows) == 1
    with stream.open('a') as f:
        f.write('\n')
    assert load() is first and len(first.rows) == 2
    newer = agent / 'version_1'
    newer.mkdir()
    (newer / 'events.jsonl').write_text(metric(0, 5))
    assert load() is not first
    assert len(load().rows) == 1 and load().latest('loss') == 5


def test_csv_rewrite_and_transition_to_stream(tmp_path):
    version = tmp_path / 'g/e/a/version_0'
    version.mkdir(parents=True)
    csv = version / 'metrics.csv'
    csv.write_text('step,loss\n0,1\n')
    cache = LiveRunCache()
    load = lambda: cache.load(tmp_path, 'g', 'e', ['a'])['a']
    first = load()
    csv.write_text('step,loss\n0,1000\n1,2000\n')
    second = load()
    assert second is not first
    assert second.rows == [{'step': 0., 'loss': 1000.}, {'step': 1., 'loss': 2000.}]
    (version / 'events.jsonl').write_text(metric(2, 3))
    assert load() is not second
    assert load().rows == [{'step': 2, 'loss': 3}]
