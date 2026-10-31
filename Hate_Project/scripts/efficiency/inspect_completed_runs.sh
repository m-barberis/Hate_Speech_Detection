#!/usr/bin/env bash
# Read existing results only. No model loading, inference, or GPU work.
set -eu
cd /home/barberis/hate_speech_vlm/Hate_Speech_Detection/Hate_Project
python - <<'PY'
import argparse
import contextlib
import importlib.util
import io
import json
import math
from pathlib import Path
import tempfile

spec = importlib.util.spec_from_file_location('comparison', 'scripts/efficiency/compare.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
root = Path('results/efficiency_v100')
manifest = json.loads((root / 'manifest.json').read_text())
expected = {r['file_name']: r for r in manifest['evaluation']}
assert len(expected) == 100
assert sum(r['label'] == 0 for r in expected.values()) == 50
assert sum(r['label'] == 1 for r in expected.values()) == 50
assert not set(expected) & {r['file_name'] for r in manifest['warmup']}
pin = {'model': 'QCRI/MemeLens-VLM', 'revision': 'a633183849df60a37251d29c3f1b0d379ae801f5'}

# Re-run only the CPU summary, in temporary storage. Preserve the original summary.
with tempfile.TemporaryDirectory() as tmp:
    dest = Path(tmp) / 'summary.json'
    args = argparse.Namespace(fp16=str(root / 'fp16-eslsrv13'),
        nf4=str(root / 'nf4-eslsrv13'), output=str(dest))
    with contextlib.redirect_stdout(io.StringIO()):
        c.summarize(args)
    summary = json.loads(dest.read_text())

def equal(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    if isinstance(a, (float, int)) and isinstance(b, (float, int)):
        return math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12)
    return a == b

saved = json.loads((root / 'summary-eslsrv13.json').read_text())
assert equal(summary, saved), 'Recomputed summary differs from saved summary'
records, meta = {}, {}
for mode in ('fp16', 'nf4'):
    m, rows, indexed, ids = c.load_results(root / (mode + '-eslsrv13'))
    assert m['checkpoint'] == pin
    assert m['manifest_sha256'] == c.digest(root / 'manifest.json')
    assert set(ids) == set(expected)
    assert m['settings']['answer_parser'] == 'leading-label-v1'
    assert m['settings']['dispatch_force_hooks'] is True
    assert m['device_map'] == "{'': 0}"
    assert bool(m['quantized_modules']) == (mode == 'nf4')
    for r in rows:
        assert r['true_label'] == expected[r['file_name']]['label']
        assert r['pred_label'] == c.parse_answer(r['raw_answer'])
        assert r.get('strict_format', c.strict_answer(r['raw_answer'])) == c.strict_answer(r['raw_answer'])
        assert 0 < r['generated_tokens'] <= 10
        assert math.isfinite(r['latency_ms']) and r['latency_ms'] > 0
    records[mode] = {name: indexed[0, name] for name in ids}
    q = m['quantized_modules']
    meta[mode] = {k: m[k] for k in ('checkpoint', 'settings', 'versions', 'torch_cuda',
        'gpu', 'gpu_uuid', 'device_map', 'passes')}
    meta[mode]['converted_module_counts'] = dict(total=len(q),
        vision=sum('visual' in n for n in q), language=sum('language_model' in n for n in q))

changed = {n for n in expected if records['fp16'][n]['pred_label'] != records['nf4'][n]['pred_label']}
badformat = {n for n in expected if not c.strict_answer(records['nf4'][n]['raw_answer'])}
counts = dict(disagreements=len(changed), nf4_format_violations=len(badformat),
    overlap=len(changed & badformat),
    fp16_correct_nf4_wrong=sum(records['fp16'][n]['pred_label'] == expected[n]['label'] and
        records['nf4'][n]['pred_label'] != expected[n]['label'] for n in changed),
    fp16_wrong_nf4_correct=sum(records['fp16'][n]['pred_label'] != expected[n]['label'] and
        records['nf4'][n]['pred_label'] == expected[n]['label'] for n in changed))
fields = ('pred_label', 'raw_answer', 'strict_format', 'generated_tokens', 'latency_ms', 'input_tokens', 'image_grid_thw')
examples = [dict(file_name=n, true_label=expected[n]['label'], changed=n in changed,
    nf4_format_violation=n in badformat, **{mode: {k: records[mode][n].get(k) for k in fields}
        for mode in ('fp16', 'nf4')}) for n in sorted(changed | badformat)]
print(json.dumps(dict(saved_summary_matches=True, metadata=meta, summary=summary,
    counts=counts, examples=examples), indent=2))
PY
