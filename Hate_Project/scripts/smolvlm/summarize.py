"""Compare the four completed SmolVLM test arms after checking paired inputs."""
import argparse
import json
from pathlib import Path

import common as c

ARMS = ('fp16-base', 'nf4-base', 'fp16-lora', 'nf4-qlora')


def summarize(root):
    scoring = c.metrics()
    results = {arm: c.read_json(root/arm/'complete.json') for arm in ARMS}
    reference = results[ARMS[0]]
    for arm, done in results.items():
        if done['arm'] != arm:
            raise ValueError('Arm directory/completion mismatch')
        for key in ('model', 'revision', 'versions', 'test_preparation_sha256', 'training_completion_sha256'):
            if done[key] != reference[key]:
                raise ValueError(f'Evaluation controls differ: {key}')
    tables = {}
    for condition in c.CONDITIONS:
        paired = None
        values = {}
        for arm in ARMS:
            rows = [json.loads(line) for line in (root/arm/f'{condition}.jsonl').read_text().splitlines()]
            signatures = [(r['original_id'], r['true_label'], r['input_signature']) for r in rows]
            if len(rows) != 1000 or len({r['original_id'] for r in rows}) != 1000:
                raise ValueError('Expected 1000 unique test originals in every condition/arm')
            if paired is not None and signatures != paired:
                raise ValueError('Test examples, labels, ordering or processor inputs differ across arms')
            paired = signatures
            if any(scoring.parse_answer(r['raw_answer']) != r['pred_label'] for r in rows):
                raise ValueError('Recorded prediction differs from the answer parser')
            recomputed = scoring.quality([r['true_label'] for r in rows], [r['pred_label'] for r in rows])
            saved = results[arm]['summary'][condition]
            if any(saved[key] != value for key, value in recomputed.items()):
                raise ValueError('Saved metrics differ from the prediction records')
            values[arm] = saved
        tables[condition] = dict(arms=values, macro_f1_deltas={
            'nf4_minus_fp16_base': values['nf4-base']['macro_f1']-values['fp16-base']['macro_f1'],
            'qlora_minus_lora': values['nf4-qlora']['macro_f1']-values['fp16-lora']['macro_f1'],
            'lora_minus_fp16_base': values['fp16-lora']['macro_f1']-values['fp16-base']['macro_f1'],
            'qlora_minus_nf4_base': values['nf4-qlora']['macro_f1']-values['nf4-base']['macro_f1']})
    return dict(model=reference['model'], revision=reference['revision'], conditions=tables,
        selected_steps={a: results[a]['best_step'] for a in ARMS},
        evaluation_timing_scope='Full per-condition loop including preprocessing, transfer and logging; excludes model loading',
        qualification='Different early-stopping steps are allowed. These are descriptive paired results, without significance claims.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='results/smolvlm/test')
    parser.add_argument('--output', default='results/smolvlm/test/summary.json')
    args = parser.parse_args()
    result = summarize(Path(args.root))
    c.save_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
