"""Evaluate the approved 100-meme/all-20-emoji diagnostic without changing old results."""
import argparse
import json
from pathlib import Path

import numpy as np
import prepare_emoji_sweep as data
import evaluate_unseen_patterns as shared

ARMS = shared.ARMS
CONDITION = data.CONDITIONS[0]
DEFAULT_ROOT = 'results/mami_lora_qlora/patterns-100-all20-v1'
CONTROLS = ('checkpoint', 'versions', 'processor', 'generation', 'image_budget',
            'inference_microbatch', 'generation_config', 'adapter_hash',
            'adapter_merged', 'scoring_source_sha256')


def key(row):
    return row['condition'], row['original_id'], row['emoji_code']


def quality(rows):
    metrics = shared.training.metric_functions()
    result = metrics.quality([r['true_label'] for r in rows], [r['pred_label'] for r in rows])
    result['strict_format_compliance'] = sum(r['strict_format'] for r in rows)/len(rows)
    result['n_originals'] = len({r['original_id'] for r in rows})
    return result


def read_completed(folder, conditions, preparation_sha):
    done = json.loads((folder/'complete.json').read_text())
    if (done['test_preparation_sha256'] != preparation_sha or
        done['predictions_sha256'] != shared.common.digest(folder/'predictions.jsonl') or
        done['metadata_sha256'] != shared.common.digest(folder/'metadata.json') or
        done['n_originals'] != 100 or done['predictions'] != 2000):
        raise ValueError('Sweep completion counts or hashes differ')
    rows = [json.loads(line) for line in (folder/'predictions.jsonl').read_text().splitlines()]
    expected = [(CONDITION, r['original_id'], r['emoji_code']) for r in conditions[CONDITION]]
    if [key(r) for r in rows] != expected:
        raise ValueError('Missing, duplicate or reordered meme/emoji predictions')
    metrics = shared.training.metric_functions()
    for row, src in zip(rows, conditions[CONDITION]):
        if row['true_label'] != int(src['target'] == 'misogynous') or row['pattern'] != src['pattern']:
            raise ValueError('Prediction label or layout differs')
        if row['pred_label'] != metrics.parse_answer(row['raw_answer']) or row['strict_format'] != metrics.strict_answer(row['raw_answer']):
            raise ValueError('Saved output parsing differs')
    recalculated = quality(rows)
    recalculated.pop('n_originals')
    if recalculated != done['metrics'][CONDITION]:
        raise ValueError('Saved aggregate metrics differ from predictions')
    meta = json.loads((folder/'metadata.json').read_text())
    if (meta['arm'] != done['arm'] or meta['test_preparation_sha256'] != preparation_sha or
        meta['training_completion_sha256'] != done['training_completion_sha256'] or
        meta['runner_sha256'] != shared.common.digest(__file__) or
        meta['inference_source_sha256'] != shared.common.digest(shared.__file__)):
        raise ValueError('Sweep metadata or evaluator sources differ')
    return done, rows


def run(args):
    shared.run(args, data_module=data, completion_reader=read_completed, runner_path=__file__)


def f1_counts(counts):
    result = 0.0
    for label in (0, 1):
        tp = counts[..., label, label]
        denominator = counts[..., label, :].sum(axis=-1)+counts[..., :, label].sum(axis=-1)
        result = result+np.divide(2*tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator != 0)/2
    return result


def meme_counts(rows):
    ids = sorted({r['original_id'] for r in rows})
    index = {identity: i for i, identity in enumerate(ids)}
    counts = np.zeros((len(ids), 2, 3), dtype=np.int64)
    labels, patterns = {}, {}
    for row in rows:
        identity = row['original_id']
        label, pattern = row['true_label'], row['pattern']
        if identity in labels and (labels[identity] != label or patterns[identity] != pattern):
            raise ValueError('Meme labels/layouts disagree across emoji variants')
        labels[identity], patterns[identity] = label, pattern
        pred = row['pred_label']
        counts[index[identity], label, pred if pred in (0, 1) else 2] += 1
    return ids, counts, np.array([labels[i] for i in ids]), np.array([patterns[i] for i in ids])


def paired_delta(a, b, repeats=2000):
    ids, ac, labels, patterns = meme_counts(a)
    other_ids, bc, other_labels, other_patterns = meme_counts(b)
    if (ids != other_ids or not np.array_equal(labels, other_labels) or
        not np.array_equal(patterns, other_patterns) or [key(r) for r in a] != [key(r) for r in b]):
        raise ValueError('Paired bootstrap inputs are misaligned')
    rng = np.random.default_rng(42)
    groups = [np.flatnonzero((labels == label) & (patterns == pattern))
        for label in (0, 1) for pattern in sorted(set(patterns))]
    indices = np.concatenate([rng.choice(g, size=(repeats, len(g)), replace=True)
        for g in groups if len(g)], axis=1)
    deltas = f1_counts(bc[indices].sum(axis=1))-f1_counts(ac[indices].sum(axis=1))
    return dict(delta_macro_f1=float(f1_counts(bc.sum(axis=0))-f1_counts(ac.sum(axis=0))),
        paired_meme_cluster_label_pattern_stratified_95_interval=np.quantile(deltas, [.025, .975]).tolist(),
        bootstrap_replicates=repeats, n_originals=len(ids))


def summarize(args):
    conditions, _ = data.load(args.data)
    shared.verify_test_training_link(args)
    gates = shared.training_gates(args)
    manifest = json.loads((Path(args.data)/'manifest.json').read_text())
    clean_conditions, _ = data.grid_data.load(manifest['grid_data'])
    prep_sha = shared.common.digest(Path(args.data)/'complete.json')
    training_hashes = {m: g['completion_sha256'] for m, g in gates.items()}
    records, cleans, metas, arm_summaries = {}, {}, {}, {}
    completions = {}
    for arm in ARMS:
        folder = Path(args.output_root)/arm
        done, rows = read_completed(folder, conditions, prep_sha)
        previous = Path(args.clean_results)/arm
        clean_done, clean_rows = shared.original.read_completed(previous, clean_conditions,
            manifest['grid_preparation_sha256'])
        if (done['arm'] != arm or clean_done['arm'] != arm or
            done['training_completion_sha256'] != training_hashes or
            clean_done['training_completion_sha256'] != training_hashes):
            raise ValueError('Models/training differ from the completed clean evaluation')
        meta = json.loads((folder/'metadata.json').read_text())
        clean_meta = json.loads((previous/'metadata.json').read_text())
        for field in CONTROLS:
            if meta[field] != clean_meta[field]:
                raise ValueError(f'Cannot reuse clean predictions: {arm} {field}')
        mode = arm.split('-')[0]
        if meta['adapter_hash'] != (None if arm.endswith('-base') else gates[mode]['complete']['final_adapter_sha256']):
            raise ValueError('Adapter hash differs from final epoch-two checkpoint')
        source_clean = {r['original_id']: r for r in clean_rows if r['condition'] == 'clean_with_transcription'}
        clean = []
        for row in rows:
            old = source_clean[row['original_id']]
            if old['true_label'] != row['true_label']:
                raise ValueError('Clean and overlay labels differ')
            # Repeat only for equal per-meme weighting in a 20-emoji comparison;
            # bootstrap still draws whole memes, never independent variants.
            clean.append(dict(old, condition=CONDITION, pattern=row['pattern'], emoji_code=row['emoji_code']))
        records[arm], cleans[arm], metas[arm] = rows, clean, meta
        arm_summaries[arm] = dict(
            by_pattern={p: quality([r for r in rows if p == 'all' or r['pattern'] == p])
                for p in ('all',)+data.patterns.PATTERNS},
            by_emoji={c: quality([r for r in rows if r['emoji_code'] == c]) for c in manifest['emoji_codes']},
            matched_clean=quality([source_clean[i] for i in manifest['selected_ids']]),
            evaluation_wall_seconds=done['evaluation_wall_seconds'],
            peak_inference_allocated_bytes=done['peak_inference_allocated_bytes'],
            peak_inference_reserved_bytes=done['peak_inference_reserved_bytes'])
        completions[arm] = dict(sweep=shared.common.digest(folder/'complete.json'),
            clean=shared.common.digest(previous/'complete.json'))
    for arm in ARMS:
        for field in CONTROLS+('gpu', 'gpu_uuid', 'runner_sha256', 'inference_source_sha256'):
            if field in ('adapter_hash',):
                continue
            if metas[arm][field] != metas[ARMS[0]][field]:
                raise ValueError(f'Sweep arms differ: {field}')
        for row, reference, clean, clean_reference in zip(records[arm], records[ARMS[0]], cleans[arm], cleans[ARMS[0]]):
            for field in ('true_label', 'input_signature', 'pattern', 'emoji_code'):
                if row[field] != reference[field]:
                    raise ValueError(f'Overlay inputs differ across arms: {field}')
            if clean['input_signature'] != clean_reference['input_signature']:
                raise ValueError('Reused clean inputs differ across arms')
    contrasts, corruption = {}, {}
    for pattern in ('all',)+data.patterns.PATTERNS:
        selected = {a: [r for r in records[a] if pattern == 'all' or r['pattern'] == pattern] for a in ARMS}
        selected_clean = {a: [r for r in cleans[a] if pattern == 'all' or r['pattern'] == pattern] for a in ARMS}
        contrasts[pattern] = {f'{b}_minus_{a}': paired_delta(selected[a], selected[b])
            for a, b in (('fp16-base', 'fp16-lora'), ('nf4-base', 'nf4-qlora'),
                         ('fp16-base', 'nf4-base'), ('fp16-lora', 'nf4-qlora'))}
        corruption[pattern] = {a: paired_delta(selected_clean[a], selected[a]) for a in ARMS}
    report = dict(n_originals=100, total_predictions=8000, predictions_per_arm=2000,
        condition=CONDITION, nominal_alpha=.8, pattern_counts=manifest['pattern_counts'],
        test_preparation_sha256=prep_sha, completion_sha256=completions,
        arms=arm_summaries, paired_model_contrasts=contrasts, overlay_minus_matched_clean=corruption,
        qualifications=[
            'Exploratory test subset: 100 memes, 50 per class; outcomes not used for sampling.',
            'One inherited layout and fixed positions per meme; all 20 evaluation emojis per meme.',
            'Current JPEG rendering, pixel budget, generation and leading-label parser; not exact historical evaluation.',
            'Macro-F1 is computed from pooled confusion counts, not averaged emoji F1 scores.',
            'Paired seed-42 bootstrap, 2000 replicates; whole memes with all emoji variants resampled within true-class/layout strata.',
            'Pattern subsets contain only 33 or 34 memes; variants are not independent memes.',
            'Pointwise unadjusted intervals; no equivalence margin; one training seed/split.',
            'Clean predictions reused after integrity and inference-control checks; one clean prediction per selected meme/arm.',
            'No all-20 grid predictions are available; existing single-emoji grid is not used as an identity-matched sweep baseline.',
            'Test data already examined; inference benchmarking included 100 test images, possibly overlapping this sample.',
            'One quality pass; time and memory descriptive; fixed unmerged epoch-two adapters.'])
    path = Path(args.output_root)/'comparison.json'
    if path.exists():
        if json.loads(path.read_text()) != report:
            raise ValueError('Existing comparison differs; preserve it and choose a new output root')
        print(f'Reused verified comparison: {path}')
    else:
        shared.common.save_json(path, report)
        print(f'Saved comparison: {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('run', 'summarize'))
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--data', default=data.DEFAULT_OUTPUT)
    parser.add_argument('--output-root', default=DEFAULT_ROOT)
    parser.add_argument('--clean-results', default=shared.original.DEFAULT_ROOT)
    parser.add_argument('--training-data', default='data/processed/mami_lora_qlora_seed42')
    parser.add_argument('--fp16-training', default='results/mami_lora_qlora/fp16-training-v2')
    parser.add_argument('--nf4-training', default='results/mami_lora_qlora/nf4-training-v2')
    parser.add_argument('--reuse-complete', action='store_true')
    args = parser.parse_args()
    if args.command == 'run' and not args.arm:
        parser.error('--arm is required for run')
    (run if args.command == 'run' else summarize)(args)


if __name__ == '__main__':
    main()
