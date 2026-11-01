"""Freeze one approved unseen-layout overlay per test meme; original grid untouched."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from PIL import Image, __version__ as pillow_version
import prepare_test_conditions as source_data
import prepare_comparison_data as prep

CONDITIONS = ('overlay_image_only', 'overlay_with_transcription')
PATTERNS = ('dense_random', 'center', 'sparse_random')
DEFAULT_OUTPUT = 'data/processed/mami_unseen_patterns_seed42'


def assign_patterns(rows):
    result = {}
    for label in (0, 1):
        names = sorted(r['original_id'] for r in rows if int(r['target'] == 'misogynous') == label)
        random.Random(42+label).shuffle(names)
        # For 500/class: dense 167/167, center 166/167, sparse 167/166.
        cycle = ('dense_random', 'sparse_random', 'center') if label == 0 else PATTERNS
        for i, name in enumerate(names):
            result[name] = cycle[i % 3]
    return result


def placements(size, pattern, identity):
    w, h = size
    if pattern not in PATTERNS:
        raise ValueError(pattern)
    side = max(1, int(min(w, h) * (.75 if pattern == 'center' else .25)))
    if pattern == 'center':
        return [(int((w-side)/2), int((h-side)/2), side)]
    seed = int.from_bytes(hashlib.sha256(f'42:{identity}:{pattern}'.encode()).digest(), 'big')
    rng = random.Random(seed)
    return [(rng.randint(0, w-side), rng.randint(0, h-side), side)
            for _ in range(16 if pattern == 'dense_random' else 4)]


def render(base, asset, positions, alpha=.8):
    # Same resize, alpha scaling, masked paste and composite as original apply_4x4_grid.
    base = base.convert('RGBA')
    overlay = Image.new('RGBA', base.size, (0, 0, 0, 0))
    for x, y, side in positions:
        emoji = asset.resize((side, side), resample=Image.Resampling.LANCZOS)
        r, g, b, a = emoji.split()
        a = a.point(lambda p: int(p * alpha))
        emoji = Image.merge('RGBA', (r, g, b, a))
        overlay.paste(emoji, (x, y), emoji)
    return Image.alpha_composite(base, overlay).convert('RGB')


def create(args):
    source = Path(args.source_data).resolve()
    conditions, _ = source_data.load(source)
    out = Path(args.output).resolve()
    if out.exists():
        raise FileExistsError(f'Preserve existing patterns: {out}')
    manifest = json.loads((source/'manifest.json').read_text())
    assets = {}
    for meta in manifest['assets']:
        path = Path(meta['path'])
        if prep.digest(path) != meta['sha256']:
            raise ValueError(f'Evaluation emoji bytes changed: {path}')
        with Image.open(path) as asset:
            assets[meta['code']] = asset.convert('RGBA')
    rows = conditions['clean_image_only']
    assignment = assign_patterns(rows)
    source_overlay = {r['original_id']: r for r in conditions['overlay_image_only']}
    out.mkdir(parents=True)
    (out/'images_overlay').mkdir()
    geometry, converted = {}, {c: [] for c in CONDITIONS}
    for row in rows:
        identity = row['original_id']
        code = source_overlay[identity]['emoji_code']
        destination = out/'images_overlay'/f'{Path(identity).stem}.jpg'
        with Image.open(row['image_path']) as image:
            positions = placements(image.size, assignment[identity], identity)
            render(image, assets[code], positions).save(destination, format='JPEG')
        geometry[identity] = dict(pattern=assignment[identity], emoji_code=code,
            placements=positions, nominal_alpha=.8)
    for condition in CONDITIONS:
        for row in conditions[condition]:
            identity = row['original_id']
            path = out/'images_overlay'/f'{Path(identity).stem}.jpg'
            converted[condition].append(dict(row, image_path=str(path), image_sha256=prep.digest(path),
                pattern=assignment[identity]))
        with (out/f'{condition}.jsonl').open('x') as stream:
            for row in converted[condition]:
                stream.write(json.dumps(row, ensure_ascii=False)+'\n')
    prep.save_json(out/'manifest.json', dict(source_data=str(source),
        source_preparation_sha256=prep.digest(source/'complete.json'),
        training_manifest_sha256=manifest['training_manifest_sha256'], geometries=geometry,
        counts=dict(Counter(assignment.values())),
        class_pattern_counts={str(label): dict(Counter(assignment[r['original_id']] for r in rows
            if int(r['target'] == 'misogynous') == label)) for label in (0, 1)},
        original_augmentation_sha256=prep.digest(prep.original.__file__),
        creator_sha256=prep.digest(__file__), Pillow=pillow_version,
        qualification='New approved patterns, not recovered historical implementations; fixed one pattern per meme; overlap allowed; no rotation.'))
    done = dict(schema='mami-unseen-patterns-v1', n=1000,
        manifest_sha256=prep.digest(out/'manifest.json'),
        jsonl_sha256={c: prep.digest(out/f'{c}.jsonl') for c in CONDITIONS})
    prep.save_json(out/'complete.json', done)
    print(json.dumps(dict(done, pattern_counts=dict(Counter(assignment.values()))), indent=2))


def load(root):
    root = Path(root).resolve()
    done = json.loads((root/'complete.json').read_text())
    if done['schema'] != 'mami-unseen-patterns-v1' or done['n'] != 1000 or prep.digest(root/'manifest.json') != done['manifest_sha256']:
        raise ValueError('Unseen-pattern preparation changed or incomplete')
    manifest = json.loads((root/'manifest.json').read_text())
    source = Path(manifest['source_data'])
    if prep.digest(source/'complete.json') != manifest['source_preparation_sha256']:
        raise ValueError('Source grid/clean preparation changed')
    original, _ = source_data.load(source)
    if set(done['jsonl_sha256']) != set(CONDITIONS):
        raise ValueError('Unexpected unseen-pattern conditions')
    converted = {}
    for condition in CONDITIONS:
        path = root/f'{condition}.jsonl'
        if prep.digest(path) != done['jsonl_sha256'][condition]:
            raise ValueError('Unseen-pattern records changed')
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows) != 1000:
            raise ValueError('Expected 1000 records per unseen-pattern condition')
        for row, old in zip(rows, original[condition]):
            for key in ('original_id', 'target', 'split', 'system_prompt', 'user_prompt', 'emoji_code', 'is_augmented'):
                if row[key] != old[key]:
                    raise ValueError(f'Pattern data differs from source: {key}')
            if row['pattern'] != manifest['geometries'][row['original_id']]['pattern'] or prep.digest(row['image_path']) != row['image_sha256']:
                raise ValueError('Unseen layout or overlay bytes changed')
        converted[condition] = rows
    return converted, done


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-data', default=source_data.DEFAULT_OUTPUT)
    p.add_argument('--output', default=DEFAULT_OUTPUT)
    create(p.parse_args())


if __name__ == '__main__':
    main()
