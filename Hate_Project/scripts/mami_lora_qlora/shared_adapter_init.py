"""Create and verify a common FP32 LoRA initialization without loading a model."""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

import numpy as np


SCHEMA = 'shared-lora-fp32-v1'
HASH_SCHEMA = 'sorted-name-shape-fp32-le-v1'


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical_hash(arrays):
    entries = list(arrays.items()) if isinstance(arrays, dict) else list(arrays)
    if len({name for name, _ in entries}) != len(entries):
        raise ValueError('Duplicate adapter parameter name')
    h = hashlib.sha256(HASH_SCHEMA.encode())
    for name, value in sorted(entries, key=lambda entry: entry[0]):
        array = np.ascontiguousarray(value, dtype='<f4')
        descriptor = json.dumps([name, list(array.shape), 'float32-le'], separators=(',', ':')).encode()
        h.update(len(descriptor).to_bytes(8, 'little'))
        h.update(descriptor)
        h.update(array.tobytes())
    return h.hexdigest()


def inventory_map(inventory):
    entries = inventory['trainable_parameters']
    names = [r['name'] for r in entries]
    if len(set(names)) != len(names):
        raise ValueError('Duplicate parameter name in inventory')
    result = {}
    for row in entries:
        name, shape = row['name'], row['shape']
        if row['dtype'] != 'torch.float32' or len(shape) != 2 or any(
                type(n) is not int or n <= 0 for n in shape):
            raise ValueError(f'Expected FP32 adapter matrix: {name}')
        if not any(tag in name for tag in
                   ('.lora_A.', '.lora_B.', '.lora_embedding_A.', '.lora_embedding_B.')):
            raise ValueError(f'Unsupported adapter parameter: {name}')
        result[name] = tuple(shape)
    if not result:
        raise ValueError('Empty adapter inventory')
    if sum(math.prod(shape) for shape in result.values()) != inventory['trainable_parameter_count']:
        raise ValueError('Inventory parameter total does not match shapes')
    return result


def compare_runs(fp16, nf4):
    runs = [Path(fp16), Path(nf4)]
    requests = [json.loads((p / 'request.json').read_text()) for p in runs]
    inventories = [json.loads((p / 'model_inventory.json').read_text()) for p in runs]
    completions = [json.loads((p / 'complete.json').read_text()) for p in runs]
    if [r['mode'] for r in requests] != ['fp16', 'nf4'] or [r['mode'] for r in completions] != ['fp16', 'nf4']:
        raise ValueError('Expected successful FP16 and NF4 smoke directories, in that order')
    control_keys = ('checkpoint', 'versions', 'prepared_complete_sha256', 'prepared_manifest_sha256',
                    'legacy_trainer_sha256', 'runner_sha256', 'lora', 'learning_rate',
                    'smoke_microbatches', 'gradient_checkpointing', 'max_pixels', 'min_pixels',
                    'dispatch_force_hooks', 'base_policy', 'selected')
    for key in control_keys:
        if requests[0][key] != requests[1][key]:
            raise ValueError(f'Smoke control differs: {key}')
    for key in ('processor', 'gpu', 'gpu_uuid'):
        if inventories[0][key] != inventories[1][key]:
            raise ValueError(f'Smoke inventory control differs: {key}')
    if inventories[0]['quantized_modules'] or not inventories[1]['quantized_modules']:
        raise ValueError('Unexpected base quantization inventories')
    shapes = [inventory_map(i) for i in inventories]
    if shapes[0] != shapes[1]:
        raise ValueError('FP16 and NF4 adapter names or shapes differ')
    if set(inventories[0]['adapter_target_matches']) != set(inventories[1]['adapter_target_matches']):
        raise ValueError('Adapter target modules differ')
    audited = []
    for path in runs:
        rows = [json.loads(line) for line in (path / 'microbatches.jsonl').read_text().splitlines()]
        audited.append([(r['original_id'], r['is_augmented'], r['input_signature']) for r in rows])
    if audited[0] != audited[1] or len(audited[0]) != 4:
        raise ValueError('Expected four matching smoke input signatures')
    controls = {k: requests[0][k] for k in control_keys}
    diagnostics = dict(parameter_names_and_shapes_match=True,
        parameter_order_matches=[r['name'] for r in inventories[0]['trainable_parameters']] ==
                                [r['name'] for r in inventories[1]['trainable_parameters']],
        old_initial_hashes=[r['initial_adapter_sha256'] for r in inventories],
        tensor_count=len(shapes[0]), input_signatures_match=True)
    return shapes[0], controls, diagnostics


def initialize_arrays(shapes, seed=42):
    # Equivalent default LoRA initialization distributions, sampled once in
    # CPU FP32 and independent of PEFT's base-dependent dtype/device casts.
    rng = np.random.Generator(np.random.PCG64(seed))
    arrays = {}
    for name, shape in sorted(shapes.items()):
        if '.lora_embedding_A.' in name or '.lora_B.' in name:
            value = np.zeros(shape, dtype=np.float32)
        elif '.lora_embedding_B.' in name:
            value = rng.normal(0.0, 1.0, size=shape).astype(np.float32)
        elif '.lora_A.' in name:
            bound = 1.0 / math.sqrt(shape[1])
            value = rng.uniform(-bound, bound, size=shape).astype(np.float32)
        else:
            raise ValueError(f'Unsupported adapter initialization: {name}')
        arrays[name] = value
    return arrays


def create(args):
    out = Path(args.output).resolve()
    if out.exists():
        raise FileExistsError(f'Choose a new shared initialization directory: {out}')
    shapes, controls, diagnostics = compare_runs(args.fp16, args.nf4)
    arrays = initialize_arrays(shapes)
    out.mkdir(parents=True)
    weights = out / 'adapter_init.npz'
    with weights.open('xb') as stream:
        np.savez(stream, **arrays)
    metadata = dict(schema=SCHEMA, hash_schema=HASH_SCHEMA, seed=42,
        initializer='CPU NumPy PCG64; default LoRA distributions; sorted parameter names',
        numpy_version=importlib.metadata.version('numpy'),
        initializer_source_sha256=file_hash(__file__), controls=controls,
        diagnostics=diagnostics, shapes={k: list(v) for k, v in shapes.items()},
        weights_sha256=file_hash(weights), initial_adapter_sha256=canonical_hash(arrays),
        qualification='New common initialization, not a reconstruction of either old smoke initialization')
    with (out / 'metadata.json').open('x') as stream:
        json.dump(metadata, stream, indent=2, allow_nan=False)
    # Verify the saved file with the same checks used by both GPU processes.
    load(out)
    print(json.dumps(dict(output=str(out), **diagnostics,
                         shared_initial_adapter_sha256=metadata['initial_adapter_sha256']), indent=2))


def load(root):
    root = Path(root)
    meta = json.loads((root / 'metadata.json').read_text())
    if meta['schema'] != SCHEMA or meta['hash_schema'] != HASH_SCHEMA:
        raise ValueError('Unknown shared initialization schema')
    weights = root / 'adapter_init.npz'
    if file_hash(weights) != meta['weights_sha256']:
        raise ValueError('Shared initialization file hash mismatch')
    with np.load(weights, allow_pickle=False) as archive:
        if len(archive.files) != len(set(archive.files)):
            raise ValueError('Duplicate shared initialization array')
        arrays = {name: archive[name] for name in archive.files}
    validate_arrays(arrays, {k: tuple(v) for k, v in meta['shapes'].items()})
    if canonical_hash(arrays) != meta['initial_adapter_sha256']:
        raise ValueError('Shared initialization tensor hash mismatch')
    return arrays, meta


def validate_arrays(arrays, shapes):
    if set(arrays) != set(shapes):
        raise ValueError('Shared initialization parameter names differ')
    for name, shape in shapes.items():
        array = arrays[name]
        if tuple(array.shape) != tuple(shape) or array.dtype != np.float32:
            raise ValueError(f'Shared initialization shape/dtype differs: {name}')
        if not np.isfinite(array).all():
            raise ValueError(f'Non-finite shared initialization: {name}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fp16', required=True)
    parser.add_argument('--nf4', required=True)
    parser.add_argument('--output', required=True)
    create(parser.parse_args())


if __name__ == '__main__':
    main()
