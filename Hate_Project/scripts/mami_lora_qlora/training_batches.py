"""Shared full-logit batching and bounded CPU prefetch for matched training."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor


def paired_indices(order):
    if len(order) % 2:
        raise ValueError('Training order must contain complete pairs')
    return [order[i:i+2] for i in range(0, len(order), 2)]


def prefetched(items, prepare, depth=2):
    """One CPU producer, at most depth prepared/in-flight items plus consumer."""
    if depth < 1:
        raise ValueError('Prefetch depth must be positive')
    iterator = iter(items)
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='mami-input')
    pending = deque()
    try:
        for _ in range(depth):
            try:
                item = next(iterator)
            except StopIteration:
                break
            pending.append(executor.submit(prepare, item))
        while pending:
            future = pending.popleft()
            value = future.result()
            del future
            try:
                item = next(iterator)
            except StopIteration:
                pass
            else:
                pending.append(executor.submit(prepare, item))
            yield value
            del value
    finally:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


def combine_batches(entries, pad_token_id):
    """Right-pad already verified single-image records without reprocessing images."""
    import torch
    from transformers.feature_extraction_utils import BatchFeature
    allowed = {'input_ids', 'attention_mask', 'labels', 'pixel_values', 'image_grid_thw'}
    if any(set(batch) != allowed for batch, _ in entries):
        raise ValueError('Unexpected processor fields; review batching before proceeding')
    max_len = max(batch['input_ids'].shape[1] for batch, _ in entries)
    result = {}
    for key, fill in (('input_ids', pad_token_id), ('attention_mask', 0), ('labels', -100)):
        padded = []
        for batch, _ in entries:
            tensor = batch[key]
            padded.append(torch.nn.functional.pad(tensor, (0, max_len-tensor.shape[1]), value=fill))
        result[key] = torch.cat(padded, dim=0)
    for key in ('pixel_values', 'image_grid_thw'):
        result[key] = torch.cat([batch[key] for batch, _ in entries], dim=0)
    return BatchFeature(result)



def full_record_losses(model, batch):
    """Full causal logits, token mean within records; caller averages records."""
    import torch
    inputs = {key: value for key, value in batch.items() if key != 'labels'}
    logits = model(**inputs).logits[:, :-1].float()
    targets = batch['labels'][:, 1:]
    losses = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1), ignore_index=-100, reduction='none').view_as(targets)
    counts = (targets != -100).sum(dim=1)
    if (counts == 0).any().item():
        raise ValueError('Every record requires supervised answer tokens')
    return losses.sum(dim=1) / counts
