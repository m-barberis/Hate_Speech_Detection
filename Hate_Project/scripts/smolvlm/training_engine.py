"""Bounded CPU prefetch and per-phase timings for unchanged 48-record updates."""
from contextlib import closing
import time

import common as c


def validation_windows(indices, effective_batch=48, interval=75):
    if len(indices) % effective_batch:
        raise ValueError('Window must contain complete optimizer updates')
    size = effective_batch * interval
    return [indices[i:i+size] for i in range(0, len(indices), size)]


def updates(model, processor, records, indices, trainable, optimizer, scaler,
            scheduler, microbatch, step_start, optimizer_step):
    """The producer is closed when this generator ends or is explicitly closed.

    Call separately for each validation window; no producer shares a processor
    with validation. CPU preprocessing time overlaps GPU work and is not additive
    with the other timing fields.
    """
    import torch
    effective_batch = 48
    if microbatch not in (2, 4, 8) or len(indices) % effective_batch:
        raise ValueError('Expected complete 48-record groups and microbatch 2/4/8')
    batching = c.legacy('training_batches')
    accumulation = effective_batch // microbatch

    def prepare(chosen):
        started = time.perf_counter()
        entries = [c.make_batch(processor, records[i]) for i in chosen]
        batch = c.combine_batches(entries, processor.tokenizer.pad_token_id)
        return batch, [a for _, a in entries], chosen, time.perf_counter()-started

    microbatches = [indices[i:i+microbatch] for i in range(0, len(indices), microbatch)]
    with closing(batching.prefetched(microbatches, prepare, depth=2)) as inputs:
        for start in range(0, len(indices), effective_batch):
            c.gpu_free()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            losses, events = [], []
            wait_seconds, preprocessing_seconds = 0.0, 0.0
            for offset in range(0, effective_batch, microbatch):
                waiting = time.perf_counter()
                batch, audits, chosen, prep_seconds = next(inputs)
                wait_seconds += time.perf_counter()-waiting
                preprocessing_seconds += prep_seconds
                expected = indices[start+offset:start+offset+microbatch]
                if chosen != expected or [(a['original_id'], a['is_augmented']) for a in audits] != [
                        (records[i]['original_id'], records[i]['is_augmented']) for i in expected]:
                    raise RuntimeError('Prefetch changed training examples or order')
                transfer_start, transfer_end, forward_start, forward_end, backward_start, backward_end = [
                    torch.cuda.Event(enable_timing=True) for _ in range(6)]
                transfer_start.record()
                batch = batch.to('cuda')
                transfer_end.record()
                forward_start.record()
                with torch.autocast('cuda', dtype=torch.float16):
                    record_losses = batching.full_record_losses(model, batch)
                    loss = record_losses.mean()
                forward_end.record()
                if not torch.isfinite(loss).item():
                    raise RuntimeError('Non-finite training loss')
                losses.extend(record_losses.detach().float().cpu().tolist())
                backward_start.record()
                scaler.scale(loss/accumulation).backward()
                backward_end.record()
                events.append((transfer_start, transfer_end, forward_start, forward_end,
                               backward_start, backward_end))
                del batch, record_losses, loss, audits
            optimizer_started = time.perf_counter()
            norm = optimizer_step(model, trainable, optimizer, scaler)
            used_lr = optimizer.param_groups[0]['lr']
            scheduler.step()
            torch.cuda.synchronize()
            optimizer_seconds = time.perf_counter()-optimizer_started
            yield dict(step=step_start+start//effective_batch+1,
                mean_loss=sum(losses)/len(losses), gradient_norm=norm, learning_rate=used_lr,
                records_in_update=len(losses), microbatch=microbatch, accumulation=accumulation,
                wall_seconds=time.perf_counter()-started,
                cpu_wait_seconds=wait_seconds, cpu_preprocessing_seconds=preprocessing_seconds,
                transfer_gpu_seconds=sum(e[0].elapsed_time(e[1]) for e in events)/1000,
                forward_gpu_seconds=sum(e[2].elapsed_time(e[3]) for e in events)/1000,
                backward_gpu_seconds=sum(e[4].elapsed_time(e[5]) for e in events)/1000,
                optimizer_wall_seconds=optimizer_seconds,
                peak_allocated_bytes=torch.cuda.max_memory_allocated())
