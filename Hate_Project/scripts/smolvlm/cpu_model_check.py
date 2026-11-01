"""Supplemental actual-checkpoint CPU inference; not a CUDA/NF4 training gate."""
import argparse
import json
import time

import common as c


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    import torch
    from PIL import Image
    from transformers import Idefics3ForConditionalGeneration
    # Leave the occupied V100 alone. Limit CPU use on the shared server.
    torch.set_num_threads(2)
    processor = c.processor_load()
    model = Idefics3ForConditionalGeneration.from_pretrained(c.MODEL, revision=c.REVISION,
        local_files_only=True, trust_remote_code=False, use_safetensors=True,
        dtype=torch.float32, attn_implementation='sdpa').to('cpu').eval()
    messages = [dict(role='user', content=[dict(type='image'),
                dict(type='text', text='Describe this image briefly.')])]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    batch = processor(text=[text], images=[Image.new('RGB', (96, 64), (80, 120, 160))],
                      return_tensors='pt')
    n = batch['input_ids'].shape[1]
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**batch, max_new_tokens=5, do_sample=False, use_cache=True)
    result = dict(model=c.MODEL, revision=c.REVISION, versions=c.versions(), device='cpu',
        dtype='float32', parameter_count=sum(p.numel() for p in model.parameters()),
        input_shapes={k: list(v.shape) for k, v in batch.items()},
        raw_answer=processor.decode(output[0, n:], skip_special_tokens=True),
        generated_tokens=output.shape[1]-n, inference_seconds=time.perf_counter()-started,
        qualification='Synthetic image, real pretrained weights; not a quality benchmark or GPU/NF4 feasibility gate')
    if result['generated_tokens'] <= 0:
        raise RuntimeError('No generated tokens')
    c.save_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
