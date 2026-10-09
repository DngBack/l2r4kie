"""Zero-shot probe: what does base Qwen2-VL-2B emit after candidate marker pairs?"""
import json, random, sys
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

data, device = sys.argv[1], 'cuda:0'
name = 'Qwen/Qwen2-VL-2B-Instruct'
proc = AutoProcessor.from_pretrained(name, min_pixels=56*56, max_pixels=1048576, use_fast=False)
model = Qwen2VLForConditionalGeneration.from_pretrained(name, dtype=torch.bfloat16, attn_implementation='sdpa').to(device).eval()
tok = proc.tokenizer
docs = [json.loads(l) for l in open(data)]
random.Random(0).shuffle(docs)
docs = [d for d in docs if 1 <= len(d['pages']) <= 2][:6]
pairs = {'object_ref+box': ('<|object_ref_start|>', '<|object_ref_end|>', '<|box_start|>', '<|box_end|>'),
         'object_ref+quad': ('<|object_ref_start|>', '<|object_ref_end|>', '<|quad_start|>', '<|quad_end|>')}
special = set(tok.all_special_ids) | set(range(151643, 151657))
stats = {k: {'n': 0, 'closed': 0, 'coords': 0, 'exact': 0, 'stray_special': 0} for k in pairs}
examples = {k: [] for k in pairs}
for doc in docs:
    images = [Image.open(p).convert('RGB') for p in doc['pages']]
    msgs = [{'role': 'user', 'content': [{'type': 'image'} for _ in images] + [{'type': 'text', 'text': 'Extract the requested field.'}]}]
    prefix = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    fields = [f for f in doc['fields'] if isinstance(f['value'], str) and f['value'].strip()][:4]
    for f in fields:
        for label, (ks, ke, vs, ve) in pairs.items():
            text = prefix + f"{ks}{f['id']}: {f['description']}{ke}{vs}"
            inputs = proc(text=[text], images=images, return_tensors='pt').to(device)
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=40, do_sample=False)
            new = out[0, inputs['input_ids'].shape[1]:].tolist()
            close_id = tok.convert_tokens_to_ids(ve)
            body = new[:new.index(close_id)] if close_id in new else new
            s = stats[label]; s['n'] += 1
            s['closed'] += close_id in new
            s['stray_special'] += any(t in special and t != close_id for t in body)
            dec = tok.decode(body, skip_special_tokens=False)
            s['coords'] += dec.lstrip().startswith('(') and ',' in dec
            s['exact'] += dec.strip() == f['value'].strip()
            if len(examples[label]) < 5:
                examples[label].append({'field': f['id'], 'target': f['value'][:40], 'output': tok.decode(new, skip_special_tokens=False)[:80]})
print(json.dumps({'stats': stats, 'examples': examples}, ensure_ascii=False, indent=1))
