#!/usr/bin/env python3
"""
Export reference data from a Hugging Face model for DSL verification.

Outputs (into ./hf_export_out):
- tokens.json: input_ids for the prompt
- embeddings.npy: token embeddings after the embedding lookup
- position_embeddings.npy: positional embeddings for the sequence length
- logits.npy: final logits from the model
- softmax.npy: softmax(logits)
- top1.json: index/value of the top-1 prediction
- attentions_layer0.npy: attention weights from the first layer (if available)
- hidden_states_last.npy: last hidden state tensor

Requires: `pip install transformers torch numpy`.
"""
import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    model_name = os.environ.get("HF_MODEL", "gpt2")
    prompt = os.environ.get("HF_PROMPT", "Hello world")
    out_dir = Path("hf_export_out")
    out_dir.mkdir(exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True, output_hidden_states=True)
    model.eval()

    with torch.no_grad():
        enc = tokenizer(prompt, return_tensors="pt")
        tokens = enc["input_ids"]
        attn_mask = enc.get("attention_mask", torch.ones_like(tokens))

        outputs = model(**enc)
        logits = outputs.logits  # [1, seq, vocab]
        softmax = torch.softmax(logits, dim=-1)

        # Grab embeddings: token + position (from model internals)
        # For GPT-2: wte (token), wpe (position)
        try:
            tok_emb = model.transformer.wte(tokens)  # [1, seq, hidden]
            pos_ids = torch.arange(tokens.size(1), device=tokens.device).unsqueeze(0)
            pos_emb = model.transformer.wpe(pos_ids)  # [1, seq, hidden]
            emb_sum = tok_emb + pos_emb
        except Exception:
            # Fallback: use hidden_states[0] as the input embeddings if available
            tok_emb = outputs.hidden_states[0] if outputs.hidden_states else None
            pos_emb = None
            emb_sum = tok_emb

        # First layer attention weights (optional)
        attn = outputs.attentions[0] if outputs.attentions else None

        # Last hidden state
        last_hidden = outputs.hidden_states[-1] if outputs.hidden_states else None

    def save_json(obj, name):
        (out_dir / name).write_text(json.dumps(obj, indent=2))

    def save_np(tensor, name):
        if tensor is None:
            return
        np.save(out_dir / name, tensor.cpu().numpy())

    # Save artifacts
    save_json({"prompt": prompt, "tokens": tokens.cpu().tolist()}, "tokens.json")
    save_np(tok_emb, "embeddings.npy")
    save_np(pos_emb, "position_embeddings.npy")
    save_np(emb_sum, "embeddings_sum.npy")
    save_np(logits, "logits.npy")
    save_np(softmax, "softmax.npy")
    if attn is not None:
        save_np(attn, "attentions_layer0.npy")
    save_np(last_hidden, "hidden_states_last.npy")

    # Top-1 prediction on last token
    last_softmax = softmax[0, -1]
    top_val, top_idx = torch.max(last_softmax, dim=-1)
    save_json({"top1_index": int(top_idx), "top1_value": float(top_val)}, "top1.json")

    print(f"Export complete for model={model_name}, prompt='{prompt}'. Files written to {out_dir}/")


if __name__ == "__main__":
    main()
