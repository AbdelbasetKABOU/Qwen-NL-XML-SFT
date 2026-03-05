# -------------------------------------
# STEP 2 — Build 4K Tokenized Dataset
# -------------------------------------
# - Reads JSONL chat dataset
# - Applies Qwen Chat Template
# - Length filtering (8K max)
# - Saves TWO outputs : - Filtered JSONL and - Tokenized PyTorch cache
#
#     01- Raw NL/XML dataset
#     02- Validation (00-check)    ↓
#     03- Tokenization + masking
#     04- Length filtering (8K)
#     05- .pt cached tensors
#
#     Example --> python 02-build-sft-min.py \
#                    --in-dir out/sft_min \
#                    --out-dir out/sft_min_4k \
#                    --max-len 4096 \
#                    --trust-remote-code
#              
#         This means : Input : out/sft_min/train.jsonl | out/sft_min/val.jsonl
#                      Output  out/sft_min_4k/
#                              ├── train_4k.jsonl
#                              ├── val_4k.jsonl
#                              ├── train_4k_tokenized.pt
#                              └── val_4k_tokenized.pt
# -------------------------------------------------------      

import os
import json
import argparse
from typing import List, Dict, Any, Tuple

import torch
from transformers import AutoTokenizer


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def common_prefix_len(a: List[int], b: List[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def build_prompt_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    # Your dataset is system + user + assistant.
    # Keep only system/user for the prompt.
    prompt = [m for m in messages if m.get("role") in ("system", "user")]
    # Preserve order (already is), but ensure system first if present
    return prompt


def tokenize_with_labels_qwen(
    tok,
    messages: List[Dict[str, str]],
) -> Tuple[List[int], List[int], List[int], int]:
    """
    Returns:
      input_ids, attention_mask, labels, prompt_len
    Labels are -100 for prompt part (system+user + assistant header), and actual tokens for assistant content.
    """
    prompt_msgs = build_prompt_messages(messages)

    # Key idea:
    # prompt_ids includes the assistant "start" header by add_generation_prompt=True
    prompt_ids = tok.apply_chat_template(
        prompt_msgs,
        tokenize=True,
        add_generation_prompt=True,
    )

    # full_ids includes assistant content because messages already include role=assistant
    full_ids = tok.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )

    # Ideally prompt_ids is a prefix of full_ids. If not, fall back to common prefix length.
    pfx = common_prefix_len(prompt_ids, full_ids)
    prompt_len = pfx

    input_ids = full_ids
    attention_mask = [1] * len(input_ids)

    labels = [-100] * prompt_len + input_ids[prompt_len:]
    return input_ids, attention_mask, labels, prompt_len


def summarize_lengths(lengths: List[int]) -> Dict[str, int]:
    if not lengths:
        return {}
    s = sorted(lengths)
    def pct(p):
        idx = int(round((p / 100.0) * (len(s) - 1)))
        return s[idx]
    return {
        "count": len(s),
        "min": s[0],
        "p50": pct(50),
        "p90": pct(90),
        "p95": pct(95),
        "p99": pct(99),
        "max": s[-1],
        "mean": int(sum(s) / len(s)),
    }


def process_split(
    tok,
    in_path: str,
    out_jsonl_path: str,
    out_pt_path: str,
    max_len: int,
) -> None:
    rows = read_jsonl(in_path)

    kept_rows = []
    tokenized_cache = []

    total_lens = []
    dropped = 0

    for r in rows:
        messages = r["messages"]
        input_ids, attention_mask, labels, prompt_len = tokenize_with_labels_qwen(tok, messages)

        L = len(input_ids)
        if L > max_len:
            dropped += 1
            continue

        total_lens.append(L)

        # Write filtered jsonl (keep original structure, add metadata)
        rr = dict(r)
        rr["token_len_total"] = L
        rr["token_len_prompt"] = prompt_len
        rr["token_len_target"] = L - prompt_len
        kept_rows.append(rr)

        # Save a deterministic tokenized cache (no datasets dependency)
        tokenized_cache.append({
            "id": r.get("id"),
            "provider": r.get("provider"),
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        })

    os.makedirs(os.path.dirname(out_jsonl_path), exist_ok=True)
    write_jsonl(out_jsonl_path, kept_rows)
    torch.save(tokenized_cache, out_pt_path)

    print(f"[INFO] Input: {in_path}")
    print(f"[INFO] Kept: {len(kept_rows)} | Dropped(>{max_len}): {dropped}")
    print(f"[INFO] Wrote: {out_jsonl_path}")
    print(f"[INFO] Wrote tokenized cache: {out_pt_path}")
    print(f"[INFO] Length stats: {summarize_lengths(total_lens)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--in-dir", default=os.path.join("out", "sft_min"))
    ap.add_argument("--out-dir", default=os.path.join("out", "sft_min_8k"))
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--trust-remote-code", action="store_true")
    args = ap.parse_args()

    train_in = os.path.join(args.in_dir, "train.jsonl")
    val_in   = os.path.join(args.in_dir, "val.jsonl")

    train_out_jsonl = os.path.join(args.out_dir, "train_8k.jsonl")
    val_out_jsonl   = os.path.join(args.out_dir, "val_8k.jsonl")

    train_out_pt = os.path.join(args.out_dir, "train_8k_tokenized.pt")
    val_out_pt   = os.path.join(args.out_dir, "val_8k_tokenized.pt")

    print(f"[INFO] Loading tokenizer: {args.model}")
    tok = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )

    # Ensure pad token exists
    if tok.pad_token is None and tok.eos_token is not None:
       tok.pad_token = tok.eos_token

    print(f"[INFO] in_dir={args.in_dir}")
    print(f"[INFO] out_dir={args.out_dir}")
    print(f"[INFO] max_len={args.max_len}")

    process_split(tok, train_in, train_out_jsonl, train_out_pt, args.max_len)
    process_split(tok, val_in, val_out_jsonl, val_out_pt, args.max_len)

    print("[DONE] sft_min_xxk build complete.")


if __name__ == "__main__":
    main()
