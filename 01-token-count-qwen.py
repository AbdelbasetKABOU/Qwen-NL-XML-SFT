# ----------------------------------------------------------------------
# STEP 01 : TRUE TOKEN LENGTH COUNTER (Qwen tokenizer + chat template)
# ----------------------------------------------------------------------
# Purpose:    Compute the REAL token lengths of a chat-style SFT JSONL dataset using
#             Qwen tokenizer and apply_chat_template(). This helps choose a safe
#             max sequence length (e.g., 4096 vs 8192) before building the tokenized
#             .pt cache and running training.
#
#             Measures:  - TOTAL tokens: token count of the full chat template text
#                          (system + user + assistant), which matches what SFT training sees.
#                        - SYSTEM / USER / ASSISTANT: token counts of message contents only
#                          (useful breakdown; does not include chat-template overhead).
#            
#             Outputs:   - Distribution stats (min/p50/p90/p95/p99/max/mean)
#                        - Top-N longest examples (id + token length)
#                        - Suggested max_seq_length buckets based on the true maximum length
# ----------------------------------------------------------------------


import json
import argparse
from pathlib import Path
from statistics import mean

import numpy as np
from transformers import AutoTokenizer

def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"JSON decode error in {path} at line {line_no}: {e}")

def percentile(xs, p):
    if not xs:
        return None
    return int(np.percentile(np.array(xs), p))

def main():
    ap = argparse.ArgumentParser(description="Compute TRUE token lengths using Qwen tokenizer + chat template.")
    ap.add_argument("--jsonl", required=True, help="Path to SFT jsonl (with {'messages':[...]} per line).")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Tokenizer model name.")
    ap.add_argument("--max-records", type=int, default=0, help="Analyze only first N records (0 = all).")
    ap.add_argument("--print-top", type=int, default=5, help="Print top-N longest examples (ids + lengths).")
    args = ap.parse_args()

    path = Path(args.jsonl)
    if not path.exists():
        raise SystemExit(f"[ERROR] File not found: {path.resolve()}")

    print(f"[INFO] Loading tokenizer: {args.model!r}")
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    # Qwen Instruct typically uses <|im_end|> as eos; pad to eos is fine
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    lengths_total = []
    lengths_sys = []
    lengths_user = []
    lengths_asst = []

    longest = []  # (len_total, id)

    n = 0
    for line_no, rec in iter_jsonl(path):
        msgs = rec.get("messages")
        if not isinstance(msgs, list):
            raise RuntimeError(f"[ERROR] Missing/invalid 'messages' at line {line_no}")

        # --- total tokens: what you actually feed the model in SFT (system+user+assistant) ---
        chat_text = tok.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=False,
        )
        ids = tok(chat_text, add_special_tokens=False).input_ids
        lt = len(ids)

        # --- breakdown (approx but useful): tokenize each message content alone ---
        def msg_tokens(role):
            parts = [m.get("content", "") for m in msgs if m.get("role") == role]
            if not parts:
                return 0
            return len(tok("\n".join(parts), add_special_tokens=False).input_ids)

        ls = msg_tokens("system")
        lu = msg_tokens("user")
        la = msg_tokens("assistant")

        lengths_total.append(lt)
        lengths_sys.append(ls)
        lengths_user.append(lu)
        lengths_asst.append(la)

        rid = rec.get("id", f"line{line_no}")
        longest.append((lt, rid))

        n += 1
        if args.max_records and n >= args.max_records:
            break

    longest.sort(reverse=True, key=lambda x: x[0])

    def summarize(name, xs):
        xs_sorted = sorted(xs)
        print(f"\n== {name} token lengths ==")
        print("count:", len(xs_sorted))
        print("min:", xs_sorted[0])
        print("p50:", percentile(xs_sorted, 50))
        print("p90:", percentile(xs_sorted, 90))
        print("p95:", percentile(xs_sorted, 95))
        print("p99:", percentile(xs_sorted, 99))
        print("max:", xs_sorted[-1])
        print("mean:", int(mean(xs_sorted)))

    print(f"[INFO] File: {path.resolve()}")
    summarize("TOTAL (chat_template(system+user+assistant))", lengths_total)
    summarize("SYSTEM content only", lengths_sys)
    summarize("USER content only", lengths_user)
    summarize("ASSISTANT content only (XML)", lengths_asst)

    print(f"\n== Top {args.print_top} longest examples ==")
    for lt, rid in longest[: args.print_top]:
        print(f"- {rid}: {lt} tokens")

    # Suggested buckets
    m = max(lengths_total) if lengths_total else 0
    print("\n== Suggested max_seq_length buckets (true tokens) ==")
    for b in [4096, 8192, 16384, 24576, 32768]:
        ok = "✅" if m <= b else "❌"
        print(f"{ok} {b} (max={m})")

if __name__ == "__main__":
    main()
