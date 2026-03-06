# ----------------------------------------------------------------------
# STEP (7) : VALIDATION SCRIPT (3-step offline evaluation)
# ----------------------------------------------------------------------
# Evaluates model on a subset of validation set in three separate stages:
#
#   1) gen-base : loads base model, runs inference on first N validation
#      samples, and saves outputs to base_outputs.jsonl
#
#   2) gen-adapter : loads base model + LoRA adapter, runs inference on the same
#      prompts used in step 1, and saves outputs to adapter_outputs.jsonl
#
#   3) eval : loads both saved output files and compares predictions against 
#      reference XML from validation set. This step is offline and fast.
#
#      Current evaluation metrics:
#        - XML valid rate
#        - Exact match rate
#        - Mean Jaccard similarity
#
#      Why : generation is the expensive part. By splitting generation and evaluation,
#            we can improve metrics and analysis later without re-running inference.
#     
#      Typical usage:
#        python 07-eval-valset_3step.py gen-base --n-samples 50 --out-dir eval_runs/val50
#        python 07-eval-valset_3step.py gen-adapter --n-samples 50 --out-dir eval_runs/val50
#        python 07-eval-valset_3step.py eval --out-dir eval_runs/val50
# ----------------------------------------------------------------------

import os
import json
import argparse
from pathlib import Path

from typing import Dict
from statistics import mean
from xml.etree import ElementTree as ET

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from peft import PeftModel

from eval_utils import extract_xml, xml_is_valid, jaccard, read_jsonl
from eval_utils import generate_xml, ensure_dir, build_prompt_from_record



def cmd_gen_base(args):
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    out_path = out_dir / "base_outputs.jsonl"

    val_path = Path(args.val_jsonl)
    print(f"[INFO] Reading val set: {val_path.resolve()}")
    print(f"[INFO] Writing BASE outputs to: {out_path.resolve()}")

    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bad = ["```", "```xml", "```XML"]
    bad_words_ids = [tok.encode(s, add_special_tokens=False) for s in bad] if args.block_code_fence else None

    print("[INFO] Loading BASE model...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation=args.attn_impl,
    ).to(args.device).eval()

    n = 0
    with out_path.open("w", encoding="utf-8") as f_out:
        for line_no, rec in read_jsonl(val_path):
            rid = rec.get("id", f"line{line_no}")
            prompt, ref_xml = build_prompt_from_record(rec)

            pred_xml = generate_xml(
                base, tok, prompt, args.device,
                max_new_tokens=args.max_new_tokens,
                bad_words_ids=bad_words_ids,
            )

            row = {
                "id": rid,
                "source_line": line_no,
                "prompt": prompt,           # keep for reproducibility
                "ref_xml": ref_xml,         # keep for offline evaluation
                "base_xml": pred_xml,
            }
            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")

            n += 1
            if n >= args.n_samples:
                break
            if n % 10 == 0:
                print(f"[INFO] BASE generated {n}/{args.n_samples}")

    print(f"[DONE] BASE generation complete: {n} samples")


def cmd_gen_adapter(args):
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    base_path = out_dir / "base_outputs.jsonl"
    out_path = out_dir / "adapter_outputs.jsonl"

    if not base_path.exists():
        raise SystemExit(f"[ERROR] Missing {base_path}. Run gen-base first.")

    print(f"[INFO] Reading prompts from: {base_path.resolve()}")
    print(f"[INFO] Writing ADAPTER outputs to: {out_path.resolve()}")

    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bad = ["```", "```xml", "```XML"]
    bad_words_ids = [tok.encode(s, add_special_tokens=False) for s in bad] if args.block_code_fence else None

    print("[INFO] Loading BASE model (for adapter)...")
    base_for_adapter = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation=args.attn_impl,
    ).to(args.device).eval()

    print(f"[INFO] Loading adapter: {Path(args.adapter_dir).resolve()}")
    adapted = PeftModel.from_pretrained(base_for_adapter, args.adapter_dir).to(args.device).eval()

    n = 0
    with base_path.open("r", encoding="utf-8") as f_in, out_path.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            row = json.loads(line)
            rid = row["id"]
            prompt = row["prompt"]
            # ref_xml carried from base file; keep identical
            ref_xml = row["ref_xml"]

            print(f"[INFO] Generating ADAPTER for {rid} ...")
            pred_xml = generate_xml(
                adapted, tok, prompt, args.device,
                max_new_tokens=args.max_new_tokens,
                bad_words_ids=bad_words_ids,
            )

            out_row = {
                "id": rid,
                "ref_xml": ref_xml,
                "adapter_xml": pred_xml,
            }
            f_out.write(json.dumps(out_row, ensure_ascii=False) + "\n")

            n += 1
            if n % 10 == 0:
                print(f"[INFO] ADAPTER generated {n}")

    print(f"[DONE] ADAPTER generation complete: {n} samples")


def cmd_eval(args):
    out_dir = Path(args.out_dir)
    base_path = out_dir / "base_outputs.jsonl"
    adapt_path = out_dir / "adapter_outputs.jsonl"

    if not base_path.exists() or not adapt_path.exists():
        raise SystemExit("[ERROR] Missing base_outputs.jsonl or adapter_outputs.jsonl. Run gen-base and gen-adapter first.")

    # Load base outputs
    base_map = {}
    with base_path.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            base_map[r["id"]] = r

    # Load adapter outputs
    adapt_map = {}
    with adapt_path.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            adapt_map[r["id"]] = r

    common_ids = [rid for rid in base_map.keys() if rid in adapt_map]
    if not common_ids:
        raise SystemExit("[ERROR] No common ids between base and adapter outputs.")

    # Metrics
    base_xml_ok = 0
    adapt_xml_ok = 0
    base_exact = 0
    adapt_exact = 0
    base_j = []
    adapt_j = []

    # Optional: write per-sample CSV-ish jsonl for inspection
    details_path = out_dir / "eval_details.jsonl"

    with details_path.open("w", encoding="utf-8") as fdet:
        for rid in common_ids:
            ref = extract_xml(base_map[rid].get("ref_xml", ""))
            b = extract_xml(base_map[rid].get("base_xml", ""))
            a = extract_xml(adapt_map[rid].get("adapter_xml", ""))

            b_ok = xml_is_valid(b)
            a_ok = xml_is_valid(a)

            base_xml_ok += int(b_ok)
            adapt_xml_ok += int(a_ok)

            base_exact += int(b.strip() == ref.strip())
            adapt_exact += int(a.strip() == ref.strip())

            bj = jaccard(b, ref)
            aj = jaccard(a, ref)
            base_j.append(bj)
            adapt_j.append(aj)

            fdet.write(json.dumps({
                "id": rid,
                "xml_valid_base": b_ok,
                "xml_valid_adapter": a_ok,
                "exact_base": (b.strip() == ref.strip()),
                "exact_adapter": (a.strip() == ref.strip()),
                "jaccard_base": bj,
                "jaccard_adapter": aj,
            }, ensure_ascii=False) + "\n")

    n = len(common_ids)
    print("\n====== VALIDATION (offline) ======")
    print(f"Samples: {n}")
    print("\nXML valid rate:")
    print(f"BASE   : {base_xml_ok/n:.3f}")
    print(f"ADAPTER: {adapt_xml_ok/n:.3f}")

    print("\nExact match rate:")
    print(f"BASE   : {base_exact/n:.3f}")
    print(f"ADAPTER: {adapt_exact/n:.3f}")

    print("\nMean Jaccard (whitespace tokens):")
    print(f"BASE   : {mean(base_j):.3f}")
    print(f"ADAPTER: {mean(adapt_j):.3f}")

    print(f"\n[INFO] Wrote per-sample details: {details_path.resolve()}")


def main():
    ap = argparse.ArgumentParser(description="3-step validation: gen-base → gen-adapter → eval (offline).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_common(p):
        p.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
        p.add_argument("--val-jsonl", default="out/sft_min/val.jsonl")
        p.add_argument("--out-dir", default="eval_runs/val50")
        p.add_argument("--n-samples", type=int, default=50)
        p.add_argument("--device", default="cuda")
        p.add_argument("--max-new-tokens", type=int, default=1200)
        p.add_argument("--attn-impl", default="sdpa", choices=["eager", "sdpa"])
        p.add_argument("--block-code-fence", action="store_true", help="Ban ``` and ```xml in generation.")

    p1 = sub.add_parser("gen-base", help="Generate BASE outputs on first N val samples and save to jsonl.")
    add_common(p1)

    p2 = sub.add_parser("gen-adapter", help="Generate ADAPTER outputs using the same prompts saved in base_outputs.jsonl.")
    add_common(p2)
    p2.add_argument("--adapter-dir", default="outputs/qwen2p5_3b_sft_min_4k_lora/adapter_final")

    p3 = sub.add_parser("eval", help="Offline evaluation using saved base+adapter outputs.")
    p3.add_argument("--out-dir", default="eval_runs/val50")

    args = ap.parse_args()

    # Basic CUDA sanity
    if args.cmd in ("gen-base", "gen-adapter"):
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise SystemExit("[ERROR] CUDA device requested but torch.cuda.is_available() is False.")

    if args.cmd == "gen-base":
        cmd_gen_base(args)
    elif args.cmd == "gen-adapter":
        cmd_gen_adapter(args)
    elif args.cmd == "eval":
        cmd_eval(args)


if __name__ == "__main__":
    main()