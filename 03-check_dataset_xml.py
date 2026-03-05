# ----------------------------------------------------------------------
# STEP 03 - DATASET XML AUDIT 
# ----------------------------------------------------------------------
# Purpose: Validates the SFT dataset AFTER tokenization/length filtering and before training. 
#          It ensures:
#            1) Proper chat structure (system/user/assistant roles present)
#            2) Assistant output contains a valid <Gen> ... </Gen> XML block
#            3) XML is well-formed and parsable
#            4) Each <flow> inside <Flows> contains required schema fields:
#                 - type
#                 - name
#                 - source
#                 - destination
#                 - expectedDelaySeconds
#                 - expectedReliabilityPercent
#         Why :
#         The model is trained to generate structured XML.
#         Any malformed or incomplete XML in the dataset will be learned
#         and amplified during LoRA fine-tuning.
#
# Example:    python 03-check_dataset_xml.py \
#                    --train out/sft_min_4k/train_4k.jsonl \
#                    --val out/sft_min_4k/val_4k.jsonl
# -------------------------------------------------------


import json
import re
import argparse
from collections import Counter, defaultdict
from xml.etree import ElementTree as ET

# python 02.3-check_dataset_xml.py --train out/sft_min_4k/train_4k.jsonl --val out/sft_min_4k/val_4k.jsonl



RE_XML = re.compile(r"(<\?xml[^>]*\?>\s*)?(<Gen\b.*?</Gen>)", re.DOTALL)

REQUIRED_FLOW_FIELDS = [
    "type",
    "name",
    "source",
    "destination",
    "expectedDelaySeconds",
    "expectedReliabilityPercent",
]


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            yield ln, json.loads(line)


def extract_gen_xml(text: str):
    """
    Returns (xml_str, reason_if_none)
    """
    if "```" in text:
        return None, "contains_code_fence"
    m = RE_XML.search(text)
    if not m:
        # maybe it's plain <Gen ...> without xml header but truncated
        if "<Gen" in text and "</Gen>" not in text:
            return None, "missing_closing_Gen"
        return None, "no_Gen_block"
    xml_str = (m.group(1) or "") + m.group(2)
    return xml_str.strip(), None


def parse_xml(xml_str: str):
    try:
        # ElementTree can parse if single root, which is <Gen>
        root = ET.fromstring(xml_str.replace('<?xml version="1.0" encoding="UTF-8"?>', '').strip())
        return root, None
    except Exception as e:
        return None, f"xml_parse_error: {type(e).__name__}"


def check_flow_fields(root):
    """
    returns list of missing fields per flow: [(flow_name_or_idx, [missing_fields...]), ...]
    """
    missing = []
    flows = root.findall(".//Flows/flow")
    for i, flow in enumerate(flows):
        # flow name if present
        name_el = flow.find("name")
        flow_id = name_el.text if (name_el is not None and name_el.text) else f"idx={i}"
        miss = []
        for field in REQUIRED_FLOW_FIELDS:
            el = flow.find(field)
            if el is None or (el.text is None) or (el.text.strip() == ""):
                miss.append(field)
        if miss:
            missing.append((flow_id, miss))
    return missing


def check_messages(row):
    msgs = row.get("messages", [])
    if not isinstance(msgs, list) or len(msgs) < 3:
        return "bad_messages"
    roles = [m.get("role") for m in msgs]
    if "system" not in roles or "user" not in roles or "assistant" not in roles:
        return "missing_role"
    return None


def audit_file(path, max_show=10):
    counters = Counter()
    examples = defaultdict(list)

    total = 0
    for ln, row in read_jsonl(path):
        total += 1

        rid = row.get("id", f"line={ln}")
        msg_err = check_messages(row)
        if msg_err:
            counters[msg_err] += 1
            if len(examples[msg_err]) < max_show:
                examples[msg_err].append(rid)
            continue

        assistant = None
        for m in row["messages"]:
            if m.get("role") == "assistant":
                assistant = m.get("content", "")
        if not assistant:
            counters["empty_assistant"] += 1
            if len(examples["empty_assistant"]) < max_show:
                examples["empty_assistant"].append(rid)
            continue

        xml_str, why = extract_gen_xml(assistant)
        if why:
            counters[why] += 1
            if len(examples[why]) < max_show:
                examples[why].append(rid)
            continue

        root, perr = parse_xml(xml_str)
        if perr:
            counters[perr] += 1
            if len(examples[perr]) < max_show:
                examples[perr].append(rid)
            continue

        # flow schema checks
        flow_missing = check_flow_fields(root)
        if flow_missing:
            counters["missing_flow_fields"] += 1
            if len(examples["missing_flow_fields"]) < max_show:
                # store a compact reason
                examples["missing_flow_fields"].append(
                    f"{rid} -> {flow_missing[0][0]} missing {flow_missing[0][1]}"
                )
            continue

        counters["ok"] += 1

    # report
    print(f"\n=== Audit: {path} ===")
    print(f"Total rows: {total}")
    for k, v in counters.most_common():
        print(f"{k:28s} : {v}")

    print("\nExamples (up to N each):")
    for k, lst in examples.items():
        print(f"- {k}:")
        for x in lst:
            print(f"  - {x}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="path to train.jsonl")
    ap.add_argument("--val", required=True, help="path to val.jsonl")
    ap.add_argument("--max-show", type=int, default=10)
    args = ap.parse_args()

    audit_file(args.train, max_show=args.max_show)
    audit_file(args.val, max_show=args.max_show)


if __name__ == "__main__":
    main()
