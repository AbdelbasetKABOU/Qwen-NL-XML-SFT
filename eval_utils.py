import re
from xml.etree import ElementTree as ET
import json
from pathlib import Path
from transformers import StoppingCriteria, StoppingCriteriaList
from typing import Dict, Any, List, Tuple

RE_GEN = re.compile(r"(<\?xml[^>]*\?>\s*)?(<Gen\b.*?</Gen>)", re.DOTALL)

class StopOnTokenSequence(StoppingCriteria):
    def __init__(self, stop_ids):
        self.stop_ids = stop_ids

    def __call__(self, input_ids, scores, **kwargs):
        seq = input_ids[0].tolist()
        if len(seq) < len(self.stop_ids):
            return False
        return seq[-len(self.stop_ids):] == self.stop_ids    
    
def build_prompt_from_record(rec: Dict[str, Any]) -> Tuple[List[Dict[str, str]], str]:
    """
    Returns (prompt_messages, reference_xml)
    prompt_messages = [system, user]
    reference_xml = assistant content (XML target)
    """
    msgs = rec.get("messages", [])
    sys_msg = next((m for m in msgs if m.get("role") == "system"), None)
    user_msg = next((m for m in msgs if m.get("role") == "user"), None)
    asst_msg = next((m for m in msgs if m.get("role") == "assistant"), None)

    if not sys_msg or not user_msg or not asst_msg:
        raise ValueError("Record missing one of roles: system/user/assistant")

    prompt = [
        {"role": "system", "content": sys_msg.get("content", "")},
        {"role": "user", "content": user_msg.get("content", "")},
    ]
    ref_xml = extract_xml(asst_msg.get("content", ""))
    return prompt, ref_xml


def generate_xml(model, tok, prompt_messages, device: str, max_new_tokens: int, bad_words_ids=None):
    text = tok.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    
    # stopping criterion: stop when </Gen> appears
    stop_ids = tok.encode("</Gen>", add_special_tokens=False)
    stopping = StoppingCriteriaList([StopOnTokenSequence(stop_ids)])
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
        bad_words_ids=bad_words_ids,
        stopping_criteria=stopping,
    )
    prompt_len = enc["input_ids"].shape[1]
    gen_ids = out[0][prompt_len:]
    decoded = tok.decode(gen_ids, skip_special_tokens=True)
    return extract_xml(decoded)


def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


def extract_xml(text):
    m = RE_GEN.search(text)
    if not m:
        return text.strip()
    return ((m.group(1) or "") + m.group(2)).strip()


def xml_is_valid(x):
    try:
        ET.fromstring(x.replace('<?xml version="1.0" encoding="UTF-8"?>', '').strip())
        return True
    except:
        return False


def jaccard(a, b):
    sa = set(a.split())
    sb = set(b.split())
    if not sa or not sb:
        return 0
    return len(sa & sb) / len(sa | sb)