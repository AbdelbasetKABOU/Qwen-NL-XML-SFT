import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_DIR = r"D:\workspace\twin-gpt\model-training\outputs\qwen2p5_3b_sft_min_4k_lora\adapter_final"
DEVICE = "cuda"

RE_GEN = re.compile(r"(<\?xml[^>]*\?>\s*)?(<Gen\b.*?</Gen>)", re.DOTALL)

def extract_gen(text: str) -> str:
    m = RE_GEN.search(text)
    if not m:
        return text.strip()
    return ((m.group(1) or "") + m.group(2)).strip()

def build_inputs(tok, messages):
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt")
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    return enc, enc["input_ids"].shape[1]

@torch.no_grad()
def generate_only(model, tok, enc, prompt_len, max_new_tokens=10000):
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )
    gen_ids = out[0][prompt_len:]
    return tok.decode(gen_ids, skip_special_tokens=True)

def main():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    # Two independent base models
    base_plain = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, trust_remote_code=True
    ).to(DEVICE).eval()

    base_for_adapter = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, trust_remote_code=True
    ).to(DEVICE).eval()

    adapted = PeftModel.from_pretrained(base_for_adapter, ADAPTER_DIR).to(DEVICE).eval()

    prompt = (
        "Simulate an AMI neighborhood with 16 smart meters sending periodic 128 byte reports over 802.11 network "
        "to a collector, which forwards traffic to a headend over a 2 Mbps backhaul with 5 ms delay. "
        "Each meter transmits every 30 seconds from time 1 to 300. Wi-Fi is 100 Mbps with 10 ms delay. "
        "Note a reliability target of 98% and a latency target of 20 seconds."
        "Include a small LAN at the headend."
    )

    messages = [
        {"role": "system", "content": "Translate an AMI scenario request into a valid XML following TopologySchema.xsd. Output XML only."},
        {"role": "user", "content": prompt},
    ]

    enc, prompt_len = build_inputs(tok, messages)

    base_text = generate_only(base_plain, tok, enc, prompt_len)
    adapt_text = generate_only(adapted, tok, enc, prompt_len)

    print("\n=== BASE (plain) ===")
    print(extract_gen(base_text))

    print("\n=== BASE+ADAPTER ===")
    print(extract_gen(adapt_text))

if __name__ == "__main__":
    main()
