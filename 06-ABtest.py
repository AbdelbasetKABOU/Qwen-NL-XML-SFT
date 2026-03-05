# ----------------------------------------------------------------------
# STEP (06) A/B QUALITATIVE TEST (Base vs LoRA Adapter)
# ----------------------------------------------------------------------
# Purpose: Compares the base Qwen model against the fine-tuned
#          LoRA adapter on a fixed AMI scenario prompt.
#
#          It is intended to be run AFTER training to:
#            - Visually compare generation quality
#            - Verify structural XML improvements
#            - Confirm adapter modifies behavior as expected
#
#          What It Does:
#            - Builds chat-formatted prompt using Qwen template
#            - Generates deterministic output (no sampling)
#            - Blocks markdown code fences (```xml) during generation
#            - Prints outputs for:
#                  1) Base model
#                  2) Base + LoRA adapter
#
#         Pipeline Position:
#           1) Dataset build
#           2) Dataset validation
#           3) LoRA training
#           4) Inference sanity check
#           5) This A/B comparison script
# ----------------------------------------------------------------------



import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model = "Qwen/Qwen2.5-3B-Instruct"
adapter_dir = (Path("outputs") / "qwen2p5_sft_min_8k_lora" / "adapter_final").resolve()

tok = AutoTokenizer.from_pretrained(base_model, use_fast=True, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

prompt_msgs = [
    {"role": "system", "content": "Translate an AMI scenario request into a valid XML. Output XML only."},
    {"role": "user", "content": "Simulate an AMI neighborhood with 2 smart meters sending 128-byte reports every 30 seconds over Wi-Fi to a collector, forwarded to a headend over a 2 Mbps backhaul with 5 ms delay. Run from time 1 to 60."},
]

# IMPORTANT: build attention_mask properly
text = tok.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
enc = tok(text, return_tensors="pt")
enc = {k: v.cuda() for k, v in enc.items()}

bad = ["```", "```xml", "```XML"]
bad_words_ids = [tok.encode(s, add_special_tokens=False) for s in bad]

def gen(m):
    with torch.no_grad():
        out = m.generate(
            **enc,
            max_new_tokens=1200,
            do_sample=False,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id,                                    
            bad_words_ids=bad_words_ids,
        )
    return tok.decode(out[0], skip_special_tokens=True)

# BASE
base = AutoModelForCausalLM.from_pretrained(
    base_model, device_map="cuda:0", torch_dtype=torch.float16, trust_remote_code=True
).eval()
print("\n=== BASE ===\n", gen(base))

# ADAPTER
adapted = PeftModel.from_pretrained(base, str(adapter_dir)).eval()
print("\n=== BASE+ADAPTER ===\n", gen(adapted))
