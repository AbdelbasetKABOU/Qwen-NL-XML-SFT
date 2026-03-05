# Purpose : -------------------------------------------------------
#     Sanity check dataset before training.
#     - It verifies for each record in: out/sft_min/train.jsonl
#          01- There are exactly 3 messages system, user, assistant
#          02- The last message is: role == "assistant"
#          03- The assistant content starts with: <?xml
# -------------------------------------------------------

import json
from pathlib import Path

p = Path("out/sft_min/train.jsonl")
bad = 0
n = 0
with p.open("r", encoding="utf-8") as f:
    for line in f:
        n += 1
        r = json.loads(line)
        msgs = r.get("messages", [])
        if len(msgs) != 3:
            bad += 1; continue
        if msgs[2]["role"] != "assistant":
            bad += 1; continue
        a = msgs[2].get("content","")
        if not a.lstrip().startswith("<?xml"):
            bad += 1
print("records:", n, "bad:", bad)
