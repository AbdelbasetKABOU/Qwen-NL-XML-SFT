# Qwen NL to XML SFT

  
A lightweight pipeline for fine-tuning LLMs to translate **AMI** (Advanced Metering Infrastructure) scenario descriptions in natural luanguage **(NL)** into structured **XML**.  The project uses **Supervised Fine-Tuning (SFT)** with **LoRA adapters** on Qwen models. 

Experiments are conducted using **4K** context window (`max_len = 4096`). An alternative experiment using an **8K context window (8192 tokens)** + **Colab** (using an **A100** GPU)  is available in ***[Qwen7B-NL-XML-SFT-8K](https://github.com/AbdelbasetKABOU/Qwen7B-NL-XML-SFT-8K)***.

### Dataset Format

The dataset consists of pairs: _(Natural Language scenario,  XML configuration)_.  It follows the **chat format used for SFT training**:

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "<Gen>...</Gen>"}
  ]
}
```
The assistant response contains the **target XML configuration**.

### Typical Workflow        
0. Check dataset structure 
    > `python 00-check-sft.py`
1. Compute REAL token lengths using  qwen tokenizer    
    >`python 01-token-count-qwen.py --jsonl out/sft_min/train.jsonl --model Qwen/Qwen2.5-7B-Instruct`
2. Build filtered/tokenized dataset   
   > `python 02-build-sft-min.py --in-dir out/sft_min --out-dir out/sft_min_4k --max-len 4096 --trust-remote-code`
3. Validate XML structure  python
    > `03-check_dataset_xml.py   --train out/sft_min_4k/train_4k.jsonl   --val out/sft_min_4k/val_4k.jsonl`
4. Train LoRA adapter  
    > `python 04-train-qwen-lora.py   --model Qwen/Qwen2.5-3B-Instruct   --train-pt out/sft_min_4k/train_4k_tokenized.pt   --val-pt out/sft_min_4k/val_4k_tokenized.pt   --out-dir outputs/qwen2p5_3b_sft_min_4k_lora   --max-steps 140   --lr 1e-4   --grad-accum 16`
5. Run inference sanity check 
    > `python 05-inference-sanity-check.py`
6. Compare Base vs Adapter
    > `python 06-ABtest.py`
7. Evaluate on validation set
    >`python 07-eval-valset_3steps.py`

### Notes
-   Training uses **LoRA adapters**.
    -  _and generation using **chat templates compatible with Qwen models**_.    
-   The evaluation scripts measure:
    -   _XML validity._        
    -   _exact match._        
    -   _structural similarity._


