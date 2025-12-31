from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

MODEL_PATH = "/Users/huhao/models/Qwen2.5-0.5B-Instruct"
LORA_PATH = "./xiyou-lora-m4"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# ===== 纯 base（没有 LoRA）=====
base = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float16,
    trust_remote_code=True,
)
base.to("mps")
base.eval()

# ===== base + LoRA =====
base_for_lora = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float16,
    trust_remote_code=True,
)
base_for_lora.to("mps")

model = PeftModel.from_pretrained(base_for_lora, LORA_PATH)
model.eval()


def get_logits(m, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("mps")
    with torch.no_grad():
        out = m(**inputs)
    return out.logits[0, -1]


prompt = """你是一个有帮助的中文助手。

### 问题：
孙悟空为何被如来佛祖镇压？

### 回答：
"""

logits_base = get_logits(base, prompt)
logits_lora = get_logits(model, prompt)

diff = torch.mean(torch.abs(logits_base - logits_lora)).item()
print("平均 logit 差异:", diff)

gen_kwargs = dict(
    max_new_tokens=256,
    do_sample=False,
)

def ask(model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, **gen_kwargs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print(ask(base, "孙悟空为何被如来佛祖镇压？"))
print("=" * 50)
print(ask(model, "孙悟空为何被如来佛祖镇压？"))
