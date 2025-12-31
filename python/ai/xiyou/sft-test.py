from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

base = AutoModelForCausalLM.from_pretrained(
    "/Users/huhao/models/Qwen2.5-0.5B-Instruct",
    torch_dtype=torch.float16,
    device_map="mps",
    trust_remote_code=True
)

model = PeftModel.from_pretrained(base, "./xiyou-lora-m4")
tokenizer = AutoTokenizer.from_pretrained("/Users/huhao/models/Qwen2.5-0.5B-Instruct", trust_remote_code=True)

gen_kwargs = dict(
    max_new_tokens=256,
    do_sample=False,
)

def ask(model, prompt):
    prompt2 = f"""你是一个有帮助的中文助手。

    ### 问题：
    {prompt}

    ### 回答：
    """
    inputs = tokenizer(prompt2, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, **gen_kwargs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

ask(model, "孙悟空第一次大闹天宫发生了什么？")
ask(model, "孙悟空为何被如来佛祖镇压？")

