from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

model_name = "/Users/huhao/models/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

# 1干净地加载 base model
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=None,
    trust_remote_code=True
).eval()

# 再加载 LoRA adapter
dapt_model = PeftModel.from_pretrained(
    base_model,
    "dapt_ckpt/final"
).eval()

base_model = base_model.to("mps") 
dapt_model = dapt_model.to("mps") 

dapt_model.print_trainable_parameters()

gen_kwargs = dict(
    max_new_tokens=256,
    do_sample=False,
)

def ask(model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, **gen_kwargs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

ask(base_model, "孙悟空第一次大闹天宫发生了什么？")
ask(base_model, "孙悟空第一次大闹天宫发生了什么？")

ask(dapt_model, "孙悟空第一次大闹天宫发生了什么？")
ask(dapt_model, "孙悟空第一次大闹天宫发生了什么？")
dapt_model.disable_adapter()
ask(dapt_model, "孙悟空第一次大闹天宫发生了什么？")
