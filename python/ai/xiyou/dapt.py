from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

model_name = "/Users/huhao/models/Qwen2.5-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
import torch

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map=None
)
print("load origin model")
model.to("mps")
model.config.use_cache = False
model.gradient_checkpointing_enable()

# LoRA 配置（DAPT 用小一点）
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM"
)


model = get_peft_model(model, lora_config)
print("get peft model")

dataset = load_dataset(
    "text",
    data_files={"train": "data/xiyou_raw.txt"}
)
print("load dataset")

def tokenize(example):
    text = example["text"].strip()

    # 无效样本：字段齐全，但为空
    if len(text) < 5:
        return {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
        }

    tokens = tokenizer(
        text,
        truncation=True,
        max_length=512
    )

    if len(tokens["input_ids"]) < 2:
        return {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
        }

    tokens["labels"] = tokens["input_ids"].copy()
    return tokens



tokenized = dataset["train"].map(
    tokenize,
    remove_columns=["text"]
).filter(lambda x: len(x["input_ids"]) > 0)



print("map dataset")

args = TrainingArguments(
    output_dir="dapt_ckpt",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=1e-5,
    optim="adamw_torch",   # 不要 fused，不要 paged
    fp16=False,
    logging_steps=1,
    save_steps=1,
    save_total_limit=2,
    use_mps_device=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

print("begin train")
try:
    trainer.train()
finally:
    model.save_pretrained("dapt_ckpt/manual_save")

print("end train")
model.save_pretrained("dapt_ckpt/final")