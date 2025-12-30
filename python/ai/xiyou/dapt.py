from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

model_name = "/Users/huhao/models/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True
)

# LoRA 配置（DAPT 用小一点）
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

dataset = load_dataset(
    "text",
    data_files={"train": "data/xiyou_raw.txt"}
)

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=2048
    )

tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

args = TrainingArguments(
    output_dir="dapt_ckpt",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=1e-5,
    fp16=False,
    logging_steps=50,
    save_steps=500,
    save_total_limit=2
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

trainer.train()
model.save_pretrained("dapt_ckpt/final")