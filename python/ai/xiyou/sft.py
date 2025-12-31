import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType

# ========= 配置 =========
MODEL_PATH = "/Users/huhao/models/Qwen2.5-0.5B-Instruct"
DATA_PATH = "xiyou_instruct.jsonl"
OUTPUT_DIR = "./xiyou-lora-m4"

MAX_LENGTH = 768
DEVICE = "mps"
# ========================


def format_prompt(example):
    text = (
        "你是一个有帮助的中文助手。\n\n"
        f"### 问题：\n{example['instruction']}\n\n"
        f"### 回答：\n{example['output']}"
    )
    return {"text": text}


def tokenize(example):
    result = tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )
    # 关键：labels = input_ids
    result["labels"] = result["input_ids"].copy()
    return result



# tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH, trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token

# dataset
dataset = load_dataset("json", data_files=DATA_PATH)["train"]
dataset = dataset.map(format_prompt)
dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

# model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
model.to(DEVICE)

# LoRA（M4 必须小）
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=4,                 # 非常关键，别再大
    lora_alpha=8,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "v_proj",
        "up_proj", "down_proj"
    ],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# training args
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=2,
    learning_rate=2e-4,
    logging_steps=1,
    save_steps=5,
    save_total_limit=2,
    report_to="none",
    fp16=False,              # MPS 下必须关
    bf16=False,
    optim="adamw_torch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

print("begin train")
trainer.train()
model.save_pretrained(OUTPUT_DIR)
