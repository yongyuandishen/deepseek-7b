import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

# === 基础配置 ===
model_name_or_path = "/home/zdb/deepseek-7b/deepseek-llm-7b-chat"
dataset_path = "/home/zdb/deepseek-7b/deepseek_emotion_master_style.jsonl"
output_dir = "/home/zdb/deepseek-7b/ds7b_emotion_lora"

# 设置设备：使用多GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 加载 tokenizer 和 4bit 模型 ===
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

# 加载本地模型，禁用 Hugging Face Hub 的远程仓库验证
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    trust_remote_code=True,
    device_map="auto",
    local_files_only=True  # ⚠️ 关键：禁用远程仓库验证
)

# === 加 LoRA 层 ===
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

# 使用 DataParallel 包装模型，确保多GPU训练
if torch.cuda.device_count() > 1:
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(model)

# === 加载数据 ===
def preprocess(example):
    return tokenizer(
        f"用户：{example['input']}\n助手：{example['output']}\n理由：{example['rationale']}",
        truncation=True, padding="max_length", max_length=512
    )

# 加载数据集
dataset = load_dataset("json", data_files=dataset_path)
tokenized_dataset = dataset["train"].map(preprocess, remove_columns=dataset["train"].column_names)

# === 设置 Trainer 参数 ===
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=64,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    fp16=True,
    logging_dir="./logs",
    report_to="none",
    resume_from_checkpoint=True if os.path.exists(os.path.join(output_dir, "checkpoint-200")) else None,
    gradient_checkpointing=True,
    no_cuda=False,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# === 启动训练 ===
trainer.train()

# === 保存最终模型 ===
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"模型训练完毕，保存在：{output_dir}")