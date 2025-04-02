from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
import torch
import json
import os
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(load_in_4bit=True)

BASE_MODEL = "deepseek-ai/DeepSeek-R1"
OUTPUT_DIR = "./output"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

# Load JSON dataset
def load_local_dataset(path="data/training_data.json"):
    with open(path) as f:
        data = json.load(f)
    return {"train": data}

def format_example(example):
    prompt = (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Context:\n{example['context']}\n\n"
        f"### Response:\n{example['response']}"
    )
    return tokenizer(prompt, padding="max_length", truncation=True, max_length=512)

# Load model with 4-bit quantization (QLoRA)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# Apply LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)

# Tokenized dataset
raw_data = load_local_dataset()
tokenized_data = list(map(format_example, raw_data["train"]))

# Training
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    logging_dir=f"{OUTPUT_DIR}/logs",
    num_train_epochs=3,
    save_total_limit=2,
    fp16=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data,
    tokenizer=tokenizer
)

trainer.train()
model.save_pretrained(OUTPUT_DIR)
