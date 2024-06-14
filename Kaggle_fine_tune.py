# Install required packages
!pip install transformers datasets peft bitsandbytes

import torch
from torch.optim import AdamW
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, BitsAndBytesConfig, AutoModelForSeq2SeqLM
from tqdm.auto import tqdm
import bitsandbytes as bnb
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = "google/flan-t5-small"

# LoRA configuration
lora_config = LoraConfig(
    r=8, lora_alpha=512, target_modules=['q', 'v'], lora_dropout=0.01, bias="none", task_type="SEQ_2_SEQ_LM"
)

# Load dataset
dataset = load_dataset('csv', data_files='data.csv')

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def preprocess_function(examples):
    inputs = examples["command"]
    targets = examples["description"]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length")
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)
tokenized_dataset.set_format("torch")

# Bits and Bytes configuration for quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, quantization_config=bnb_config, device_map={"":0}, trust_remote_code=True)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=1000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["train"].select(range(1000))  # Use a subset for evaluation
)

trainer.train()
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

model.save_pretrained("./pre-tuned-model")
tokenizer.save_pretrained("./fine-tuned-model")

torch.save(model.state_dict(), "./fine-tuned-model_state.pt")
