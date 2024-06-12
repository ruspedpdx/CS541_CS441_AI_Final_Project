import torch
import pandas as pd
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments

device = torch.device("cuda")
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)

# Load the CSV file into a Hugging Face Dataset
dataset = load_dataset('csv', data_files='data.csv')

#Tokenize
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")

def preprocess_function(examples):
    inputs = examples["command"]
    targets = examples["description"]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    
    # Tokenize the targets with the tokenizer as target tokenizer
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

#model
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

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

model.save_pretrained("./fine-tuned-model")
tokenizer.save_pretrained("./fine-tuned-model")



