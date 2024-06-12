import torch
from torch.optim import AdamW
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer,T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, get_scheduler, BitsAndBytesConfig, GenerationConfig, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from tqdm.auto import tqdm
import bitsandbytes as bnb
from peft import PeftModel, PeftConfig, prepare_model_for_kbit_training, LoraConfig, get_peft_model 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = "google/flan-t5-small"

# lora_config is effective only when peft_combine = False
lora_config = LoraConfig( r=8, lora_alpha=512, target_modules=['q', 'v'], lora_dropout=0.01, bias="none", task_type= "SEQ_2_SEQ_LM" )#,modules_to_save=["lm_head"])

# dataset = load_dataset("SKT27182/Preprocessed_OpenOrca", streaming=True)
Dataset = load_dataset('csv', data_files='data.csv')

data = { "command":[], "description": []}
            
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.model_max_length
            
def preprocess_function(examples):
    inputs = examples["command"]
    targets = examples["description"]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    
    # Tokenize the targets with the tokenizer as target tokenizer
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = Dataset.map(preprocess_function, batched=True)

tokenized_dataset.set_format("torch")
 
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, quantization_config=bnb_config, device_map={"":0}, trust_remote_code=True)
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

print(model.base_model.model.lm_head.weight.requires_grad)
model.base_model.model.lm_head.weight.requires_grad = True
print(model.base_model.model.lm_head.weight.requires_grad)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=60,
    per_device_eval_batch_size=60,
    warmup_steps=500,
    weight_decay=0.8,
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

# ./fine-tuned-model_    this file needs to be created before
torch.save(model.state_dict(), "./fine-tuned-model_")









