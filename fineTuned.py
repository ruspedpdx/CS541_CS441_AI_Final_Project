from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Replace 'your_model_directory' with the path to your fine-tuned model directory
model_path = f"C:/Users/tyler/OneDrive/Documents/finalAI/fine-tuned-model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Prepare the input text
input_text = "write me a grep command that uses -A 5"

inputs = tokenizer(input_text, return_tensors='pt')

# Make predictions
with torch.no_grad():
    outputs = model(**inputs)
logits = outputs.logits
predictions = logits.argmax(dim=-1)

# Print the prediction
print("Predicted class:", predictions.item())



