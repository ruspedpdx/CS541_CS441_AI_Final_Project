# Questions that need Answers
### RAG
- What documents will be loaded into the RAG system?
    - Question=Answer Pairs?
    - GNu docs?
    - other grep tutorials?
- Should we try to optimize colbert?
- Worth it to fix extractChunk file?

### FineTuning
- Is Kaggle best choice?
- How much data?
- Do we want to finetune with context in question answer pairs?

### DSPy
- Do we want to try using it for finetuning?
- Will we use the DSPy fewShot optimizers to generate CoT?
- Could also use ensembles or RandomSearch
- Best metric to test Rag System?





# Division of tasks
## 5/26-6/4
### Russ Pedersen:
- Learn how to use ColbertV2 by utilizing Ragatouille
- Make some examples
- Documentation
- Features: finetune? basemodel okay?
### Tyler Evans:
- Look into best way to serve Flan-T5 models
- Could we run them on CPU?
- What environment makes the most sense for testing our project? Local vm? Cloud vm? docker container?
### David Baker-Robinson:
- Generate synthetic data
- Create metric with DSPy for FlanT5
- Create metric with LangChain for baseline
- Split into train/test set
### Nusrat Ila:
 investigate different fine tuning methods   
 Domain-specific fine-tuning  
 Task-specific fine-tuning  
 fine tunes that people have already done that could relate to creating a natural language->grep agent  
### Kindy:
- Investigate T5-Flan models context window size
- create specific examples given our project
    - Include chain-of-thought prompt and context from RAG
        - Prompt: Find all instances of the word cat in this file example.txt
        - Answer: grep -i "cat" example.txt
        - Promot: Count the number of time the word cat appeared in example.txt
        - Answer: grep -c "cat" example.txt
        - Prompt: Show the lines that contain the word cat in example.txt
        - Answer: grep -n "cat" example.txt
    - If we do use T5-flan what is the optimal RAG chunk size
    - How many hop reasoning and/or examples should be included?


## 6/10-6/14
### Nusrat, Tyler, Kindy
- Create multiple QLorA finetunes
    - Adjust hyperparameters such as epochs, decay rate, etc.
    - Could also try different model sizes
    - Could also try finetune with QWen1.5 1.1B
- Save the Lora's
- Are the Lora's easily interchangeable?
- Run them in multiple environments i.e. CPU-only vs GPU, Kaggle, personal computer, etc
### Russ
- Creating metric for Ragatouille RAG system
- Loading/Curating documents
- Testing RagaTouille
- Help coordinate group members
### David
- Generate Synthetic Data
- Finetuning T5 models with DSPy
- Working with Russ on integrating RAG
- Devise agent architecture
- Evaluating different configurations

## Final Tasks
### Tyler, Nusrat, Kindy
1. Host large dataset on hugging face with data split into two random subsets(Dynamic RAG vs FineTune)
2. FineTune T5-Flan on the FineTune DataSet and load finetuned model to Hugging Face Model Hub
### Russ, David
3. FineTune ColBertV2 retrieval model on part of large training set
4. Find 1B model to work with RAG
### David
5. Create DSPy agent that utilizes dynamic few shot examples with ColbertV2+1B model
### (Nusrat and Kindy)
6. Compare T5-Flan and 1B model with context on Grep course problems
