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
