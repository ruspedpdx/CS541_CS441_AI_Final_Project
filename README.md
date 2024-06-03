# CS541_CS441_AI_Final_Project

Objective: Create a capable Linux command line assistant that leverages RAG and finetuning to bootstrap 
a small language model to effectively convert natural text to command line instructions

Language to use: Python
Libraries/frameworks: Ragatouille, answer.ai? QLoRA thing? DSPy? ColBERTV2, ARES

## Relevance to ML
Using metric-based approaches to create effective agents in a complex state space
Investigating NLP techniques through RAG and fine tuning
Learning about RAG involves understanding retrieval methodologies that hinge on embedding.
contextually rich vectors. A RAG method we’re using called ColbertV2 tries to achieve this
by using a transformer text-to-text architecture
Fine tuning involves optimization algorithms and tradeoffs in terms of hyperparameter selection
weight preservation and adaptation.

## Scope
Will the assistant attempt to generate commands for any task? How many Linux tools will it utilize?
What will user interaction look like?

## Which models to use
T5 - 100 million parameters 11 billion  
Phi-3B-mini  
Many 7B models to choose from maybe Llama 3 8B  

## Attack plan
Step 1. See if we can get T5 base working really well with generating grep commands.  
  - Measure performance for out of the box T5 base on grep commands
      - generate representative examples of natural language to grep commands
      - automate testing process
  - Provide the document or part of it as context (actual file involved)
  - Use RAG to improve
  - Basic LLM pipeline, could use ReAct Agent
Step 2. 
  - Iterate over process to improve results that are more representative of actual use cases
Step 3. 
 - More Tools! (maybe a future goal)

## Definition of tasks
### Show me the Data!
- Documents for RAG system
- Create dataset for assessing RAG accuracy
- Question-Answer pairs for fine tune
### RAG
- Some kind of technical documentation
- goal is to find best way to chunk data, and the right combination of technical options and descriptions
- Figure out how to use Ragatouille
- Get metrics on search results
### FINETUNE
- Setup a finetuning environment preferably with strong GPU
- Decide on method of quantization
- Decide on method of finetuning for model
- FineTune!
### AGENT
- Coordinate components of the system
- use DSPy to optimize LLM pipeline
- Find best agent architecture
### AUTOMATION
- For testing the assistants ability
- For data collection (web scraping or synthetic data generation)

## Future Research
- What improvements does QDoRA lead to vs QLoRA on agent performance across specific tool usage
- embed entire file system to allow directly integrated natural language look up
