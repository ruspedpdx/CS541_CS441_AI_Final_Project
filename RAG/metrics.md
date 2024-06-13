## Simple metrics

```python
def validate_answer(example, pred, trace=None):
    return example.answer.lower() == pred.answer.lower()
```

convenient built-in utilities:

`dspy.evaluate.metrics.answer_exact_match`

`dspy.evaluate.metrics.answer_passage_match`

```python
def validate_context_and_answer(example, pred, trace=None):
    # check the gold label and the predicted answer are the same
    answer_match = example.answer.lower() == pred.answer.lower()

    # check the predicted answer comes from one of the retrieved contexts
    context_match = any((pred.answer.lower() in c) for c in pred.context)

    if trace is None: # if we're doing evaluation or optimization
        return (answer_match + context_match) / 2.0
    else: # if we're doing bootstrapping, i.e. self-generating good demonstrations of each step
        return answer_match and context_match
```

## Alternatively, using AI feedback

```python
# Define the signature for automatic assessments.
class Assess(dspy.Signature):
    """Assess the quality of the cli grep command produced."""

    assessed_text = dspy.InputField()
    assessment_question = dspy.InputField()
    assessment_answer = dspy.OutputField(desc="Yes or No")
```

Below is a simple metric that uses GPT-4 to check if a generated command

(1) is syntactically correct

(2) answers a given question correctly

```python
gpt4T = dspy.OpenAI(model='gpt-4', max_tokens=1000, model_type='chat')

def metric(gold, pred, trace=None):
    question, answer, command = gold.question, gold.answer, pred.output

    syntax = "Is this command format syntactically correct to return the requested data?"
    correct = f"The text should answer `{question}` with `{answer}`. Does the assessed text contain this answer?"

    with dspy.context(lm=gpt4T):
        syntax = dspy.Predict(Assess)(assessed_text=command, assessment_question=syntax)
        correct =  dspy.Predict(Assess)(assessed_text=command, assessment_question=correct)


    correct, syntax = [m.assessment_answer.lower() == 'yes' for m in [correct, syntax]]
    score = (correct + syntax) if correct else 0

    if trace is not None: return score >= 2
    return score / 2.0
```

based on examples from

https://dspy-docs.vercel.app/docs/building-blocks/metrics
