from openai import AsyncAzureOpenAI
import asyncio
import os, json
from pydantic import BaseModel
from utils.api_utils import generate_from_openai_chat_completion_structured_format
from copy import deepcopy

class EvaluationOutput(BaseModel):
    score: float
    explanation: str
    


INSTRUCTION = """
Evaluate whether the model's final answer is correct by comparing it to the provided ground-truth answer.
ground-truth answer
Scoring criteria:
If the ground-truth answer is valid:
- Score 1: The model's response is fully consistent with or equivalent to the ground-truth answer.
- Score 0.5: The response is partially correct — it captures part of the ground-truth answer but is incomplete or contains minor inaccuracies.
- Score 0: The response is incorrect or does not align with the ground-truth answer.

If the ground-truth answer is unanswerable:
it means the question cannot be answered based on the available information.
If the model's response also indicates that the question is unanswerable, assign 1 point.
If the model attempts to answer an unanswerable question, assign 0 points.

First, assign a score based on the criteria above. Then, provide a brief explanation justifying your score.

"""



FULL_INSTRUCTION=INSTRUCTION + """
Output your response in the following structured format:
{
    "score": // float value, select a score from [0,0.5,1] due to Scoring criteria.
    "explanation": // str value, a brief explanation justifying your score
}
"""


def prepare_evaluation_message(example, response):
    user_prompt = ""
    mask_info = ""
    for i,item in enumerate(example['masked_elements']):
        mask_info += '[mask'+str(i+1)+'] is '+ item + '\n'

    question_context = f"Question: {example['masked_question']}"
    gt_answer = f"{mask_info}.Ground Truth Answer: {example['final_answer']}"
    model_response = f"Model Response to the Question: {response}"
    
    user_prompt = f"{question_context}\n\n{gt_answer}\n\n{model_response}"
    

    message = [
        {"role": "system", "content": FULL_INSTRUCTION},
        {"role": "user", "content": user_prompt},
    ]

    return message

async def get_acc_async(examples, client):
    evaluation_messages = [
        prepare_evaluation_message(example, example['response'])
        for example in examples
    ]
    
    # os.makedirs("cache", exist_ok=True)
    # json.dump(evaluation_messages, open("cache/evaluation_messages.json", "w"), indent=4, ensure_ascii=False)
    
    # Just await directly, no asyncio.run():
    outputs = await generate_from_openai_chat_completion_structured_format(     
        client=client, 
        messages=evaluation_messages,
        engine_name="gpt-4o", 
        max_tokens=2048,
        requests_per_minute=600,
        structured_format=EvaluationOutput
    )
    
    count = 0
    results = []
    for example, output in zip(examples, outputs):
        result = deepcopy(example)
        try:
            result["explanation"] = output.explanation
            result["score"] = output.score
        except Exception as e:
            result["explanation"] = ""
            result["score"] = 0
            print(f"Error: {e}")
        
        results.append(result)
        count += result["score"]
            
    return count / len(examples), results
