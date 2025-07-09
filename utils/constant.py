from string import Template
MAX_TOKENS = 1024
GENERATION_TEMPERATURE = 1.0
GENERATION_SEED = 215

Mask1_COT_PROMPT = Template("""
<Question>: $question  
<Context>: $context  
<Caption>: $caption  

You are provided with a full-structure diagram from a research paper. Certain areas within the diagram have been annotated. Using both the diagram and the accompanying context, your task is to answer a question that includes a [MASK].

The [mask1] refers to the content highlighted by a red box in the image. Your first step is to perform image-text alignment by understanding the diagram in relation to the textual context. Then, the main task is to reason through <Question> step by step using a chain-of-thought approach to arrive at the correct answer.
If the question is completely unanswerable based on the context, simply respond with "unanswerable."

""")

Mask2_COT_PROMPT = Template("""
<Question>: $question  
<Context>: $context  
<Caption>: $caption  

You are provided with a full-structure diagram from a research paper. Certain areas within the diagram have been annotated. Using both the diagram and the accompanying context, your task is to answer a question that includes a [MASK].

The [mask1] refers to the content highlighted by a red box in the image.The [mask2] refers to the content highlighted by a blue box in the image.
Your first step is to perform image-text alignment by understanding the diagram in relation to the textual context. Then, the main task is to reason through the <Question> step by step using a chain-of-thought approach to arrive at the correct answer.
If the question is completely unanswerable based on the context, simply respond with "unanswerable."
""")



COT_PROMPT = {
    1: Mask1_COT_PROMPT,
    2: Mask2_COT_PROMPT
}

