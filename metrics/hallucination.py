import json
from langchain_core.messages import HumanMessage

def judge_groundedness(answer: str, chunks: list[str], client) -> dict:
    joined_chunks=""
    for chunk in chunks:
        joined_chunks+=chunk

    prompt = f"""
    Context: {joined_chunks}
    Answer: {answer}
    You are evaluating whether an AI-generated answer is factually grounded in the given context.
    Grounded means every claim in the answer is directly supported by the context. Hallucinated means the answer includes claims, facts, or details that are NOT present in or supported by the context.
    Respond with ONLY valid JSON in exactly this format, with no other text before or after:
    {{"verdict": "grounded", "reasoning": "brief one-sentence explanation"}}
    or
    {{"verdict": "hallucinated", "reasoning": "brief one-sentence explanation"}}
    """

    response=client.invoke([HumanMessage(content=prompt)])
    formatted_response=json.loads(response.content)
    print(repr(joined_chunks))
    return formatted_response