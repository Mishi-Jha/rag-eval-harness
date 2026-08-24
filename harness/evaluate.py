from metrics.retrieval import recall_at_k
from metrics.hallucination import judge_groundedness
from metrics.cost import calculate_cost
def evaluate_one_question(question: str, chunks_golden: list[str], retriever, generator, model, client):
    """Run all metrics on one question. Return a dict with all scores."""
    
    retrieved_chunks = retriever.retrieve(question)
    
    recall = recall_at_k(retrieved_chunks, chunks_golden, model, threshold=0.6)
    
    answer = generator.generate(question, retrieved_chunks)
    
    hallucination = judge_groundedness(answer, chunks_golden, client)
    
    cost = calculate_cost(generator.last_usage)
    
    return {
        "recall": recall,
        "hallucination": hallucination,
        "cost": cost
    }