import json
from metrics.retrieval import recall_at_k
from metrics.hallucination import judge_groundedness
from metrics.cost import calculate_cost
from datetime import datetime

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

def run_full_evaluation(retriever,generator,model,client,golden_data,model_name:str="openai/gpt-oss-120b"):
    """Run evaluation on all golden examples. Save to results/ folder."""
    results=[]
    for example in golden_data:
        print(f"Evaluating {example['id']}...")
        result=evaluate_one_question(
            question=example["question"],
            chunks_golden=example["source_chunks"],
            retriever=retriever,
            generator=generator,
            model=model,
            client=client
        )
        results.append({
            "question_id":example["id"],
            "question":example["question"],
            **result
        })
        print(f"  ✓ {example['id']} done")
    timestamp=datetime.now().isoformat()
    filename=f"results/eval_{timestamp.replace(':','-')}.json"
    output={
        "timestamp":timestamp,
        "model":model_name,
        "results":results
    }
    with open(filename,'w')as f:
        json.dump(output,f,indent=2)
    print(f"Saved results to {filename}")
    return output       