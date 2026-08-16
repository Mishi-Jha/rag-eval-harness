from sentence_transformers import SentenceTransformer

def recall_at_k(retrieved_chunks: list[str], golden_chunks: list[str], model, threshold: float = 0.6) -> float:
    embeddings_a=model.encode(retrieved_chunks)
    embeddings_b=model.encode(golden_chunks)
    similarities=model.similarity(embeddings_a,embeddings_b)
    matched_count=0
    for i in range(len(golden_chunks)):
        max_similarity=max(similarities[:,i])
        if max_similarity>=threshold:
            matched_count+=1
    recall=matched_count/len(golden_chunks)
    return recall

def precision_at_k(retrieved_chunks: list[str], golden_chunks: list[str], model, threshold: float = 0.6) -> float:
    embeddings_a=model.encode(retrieved_chunks)
    embeddings_b=model.encode(golden_chunks)
    similarities=model.similarity(embeddings_a,embeddings_b)
    matched_count=0
    for i in range(len(retrieved_chunks)):
        max_similarity=max(similarities[i,:])
        if max_similarity>=threshold:
            matched_count+=1
    precision=matched_count/len(retrieved_chunks)
    return precision