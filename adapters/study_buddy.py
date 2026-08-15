class StudyBuddyRetriever:
    def __init__(self,model,collection):
        self.model=model
        self.collection=collection
        
    
    def retrieve(self, question: str) -> list[str]:
        query_embedding=self.model.encode([question])
        results=self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=3
        )
        return results['documents'][0]