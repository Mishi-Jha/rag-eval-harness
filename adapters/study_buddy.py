from langchain_core.messages import HumanMessage
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

class StudyBuddyGenerator:
    def __init__(self,client):
        self.client=client
    def generate(self, question: str,chunks:list[str]) -> str:
        prompt=""
        for chunk in chunks:
            prompt+=chunk
            
        content=f"Context:{prompt}\n\nQuestion:{question}"
        response=self.client.invoke([HumanMessage(content=content)])   
        return response.content
            