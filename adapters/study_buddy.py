from langchain_core.messages import HumanMessage
from pypdf import PdfReader
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
    
    def process_pdf(self,file_path):
        model=self.model
        collection=self.collection
        reader=PdfReader(file_path)
        text=""
        for page in reader.pages:
            text+=page.extract_text()
        chunks = []
        for i in range(0, len(text), 500):
            chunks.append(text[i:i+500])

        embeddings = model.encode(chunks)
        collection.add(
            documents=chunks,
            embeddings=embeddings.tolist(),
            ids=[str(i) for i in range(len(chunks))]
        )    
    

class StudyBuddyGenerator:
    def __init__(self,client):
        self.client=client
        self.last_usage=None
    def generate(self, question: str,chunks:list[str]) -> str:
        prompt=""
        for chunk in chunks:
            prompt+=chunk
            
        content=f"Context:{prompt}\n\nQuestion:{question}"
        response=self.client.invoke([HumanMessage(content=content)])   
        self.last_usage=response.usage_metadata
        return response.content
            