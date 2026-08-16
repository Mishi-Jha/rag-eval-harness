import os
import tempfile
import chromadb
import torch
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from adapters.study_buddy import StudyBuddyRetriever, StudyBuddyGenerator
from harness.runner import Harness
load_dotenv()
from golden_dataset.loader import load_golden_dataset
from metrics.retrieval import recall_at_k
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.PersistentClient(path=os.path.join(tempfile.gettempdir(), "chroma_db"))
client = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)
collection = chroma_client.get_or_create_collection(name="eval_notes")
retriever = StudyBuddyRetriever(model, collection)
generator = StudyBuddyGenerator(client)
harness = Harness(retriever, generator)
retriever.process_pdf(r"C:\Users\MISHI JHA\Desktop\OS_Module1_1-65.pdf")
data = load_golden_dataset("golden_dataset/os.json")
example = data[0]

retrieved = retriever.retrieve(example["question"])
score = recall_at_k(retrieved, example["source_chunks"], model)
print(score)
print(retrieved)


similarities = model.similarity(model.encode(retrieved), model.encode(example["source_chunks"]))
print(similarities)




