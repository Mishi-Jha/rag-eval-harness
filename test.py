import os
import tempfile
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from adapters.study_buddy import StudyBuddyRetriever, StudyBuddyGenerator
from harness.runner import Harness
load_dotenv()

model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.PersistentClient(path=os.path.join(tempfile.gettempdir(), "chroma_db"))
collection = chroma_client.get_or_create_collection(name="notes")

client = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)


retriever = StudyBuddyRetriever(model, collection)
generator = StudyBuddyGenerator(client)
harness = Harness(retriever, generator)

answer = harness.run("What is a mutex?")
print(answer)