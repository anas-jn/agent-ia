from langchain_community.llms import OpenAI
from langchain_community.embeddings import OpenAIEmbeddings

import faiss
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

def create_index(chunks):
    embedder = OpenAIEmbeddings()
def create_index(chunks, embedder):
    vectors = embedder.embed_documents(chunks)
    index = faiss.IndexFlatL2(len(vectors[0]))
    index.add(np.array(vectors))
    index.add(np.array(vectors, dtype=np.float32))
def search_index(index, chunks, query, embedder, k=3):
    q_vec = embedder.embed_query(query)
    D, I = index.search(np.array([q_vec]), k)
    return [chunks[i] for i in I[0]]
    q_vec_np = np.array([q_vec], dtype=np.float32)
    D, I = index.search(q_vec_np, k)
    return [chunks[i] for i in I[0] if i != -1]
