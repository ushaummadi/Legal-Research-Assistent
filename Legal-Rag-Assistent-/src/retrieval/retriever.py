from typing import List
from langchain_core.documents import Document
from loguru import logger
from src.ingestion.vector_store import VectorStoreManager
from config.settings import settings

class NativeRetriever:
    def __init__(self):
        self.vs = VectorStoreManager()
        self.collection = self.vs.collection
        self.embeddings = self.vs.embeddings
        logger.info(f"Retriever init: {self.vs.count()} vectors loaded")

    def get_relevant_documents(self, query: str) -> List[Document]:
        if self.vs.count() == 0:
            logger.warning("Chroma empty - no docs available")
            return [Document(page_content="No documents indexed. Run 'Rebuild Index'.", metadata={"source": "system", "score": 0})]

        # 1. Get raw vector results (already sorted by semantic similarity)
        query_vector = self.embeddings.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=20,  # Fetch top 20 candidates
            include=["documents", "metadatas", "distances"], # Include distances for real scoring
        )
        
        docs = []
        logger.info(f"Vector search: {len(results['documents'][0])} candidates for '{query}'")
        
        # 2. Process results
        # Chroma returns distances (lower is better). We convert to a similarity score (0-10).
        # Distance range varies by model, but often 0.0 (exact) to ~2.0 (unrelated).
        # Simple score conversion: score = (2.0 - distance) * 5. Clamp to 0-10.
        
        distances = results['distances'][0]
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]

        for i, text in enumerate(documents):
            if not text or not text.strip(): 
                continue
            
            dist = distances[i]
            meta = metadatas[i] or {}
            
            # Convert distance to 0-10 score (approximate)
            # Assuming cosine distance where 0 is identical and 1+ is different
            # Adjust scaling factor as needed. This ensures lower distance = higher score.
            raw_score = max(0.0, (1.5 - dist) * 6.6) # Example mapping: dist 0->10, dist 1.5->0
            score = round(min(10.0, raw_score), 1)

            # Store the semantic score directly
            meta["score"] = score
            
            docs.append(Document(page_content=text, metadata=meta))
            logger.debug(f"Doc {i}: dist={dist:.4f} -> score={score}")
        
        # 3. Sort by score (Highest first) and return top K
        # We trust the vector score now, not just keyword length
        ranked_docs = sorted(docs, key=lambda d: d.metadata.get("score", 0), reverse=True)
        
        logger.info(f"✅ Retrieved {len(ranked_docs)} docs for '{query}'")
        return ranked_docs[:10] # Return top 10 most relevant

def get_retriever():
    return NativeRetriever()
