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
            return [Document(page_content="No documents indexed. Run 'Rebuild Index'.", metadata={"source": "system"})]

        query_vector = self.embeddings.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=20,
            include=["documents", "metadatas"],
        )
        
        docs = []
        query_words = [w for w in query.lower().split() if len(w) > 2]  # Ignore short words
        
        logger.info(f"Vector search: {len(results['documents'][0])} candidates for '{query}'")
        
        for i, (text, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
            if not text or not text.strip(): 
                continue
            
            text_lower = text.lower()
            score = 0
            
            # Keyword boost (any match, not all)
            for word in query_words:
                if word in text_lower:
                    score += len(word)  # Longer words = higher score
            
            # Vector relevance fallback
            if score > 0 or len(text) > 100:
                docs.append(Document(page_content=text, metadata={** (meta or {}), "score": score}))
                logger.debug(f"Doc {i}: score={score}, len={len(text)}")
        
        logger.info(f"✅ Retrieved {len(docs)} docs for '{query}'")
        return sorted(docs, key=lambda d: d.metadata.get("score", 0), reverse=True)[:8]

def get_retriever():
    return NativeRetriever()
