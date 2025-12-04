import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import json
import os

class RAGRetriever:
    """
    Simple RAG (Retrieval-Augmented Generation) system using FAISS
    for retrieving relevant evidence to support hallucination detection
    """
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize RAG retriever with sentence transformer model
        
        Args:
            embedding_model: HuggingFace model name for embeddings
        """
        print(f"Loading embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)
        self.dimension = self.embedder.get_sentence_embedding_dimension()
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
        self.metadata = []
        
        print(f"RAG Retriever initialized with dimension: {self.dimension}")
    
    def add_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None):
        """
        Add documents to the FAISS index
        
        Args:
            documents: List of text documents to index
            metadata: Optional metadata for each document
        """
        if not documents:
            return
        
        print(f"Adding {len(documents)} documents to index...")
        
        # Generate embeddings
        embeddings = self.embedder.encode(documents, convert_to_numpy=True)
        
        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        # Store documents and metadata
        self.documents.extend(documents)
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{"doc_id": i} for i in range(len(documents))])
        
        print(f"Total documents in index: {len(self.documents)}")
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """
        Retrieve top-k most relevant documents for a query
        
        Args:
            query: Query text
            k: Number of documents to retrieve
            
        Returns:
            List of dictionaries containing documents and their scores
        """
        if len(self.documents) == 0:
            return []
        
        # Encode query
        query_embedding = self.embedder.encode([query], convert_to_numpy=True)
        
        # Search in FAISS index
        k = min(k, len(self.documents))
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Prepare results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                results.append({
                    "document": self.documents[idx],
                    "score": float(distances[0][i]),
                    "metadata": self.metadata[idx] if idx < len(self.metadata) else {},
                    "rank": i + 1
                })
        
        return results
    
    def save_index(self, path: str):
        """Save FAISS index and documents to disk"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, f"{path}.faiss")
        
        # Save documents and metadata
        with open(f"{path}.json", 'w') as f:
            json.dump({
                "documents": self.documents,
                "metadata": self.metadata
            }, f)
        
        print(f"Index saved to {path}")
    
    def load_index(self, path: str):
        """Load FAISS index and documents from disk"""
        # Load FAISS index
        self.index = faiss.read_index(f"{path}.faiss")
        
        # Load documents and metadata
        with open(f"{path}.json", 'r') as f:
            data = json.load(f)
            self.documents = data["documents"]
            self.metadata = data["metadata"]
        
        print(f"Index loaded from {path}. Total documents: {len(self.documents)}")


class KnowledgeBase:
    """
    Pre-built knowledge base for common topics
    You can populate this with reliable sources for verification
    """
    
    @staticmethod
    def get_sample_documents() -> List[str]:
        """
        Return sample documents for testing
        In production, replace with actual reliable sources
        """
        return [
            "The Earth orbits the Sun in approximately 365.25 days, which is why we have leap years every four years.",
            "Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at sea level under standard atmospheric pressure.",
            "The speed of light in vacuum is approximately 299,792,458 meters per second, often rounded to 3 x 10^8 m/s.",
            "Python is a high-level programming language created by Guido van Rossum and first released in 1991.",
            "The human body has 206 bones in adulthood, though babies are born with around 270 bones that fuse together over time.",
            "The capital of France is Paris, which is also the most populous city in France with over 2 million residents.",
            "DNA stands for deoxyribonucleic acid and contains the genetic instructions for all living organisms.",
            "The Pacific Ocean is the largest and deepest ocean on Earth, covering about 63 million square miles.",
            "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen.",
            "The United States Declaration of Independence was signed on July 4, 1776, marking the birth of the nation.",
        ]
    
    @staticmethod
    def get_sample_metadata() -> List[Dict]:
        """Return metadata for sample documents"""
        topics = [
            "Astronomy", "Chemistry", "Physics", "Computer Science", "Biology",
            "Geography", "Molecular Biology", "Oceanography", "Botany", "History"
        ]
        return [{"topic": topic, "reliability": "high"} for topic in topics]


def create_default_knowledge_base() -> RAGRetriever:
    """
    Create a RAG retriever with default knowledge base
    """
    retriever = RAGRetriever()
    
    # Add sample documents
    docs = KnowledgeBase.get_sample_documents()
    metadata = KnowledgeBase.get_sample_metadata()
    retriever.add_documents(docs, metadata)
    
    return retriever


# Example usage and testing
if __name__ == "__main__":
    print("Testing RAG Retriever...")
    
    # Create retriever
    retriever = create_default_knowledge_base()
    
    # Test queries
    test_queries = [
        "How long does it take Earth to orbit the Sun?",
        "What is the speed of light?",
        "Who created Python programming language?",
        "How many bones does a human have?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        results = retriever.retrieve(query, k=3)
        
        for result in results:
            print(f"\nRank {result['rank']} (Score: {result['score']:.4f})")
            print(f"Document: {result['document'][:100]}...")
            print(f"Metadata: {result['metadata']}")
    
    # Save index
    retriever.save_index("./data/knowledge_base")
    print("\n\nKnowledge base saved successfully!")