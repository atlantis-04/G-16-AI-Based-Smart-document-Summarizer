from uuid import uuid4
import chromadb
from chromadb.utils import embedding_functions
from backend.config import settings


class ChromaRetriever:
    """Manages document retrieval using ChromaDB with Sentence Transformers embeddings."""

    def __init__(self, collection_name: str = "rag_docs"):
        """Initialize ChromaDB client and collection."""
        self.collection_name = collection_name

        # Initialize embedding function
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.embedding_model
        )

        # Initialize persistent ChromaDB client
        self.client = chromadb.PersistentClient(path=settings.chroma_persist_dir)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )

    def ingest_documents(self, texts: list[str], metadatas: list[dict] = None) -> int:
        """
        Ingest documents into ChromaDB.
        
        Args:
            texts: List of document texts
            metadatas: Optional list of metadata dicts for each text
            
        Returns:
            Number of chunks added
        """
        if not texts:
            return 0

        if metadatas is None:
            metadatas = [{"source": f"doc_{i}"} for i in range(len(texts))]

        # Generate unique IDs for each text
        ids = [str(uuid4()) for _ in texts]

        # Add to collection
        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )

        return len(texts)

    def retrieve(self, query: str, n_results: int = 5) -> list[dict]:
        """
        Retrieve relevant documents from the collection.
        
        Args:
            query: Search query string
            n_results: Number of results to return
            
        Returns:
            List of dicts with keys: "text", "metadata", "distance"
        """
        # Check if collection is empty
        if self.collection.count() == 0:
            return []

        # Query the collection
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )

        # Transform results to expected format
        documents = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                documents.append({
                    "text": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] and results["metadatas"][0] else {},
                    "distance": results["distances"][0][i] if results["distances"] and results["distances"][0] else 0.0
                })

        return documents

    def collection_count(self) -> int:
        """Get the number of documents in the collection."""
        return self.collection.count()
