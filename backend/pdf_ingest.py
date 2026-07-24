"""
Standalone script to ingest PDF documents into ChromaDB.
Usage: python -m backend.pdf_ingest
"""

from collections import Counter

from backend.pdf_loader import load_all_pdfs
from backend.retriever import ChromaRetriever


def ingest_pdfs() -> None:
    """Load PDFs from ./data and ingest their chunks into ChromaDB."""
    chunks = load_all_pdfs("./data")

    if not chunks:
        print("No PDF chunks found in ./data")
        print("Total chunks added: 0")
        return

    counts_by_file = Counter(chunk["metadata"]["source"] for chunk in chunks)
    for filename, count in sorted(counts_by_file.items()):
        print(f"{filename}: {count} chunks ready")

    texts = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]

    retriever = ChromaRetriever()
    chunks_added = retriever.ingest_documents(texts, metadatas)

    print(f"Total chunks added: {chunks_added}")
    print(f"Total documents in collection: {retriever.collection_count()}")


if __name__ == "__main__":
    ingest_pdfs()
