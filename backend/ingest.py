"""
Standalone script to ingest sample documents into ChromaDB.
Usage: python -m backend.ingest
"""

import sys
from pathlib import Path
from backend.retriever import ChromaRetriever


def ingest_sample_documents():
    """Read sample documents and ingest them into ChromaDB."""
    # Determine the path to sample_docs.txt
    sample_docs_path = Path(__file__).parent.parent / "data" / "sample_docs.txt"

    if not sample_docs_path.exists():
        print(f"❌ Error: {sample_docs_path} not found")
        sys.exit(1)

    # Read the file
    with open(sample_docs_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split by double newline to get individual topic paragraphs
    documents = [doc.strip() for doc in content.split("\n\n") if doc.strip()]

    if not documents:
        print("❌ Error: No documents found in sample_docs.txt")
        sys.exit(1)

    # Initialize retriever
    retriever = ChromaRetriever()

    # Check if documents are already ingested
    current_count = retriever.collection_count()
    if current_count > 0:
        print(f"⚠️  Collection already contains {current_count} documents.")
        response = input("Overwrite? (y/n): ").strip().lower()
        if response != "y":
            print("❌ Ingestion cancelled")
            sys.exit(0)

    # Ingest documents
    count = retriever.ingest_documents(documents)

    print(f"✅ Ingested {count} chunks successfully")
    print(f"📊 Total documents in collection: {retriever.collection_count()}")


if __name__ == "__main__":
    ingest_sample_documents()
