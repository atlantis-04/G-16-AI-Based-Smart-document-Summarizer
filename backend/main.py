import logging
from pathlib import Path
import shutil

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend.rag_pipeline import run_pipeline
from backend.retriever import ChromaRetriever
from backend.config import settings
from backend.pdf_loader import extract_text_from_pdf
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
ACTIVE_COLLECTION = "rag_docs"

# Create FastAPI app
app = FastAPI(
    title="Self-Healing RAG API",
    version="1.0.0",
    description="Retrieval-Augmented Generation with automatic self-healing via critic evaluation"
)
if os.path.exists("frontend"):
    app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class QueryRequest(BaseModel):
    """Request model for RAG query."""
    query: str
    max_results: int = 5
    conversation_history: list[dict] = []
    collection_name: str = "rag_docs"


class QueryResponse(BaseModel):
    """Response model for RAG query."""
    final_answer: str
    critique: str
    retry_count: int
    retrieved_chunks: list[dict]
    is_grounded: bool
    query: str
    conversation_history: list[dict] = []


class IngestRequest(BaseModel):
    """Request model for document ingestion."""
    texts: list[str]
    metadatas: list[dict] = []
    collection_name: str = "rag_docs"


class IngestResponse(BaseModel):
    """Response model for document ingestion."""
    chunks_added: int
    message: str


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    collection_count: int
    model: str


class CollectionRequest(BaseModel):
    """Request model for collection operations."""
    collection_name: str


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
def read_root():
    """Root endpoint with API information."""
    return {
        "message": "Self-Healing RAG API",
        "docs": "/docs",
        "version": "1.0.0"
    }


@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    Health check endpoint.
    Returns status, document count, and active model.
    """
    try:
        retriever = ChromaRetriever(collection_name=ACTIVE_COLLECTION)
        collection_count = retriever.collection_count()
        
        return HealthResponse(
            status="ok",
            collection_count=collection_count,
            model=settings.model_name
        )
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.get("/app")
def serve_frontend():
    """Serve the frontend app."""
    return FileResponse("frontend/index.html")


@app.get("/collections")
def list_collections():
    """List all available ChromaDB collections with document counts."""
    try:
        retriever = ChromaRetriever(collection_name=ACTIVE_COLLECTION)
        collections = []

        for collection in retriever.client.list_collections():
            name = collection.name if hasattr(collection, "name") else str(collection)
            collection_retriever = ChromaRetriever(collection_name=name)
            collections.append({
                "name": name,
                "count": collection_retriever.collection_count()
            })

        return {"collections": collections}
    except Exception as e:
        logger.error(f"Collection listing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Could not list collections: {str(e)}")


@app.post("/collections/create")
def create_collection(request: CollectionRequest):
    """Create a new ChromaDB collection."""
    collection_name = request.collection_name.strip()
    if not collection_name:
        raise HTTPException(status_code=400, detail="Collection name cannot be empty")

    try:
        ChromaRetriever(collection_name=collection_name)
        return {
            "created": collection_name,
            "message": f"Collection '{collection_name}' is ready"
        }
    except Exception as e:
        logger.error(f"Collection create error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Could not create collection: {str(e)}")


@app.post("/switch-collection")
def switch_collection(request: CollectionRequest):
    """Switch the server-side active collection used by default endpoints."""
    global ACTIVE_COLLECTION

    collection_name = request.collection_name.strip()
    if not collection_name:
        raise HTTPException(status_code=400, detail="Collection name cannot be empty")

    try:
        retriever = ChromaRetriever(collection_name=collection_name)
        ACTIVE_COLLECTION = collection_name
        return {
            "active_collection": ACTIVE_COLLECTION,
            "count": retriever.collection_count(),
            "message": f"Switched to collection '{ACTIVE_COLLECTION}'"
        }
    except Exception as e:
        logger.error(f"Collection switch error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Could not switch collection: {str(e)}")

@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    """
    Main RAG query endpoint.
    
    Takes a user question and returns:
    - final_answer: The generated answer
    - critique: Critic's evaluation
    - retry_count: Number of self-healing retries
    - retrieved_chunks: Source documents used
    - is_grounded: Whether answer passed critic evaluation
    
    The pipeline automatically retries up to 3 times if the answer
    is not grounded in the retrieved documents.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        logger.info(f"Processing query: {request.query[:100]}")
        
        # Run the RAG pipeline
        result = run_pipeline(
            request.query,
            conversation_history=request.conversation_history,
            collection_name=request.collection_name
        )
        
        is_grounded = result.get("is_grounded", False)
        
        # Map pipeline result to response model
        response = QueryResponse(
            final_answer=result.get("final_answer", ""),
            critique=result.get("critique", ""),
            retry_count=result.get("retry_count", 0),
            retrieved_chunks=result.get("retrieved_chunks", []),
            is_grounded=is_grounded,
            query=request.query,
            conversation_history=result.get("conversation_history", [])
        )
        
        logger.info(f"Query processed successfully. Retries: {response.retry_count}")
        return response
        
    except Exception as e:
        logger.error(f"Query processing error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {str(e)}"
        )


@app.post("/ingest", response_model=IngestResponse)
def ingest_endpoint(request: IngestRequest):
    """
    Document ingestion endpoint.
    
    Takes a list of texts and optional metadatas, and adds them
    to the ChromaDB vector store for future retrieval.
    """
    if not request.texts:
        raise HTTPException(status_code=400, detail="Texts list cannot be empty")
    
    try:
        logger.info(f"Ingesting {len(request.texts)} documents into {request.collection_name}")
        
        retriever = ChromaRetriever(collection_name=request.collection_name)
        
        # Prepare metadatas if not provided
        metadatas = request.metadatas if request.metadatas else None
        
        # Ingest documents
        chunks_added = retriever.ingest_documents(request.texts, metadatas)
        
        message = f"Successfully ingested {chunks_added} document(s). " \
                  f"Total documents in DB: {retriever.collection_count()}"
        
        logger.info(message)
        
        return IngestResponse(
            chunks_added=chunks_added,
            message=message
        )
        
    except Exception as e:
        logger.error(f"Ingestion error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Document ingestion failed: {str(e)}"
        )


@app.post("/upload-pdf")
def upload_pdf(
    file: UploadFile = File(...),
    collection_name: str = Form("rag_docs")
):
    """Upload a PDF, extract chunks, and ingest them into a ChromaDB collection."""
    filename = Path(file.filename or "").name
    if not filename or not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    temp_dir = Path("./temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    temp_path = temp_dir / filename

    try:
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        chunks = extract_text_from_pdf(str(temp_path))
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [
            {
                "source": filename,
                "type": "pdf_upload",
                "page": chunk.get("metadata", {}).get("page", 0),
            }
            for chunk in chunks
        ]

        retriever = ChromaRetriever(collection_name=collection_name)
        chunks_added = retriever.ingest_documents(texts, metadatas)
        total_docs = retriever.collection_count()

        return {
            "filename": filename,
            "chunks_added": chunks_added,
            "total_docs": total_docs,
            "message": f"Uploaded {filename} and added {chunks_added} chunks"
        }
    except Exception as e:
        logger.error(f"PDF upload error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"PDF upload failed: {str(e)}")
    finally:
        file.file.close()
        if temp_path.exists():
            temp_path.unlink()


# ============================================================================
# STARTUP EVENT
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize and log startup information."""
    logger.info("=" * 70)
    logger.info("Self-Healing RAG API started")
    logger.info(f"Using model: {settings.model_name}")
    logger.info(f"Using Groq: {settings.use_groq}")
    logger.info(f"Max retries: {settings.max_retries}")
    
    try:
        retriever = ChromaRetriever(collection_name=ACTIVE_COLLECTION)
        doc_count = retriever.collection_count()
        logger.info(f"Documents in DB: {doc_count}")
    except Exception as e:
        logger.warning(f"Could not fetch document count on startup: {e}")
    
    logger.info("=" * 70)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
