
from pathlib import Path

from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """Extract and chunk text from a PDF file."""
    path = Path(pdf_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = []

    reader = PdfReader(str(path))
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()

        if len(text) < 50:
            continue

        page_chunks = splitter.split_text(text)
        for chunk in page_chunks:
            chunks.append(
                {
                    "text": chunk,
                    "metadata": {
                        "source": path.name,
                        "page": page_num + 1,
                        "type": "pdf",
                    },
                }
            )

    return chunks


def load_all_pdfs(folder_path: str = "./data") -> list[dict]:
    """Load and chunk all PDF files in a folder."""
    folder = Path(folder_path)
    if not folder.exists():
        return []

    chunks = []
    for pdf_path in sorted(folder.glob("*.pdf")):
        chunks.extend(extract_text_from_pdf(str(pdf_path)))

    return chunks
