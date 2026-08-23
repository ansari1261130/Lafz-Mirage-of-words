from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader

from app.models import embedding_model
from app.ingestion.document_processor import split_documents
from app.retrieval.similarity import similarity_search


# ==================================================
# 1. LOCATE KNOWLEDGE SOURCE
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

file_path = (
    BASE_DIR
    / "knowledge"
    / "chief_writes"
    / "Arzoo e Lafz.pdf"
)


# ==================================================
# 2. LOAD PDF
# ==================================================

loader = PyPDFLoader(
    str(file_path)
)

documents = loader.load()

print(
    f"Number of pages: {len(documents)}"
)


# ==================================================
# 3. SPLIT DOCUMENTS
# ==================================================

chunks = split_documents(documents)

print(
    f"Number of chunks: {len(chunks)}"
)


# ==================================================
# 4. DISPLAY CHUNKS
# ==================================================

print("\n========== DOCUMENT CHUNKS ==========")

for i, chunk in enumerate(chunks):

    print("\n" + "=" * 70)
    print(f"CHUNK {i}")
    print("=" * 70)

    print(chunk.page_content)

    print("\nMetadata:")
    print(chunk.metadata)


# ==================================================
# 5. GET EMBEDDING MODEL
# ==================================================

embedding_model_instance = (
    embedding_model.model
)


# ==================================================
# 6. EXTRACT TEXT FROM CHUNKS
# ==================================================

chunk_texts = [
    chunk.page_content
    for chunk in chunks
]


# ==================================================
# 7. CREATE DOCUMENT EMBEDDINGS
# ==================================================

print(
    "\nGenerating document embeddings..."
)

embedded_documents = (
    embedding_model_instance.embed_documents(
        chunk_texts
    )
)


print("\n========== EMBEDDING INFO ==========")

print(
    "Number of chunks:",
    len(chunks)
)

print(
    "Number of embeddings:",
    len(embedded_documents)
)

print(
    "Embedding dimensions:",
    len(embedded_documents[0])
)


# ==================================================
# 8. USER QUERY
# ==================================================

query = (
    "How does the poet describe "
    "the beauty of her face?"
)


# ==================================================
# 9. CREATE QUERY EMBEDDING
# ==================================================

embedded_query = (
    embedding_model_instance.embed_query(
        query
    )
)


print("\n========== QUERY ==========")

print("Query:")
print(query)

print(
    "\nQuery embedding dimensions:",
    len(embedded_query)
)


# ==================================================
# 10. PERFORM SIMILARITY SEARCH
# ==================================================

results = similarity_search(
    query_vector=embedded_query,
    document_vectors=embedded_documents,
    documents=chunks,
    top_k=3
)


# ==================================================
# 11. DISPLAY TOP RESULTS
# ==================================================

print(
    "\n========== TOP SIMILAR RESULTS =========="
)

for rank, result in enumerate(
    results,
    start=1
):

    document = result["document"]

    print("\n" + "=" * 70)
    print(f"RANK {rank}")
    print("=" * 70)

    print(
        f"Similarity Score: "
        f"{result['score']:.4f}"
    )

    print(
        f"Chunk Index: "
        f"{result['index']}"
    )

    print(
        f"Page: "
        f"{document.metadata.get('page', 'Unknown')}"
    )

    print("\nContent:")
    print(document.page_content)

    print("\nMetadata:")
    print(document.metadata)