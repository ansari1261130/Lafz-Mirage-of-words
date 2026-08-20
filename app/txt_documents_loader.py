from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.models import embedding_model

# --------------------------------------------------
# 1. Locate knowledge source
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
file_path = BASE_DIR / "knowledge" / "sample_poetry.txt"

# --------------------------------------------------
# 2. Load document
# --------------------------------------------------

loader = TextLoader(
    str(file_path),
    encoding="utf-8"
)
documents = loader.load()
print(f"Number of documents: {len(documents)}")

# --------------------------------------------------
# 3. Split document into chunks
# --------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)
chunks = text_splitter.split_documents(documents)
print(f"Number of chunks: {len(chunks)}")

# --------------------------------------------------
# 4. Inspect chunks
# --------------------------------------------------

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i} ---")
    print(chunk.page_content)
    print("Metadata:", chunk.metadata)

# --------------------------------------------------
# 5. Generate embeddings for chunks
# --------------------------------------------------

embedding_model_instance = embedding_model.model
chunk_texts = [
    chunk.page_content
    for chunk in chunks
]
embedded_documents = embedding_model_instance.embed_documents(
    chunk_texts
)

# --------------------------------------------------
# 6. Inspect document embeddings
# --------------------------------------------------
print("\n========== EMBEDDING INFORMATION ==========")
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
print(
    "First 5 values of first embedding:",
    embedded_documents[0][:5]
)

# --------------------------------------------------
# 7. Generate embedding for user query
# --------------------------------------------------

query = "eyes"
embedded_query = embedding_model_instance.embed_query(query)

# --------------------------------------------------
# 8. Inspect query embedding
# --------------------------------------------------

print("\n========== QUERY EMBEDDING ==========")
print("Query:", query)
print(
    "Query embedding dimensions:",
    len(embedded_query)
)
print(
    "First 5 values of query embedding:",
    embedded_query[:5]
)

# --------------------------------------------------
# 9. Verify dimensions
# --------------------------------------------------

if len(embedded_documents[0]) == len(embedded_query):
    print("\n✓ Document and query embedding dimensions match.")
else:
    print("\n✗ ERROR: Embedding dimensions do not match.")