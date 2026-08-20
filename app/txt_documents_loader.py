from langchain_community.document_loaders import TextLoader
# from app.models import embedding_model
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter


BASE_DIR = Path(__file__).resolve().parent

file_path = BASE_DIR / "knowledge" / "sample_poetry.txt"
loader = TextLoader(str(file_path))

documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10, chunk_overlap=0)
texts = text_splitter.split_text(documents[0].page_content)


# print(type(documents))
# print(len(documents))

# print(type(documents[0]))
# print(documents[0].page_content)
# print(documents[0].metadata)
print(text_splitter)
print(texts)