from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from dotenv import load_dotenv
load_dotenv()

model = NVIDIAEmbeddings(
    model="nvidia/llama-nemotron-embed-1b-v2"
)