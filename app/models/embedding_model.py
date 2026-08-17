from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from dotenv import load_dotenv
load_dotenv()

model = NVIDIAEmbeddings(
    model = "nemotron-3-embed-1b",
)