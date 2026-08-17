from langchain_nvidia_ai_endpoints import ChatNVIDIA
from dotenv import load_dotenv

load_dotenv()


llm = ChatNVIDIA(
    model="nvidia/nemotron-3.5-lightning-30b-a3b",
    temperature=0.7,
    max_completion_tokens=512,
    chat_template_kwargs={
        "enable_thinking": False
    },
)