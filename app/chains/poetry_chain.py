from app.prompts import poetry_template
from app.models import chat_model
from langchain_core.output_parsers import StrOutputParser


chain = poetry_template.template | chat_model.llm | StrOutputParser()