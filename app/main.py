from app.chains import poetry_chain

inputs = {
    "topic": "eyes",
    "mood": "romantic",
    "language": "Hinglish",
    "style": "ghazal",
    "length": "short"
}

res = poetry_chain.chain.invoke(
    inputs,
    thinking_mode=False
)

print(res)