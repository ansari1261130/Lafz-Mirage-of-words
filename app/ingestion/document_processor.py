from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents):
    """
    Split loaded documents into meaningful chunks.

    The splitter tries to preserve paragraphs and sentences
    before falling back to smaller units.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=[
            "\n\n",   # paragraph
            "\n",     # line
            "।",      # Hindi/Devanagari sentence
            ".",      # English sentence
            "?",      # question
            "!",      # exclamation
            " ",      # word
            ""        # character fallback
        ]
    )

    return text_splitter.split_documents(documents)