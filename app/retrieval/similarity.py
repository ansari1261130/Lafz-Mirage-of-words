import math


def cosine_similarity(vector_a, vector_b):
    """
    Calculate cosine similarity between two vectors.

    Formula:

        A · B
    -------------
    ||A|| ||B||

    Returns a value between -1 and 1.
    """

    if len(vector_a) != len(vector_b):
        raise ValueError(
            "Vectors must have the same dimensions."
        )

    # Dot product
    dot_product = sum(
        a * b
        for a, b in zip(vector_a, vector_b)
    )

    # Magnitude of vector A
    magnitude_a = math.sqrt(
        sum(a * a for a in vector_a)
    )

    # Magnitude of vector B
    magnitude_b = math.sqrt(
        sum(b * b for b in vector_b)
    )

    # Prevent division by zero
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


def similarity_search(
    query_vector,
    document_vectors,
    documents,
    top_k=3
):
    """
    Compare a query vector against document vectors
    and return the most similar documents.
    """

    if len(document_vectors) != len(documents):
        raise ValueError(
            "Number of vectors must match number of documents."
        )

    results = []

    for index, document_vector in enumerate(document_vectors):

        score = cosine_similarity(
            query_vector,
            document_vector
        )

        results.append({
            "index": index,
            "score": score,
            "document": documents[index]
        })

    # Highest similarity first
    results.sort(
        key=lambda result: result["score"],
        reverse=True
    )

    return results[:top_k]