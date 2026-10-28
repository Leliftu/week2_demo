from pathlib import Path

import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer


# File and model settings
DATA_PATH = Path(file).with_name("faqs.csv")
MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "python_faqs"


def clean_text(value):
    """Remove extra spaces and handle missing values."""
    if pd.isna(value):
        return ""

    return " ".join(str(value).split())


def chunk_text(text, chunk_size=80, overlap=15):
    """Split longer text into overlapping word chunks."""
    words = text.split()

    if not words:
        return []

    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))

        if end >= len(words):
            break

        start = end - overlap

    return chunks


def load_faqs():
    """Load, clean, and prepare FAQ documents."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "faqs.csv must be in the same folder as this script."
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = {"question", "answer"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "CSV must contain question and answer columns."
        )

    df["question"] = df["question"].apply(clean_text)
    df["answer"] = df["answer"].apply(clean_text)

    df = df[
        (df["question"] != "") &
        (df["answer"] != "")
    ]

    df = df.drop_duplicates(
        subset=["question"]
    ).reset_index(drop=True)

    documents = []
    metadatas = []
    ids = []

    for index, row in df.iterrows():
        faq_text = (
            f"Question: {row['question']}\n"
            f"Answer: {row['answer']}"
        )

        chunks = chunk_text(faq_text)

        for chunk_index, chunk in enumerate(chunks):
            documents.append(chunk)

            metadatas.append({
                "question": row["question"],
                "answer": row["answer"],
                "faq_id": int(index),
                "chunk_id": int(chunk_index),
            })

            ids.append(
                f"faq-{index}-chunk-{chunk_index}"
            )

    return documents, metadatas, ids


def create_search_index(documents, metadatas, ids, model):
    """Generate embeddings and store them in Chroma."""
    client = chromadb.Client()

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    embeddings = model.encode(
        documents,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).tolist()

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    return collection


def search_faqs(query, collection, model, top_k=3):
    """Find the top three semantically similar FAQs."""
    query = clean_text(query)

    if not query:
        print("Please enter a question.")
        return

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, collection.count()),
        include=["metadatas", "distances"],
    )

    print(f"\nYour question: {query}")
    print("-" * 60)

    for rank, metadata in enumerate(
        results["metadatas"][0], start=1
    ):
        distance = results["distances"][0][rank - 1]
        similarity = 1 - distance

        print(f"\nMatch {rank}")
        print(f"Similarity: {similarity:.3f}")
        print(f"Question: {metadata['question']}")
        print(f"Answer: {metadata['answer']}")


def main():
    print("Loading and cleaning FAQ data...")

    documents, metadatas, ids = load_faqs()

    print(f"Prepared {len(documents)} text chunks.")
    print("Loading the embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Creating the semantic search index...")
    collection = create_search_index(
        documents, metadatas, ids, model
    )

    print("\nSearch is ready!")

    # Test queries with different wording
    examples = [
        "How can I store a value in my program?",
        "My code cannot find a library.",
        "How can I get part of a string?",
    ]

    for query in examples:
        search_faqs(query, collection, model)

    # Allow users to enter their own queries
    while True:
        query = input(
            "\nEnter your question (or type exit): "
        )

        if query.strip().lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        search_faqs(query, collection, model)


if __name__ == "__main__":
    main()