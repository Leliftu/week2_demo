# Mini Semantic Search: Python FAQs

## Project Overview

This project builds a semantic search system that retrieves the three most relevant Python FAQ questions and answers for a user's query.

Unlike keyword search, semantic search focuses on the meaning of text.

## Features

- Loads FAQ data from a CSV file using Pandas.
- Cleans text and removes incomplete records.
- Splits longer text into smaller chunks.
- Converts text into numerical embeddings.
- Stores embeddings using Chroma.
- Retrieves the top three relevant FAQs.
- Supports natural-language queries.

## Technologies Used

- Python
- Pandas
- Sentence Transformers
- ChromaDB

## Project Structure

mini_semantic_search/
├── faqs.csv
├── semantic_search.py
├── requirements.txt
├── README.md
└── mini_semantic_search.ipynb

## Installation

Create a virtual environment:

    python -m venv .venv

Activate it on Windows:

    .venv\Scripts\activate

Install the required libraries:

    python -m pip install -r requirements.txt

## Run the Project

    python semantic_search.py

The first run downloads the embedding model, so an internet connection is required.

## Example Queries

- How can I store a value in my program?
- My code cannot find a library.
- How can I get part of a string?

## How It Works

1. Load FAQ data.
2. Clean the data.
3. Chunk the text.
4. Generate embeddings.
5. Store vectors in Chroma.
6. Search for the three most similar FAQs.

## Future Improvements

- Add more FAQ categories.
- Save the vector database between runs.
- Improve search evaluation.
- Build a user interface.