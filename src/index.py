# index.py - chunk the 10-K text, embed each chunk, store it all in Chroma.


import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()

TICKER = "AAPL"
COMPANY_NAME = "Apple"
SOURCE_FILE = "data/apple_10k.txt"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "sec_10k_filings"


def load_text(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def split_into_chunks(text: str) -> list[Document]:
    # ~512-token chunks, 50-token overlap so a fact sitting on a boundary
    # doesn't get cut in half
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=512,
        chunk_overlap=50,
    )
    chunks = splitter.split_text(text)

    # tag every chunk with its company - the ticker tag makes re-runs safe
    # (see below) and lets retrieval filter per company later on
    docs = []
    for chunk in chunks:
        doc = Document(
            page_content=chunk,
            metadata={"company": COMPANY_NAME, "ticker": TICKER, "source": SOURCE_FILE},
        )
        docs.append(doc)
    return docs


def index_documents(docs: list[Document]) -> None:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    store = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        # cosine, to match the methodology. Chroma defaults to L2 - same
        # ranking for these embeddings, but set it explicitly anyway
        collection_metadata={"hnsw:space": "cosine"},
    )

    # Chroma only ever APPENDS, so delete this company's old chunks first -
    # otherwise every re-run stores the same chunks again and retrieval
    # starts returning duplicates
    existing = store.get(where={"ticker": TICKER})
    old_ids = existing["ids"]
    if old_ids:
        store.delete(ids=old_ids)
        print(f"Removed {len(old_ids)} previously stored {TICKER} chunks (re-run detected).")

    store.add_documents(docs)


def main():
    print(f"Loading text from {SOURCE_FILE}...")
    text = load_text(SOURCE_FILE)
    print(f"Loaded {len(text):,} characters.")

    print("Splitting into chunks...")
    docs = split_into_chunks(text)
    print(f"Created {len(docs)} chunks.")

    print("Embedding and storing in Chroma (this may take a minute)...")
    index_documents(docs)
    print(f"Done. Vector store saved to {CHROMA_DIR}/")


if __name__ == "__main__":
    main()
