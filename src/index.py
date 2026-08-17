# index.py - chunk the 10-K text, embed each chunk, store it all in Chroma.


import argparse
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# tickers/names/paths live in ingest.py - one definition, not two copies
from ingest import COMPANIES, text_path

load_dotenv()

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "sec_10k_filings"


def load_text(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def split_into_chunks(text: str, ticker: str) -> list[Document]:
    # ~512-token chunks, 50-token overlap so a fact sitting on a boundary
    # doesn't get cut in half
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=512,
        chunk_overlap=50,
    )
    chunks = splitter.split_text(text)

    # tag every chunk with its company - the ticker tag makes re-runs safe
    # (see below) and lets evaluate.py spot chunks from the wrong company
    docs = []
    for chunk in chunks:
        doc = Document(
            page_content=chunk,
            metadata={
                "company": COMPANIES[ticker]["name"],
                "ticker": ticker,
                "source": text_path(ticker),
            },
        )
        docs.append(doc)
    return docs


def index_documents(docs: list[Document], ticker: str) -> None:
    # chunk_size here = chunks per API request, not the 512-token text size.
    # default 1000 overflows OpenAI's 300k-token request cap on big filings (JPM)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", chunk_size=100)

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
    existing = store.get(where={"ticker": ticker})
    old_ids = existing["ids"]
    if old_ids:
        store.delete(ids=old_ids)
        print(f"Removed {len(old_ids)} previously stored {ticker} chunks (re-run detected).")

    store.add_documents(docs)


def index_company(ticker: str) -> None:
    source_file = text_path(ticker)

    print(f"Loading text from {source_file}...")
    text = load_text(source_file)
    print(f"Loaded {len(text):,} characters.")

    print("Splitting into chunks...")
    docs = split_into_chunks(text, ticker)
    print(f"Created {len(docs)} chunks.")

    print("Embedding and storing in Chroma (this may take a minute)...")
    index_documents(docs, ticker)
    print(f"Done. Vector store saved to {CHROMA_DIR}/")


def main():
    parser = argparse.ArgumentParser(description="Chunk, embed and store a 10-K in Chroma.")
    parser.add_argument("--ticker", default="AAPL", choices=sorted(COMPANIES.keys()))
    parser.add_argument("--all", action="store_true", help="index all five companies")
    args = parser.parse_args()

    if args.all:
        tickers = sorted(COMPANIES.keys())
    else:
        tickers = [args.ticker]

    for ticker in tickers:
        index_company(ticker)


if __name__ == "__main__":
    main()
