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
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=512,
        chunk_overlap=50,
    )
    chunks = splitter.split_text(text)
    return [
        Document(
            page_content=chunk,
            metadata={"company": COMPANY_NAME, "ticker": TICKER, "source": SOURCE_FILE},
        )
        for chunk in chunks
    ]


def index_documents(docs: list[Document]) -> None:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
    )


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
