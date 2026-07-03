# This file is the ONLINE query pipeline. Given a question, it produces two
# answers from the SAME model (GPT-4o-mini):
#   - RAG:      retrieve the most relevant 10-K chunks, then answer using them.
#   - Baseline: answer the same question with no document context.
# Holding the model constant means any quality difference is attributable to
# retrieval alone, which is the core experiment of the dissertation.

import sys

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()

# These must match how the vector store was built in index.py, or the question
# vectors won't be comparable to the chunk vectors.
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "sec_10k_filings"
EMBEDDING_MODEL = "text-embedding-3-small"

# Same generator for both pipelines. temperature=0 makes answers deterministic
# so the experiment is reproducible.
CHAT_MODEL = "gpt-4o-mini"
TEMPERATURE = 0
TOP_K = 5

# The RAG system prompt forces the model to ground its answer in the retrieved
# context and to admit when the answer isn't there. This is what makes RAG
# "faithful" — it should not fall back on its own memory.
RAG_SYSTEM_PROMPT = (
    "You are a financial analyst assistant. Answer the question using ONLY the "
    "context extracts from the company's 10-K filing provided below. If the "
    "answer cannot be found in the context, say you cannot find it in the "
    "filing. Do not use outside knowledge.\n\n"
    "Context:\n{context}"
)


def load_vectorstore() -> Chroma:
    """Reconnect to the persisted Chroma store built by index.py."""
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )


def retrieve(question: str, k: int = TOP_K) -> list[Document]:
    """Embed the question and return the k most similar chunks (cosine)."""
    store = load_vectorstore()
    return store.similarity_search(question, k=k)


def answer_rag(question: str, k: int = TOP_K) -> dict:
    """RAG answer: retrieve context, then answer grounded in it.

    Returns both the answer and the retrieved contexts. The contexts are kept
    so the later RAGAS evaluation can judge faithfulness against them.
    """
    docs = retrieve(question, k=k)
    context = "\n\n---\n\n".join(doc.page_content for doc in docs)

    llm = ChatOpenAI(model=CHAT_MODEL, temperature=TEMPERATURE)
    messages = [
        ("system", RAG_SYSTEM_PROMPT.format(context=context)),
        ("human", question),
    ]
    response = llm.invoke(messages)

    return {
        "answer": response.content,
        "contexts": [doc.page_content for doc in docs],
    }


def answer_baseline(question: str) -> str:
    """Baseline answer: same model, no context — answers from its own memory."""
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=TEMPERATURE)
    response = llm.invoke([("human", question)])
    return response.content


def main():
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("Question: ").strip()

    print(f"\nQuestion: {question}\n")

    rag = answer_rag(question)
    baseline = answer_baseline(question)

    print("=" * 70)
    print("RAG ANSWER (grounded in retrieved 10-K context)")
    print("=" * 70)
    print(rag["answer"])

    print("\n" + "=" * 70)
    print("BASELINE ANSWER (no context, model's own knowledge)")
    print("=" * 70)
    print(baseline)

    print("\n" + "=" * 70)
    print(f"RETRIEVED CONTEXT ({len(rag['contexts'])} chunks)")
    print("=" * 70)
    for i, ctx in enumerate(rag["contexts"], 1):
        preview = ctx.replace("\n", " ")[:200]
        print(f"\n[{i}] {preview}...")


if __name__ == "__main__":
    main()
