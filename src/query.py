# query.py - answer a question twice with the SAME model (gpt-4o-mini):
#   RAG:      retrieve the most relevant 10-K chunks first, answer from those
#   Baseline: no context at all, model's own memory
# Same model both times, so any quality difference = retrieval. That's the
# whole experiment.

import sys

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()

# must match index.py, or the question vectors won't line up with the store
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "sec_10k_filings"
EMBEDDING_MODEL = "text-embedding-3-small"

# temperature 0 -> most repeatable answers
CHAT_MODEL = "gpt-4o-mini"
TEMPERATURE = 0
TOP_K = 5

# RAG rules: use ONLY the context, admit it when the answer isn't there
RAG_SYSTEM_PROMPT = (
    "You are a financial analyst assistant. Answer the question using ONLY the "
    "context extracts from the company's 10-K filing provided below. If the "
    "answer cannot be found in the context, say you cannot find it in the "
    "filing. Do not use outside knowledge. Answer concisely: give just the "
    "figure or fact asked for.\n\n"
    "Context:\n{context}"
)

# baseline gets the same role and rules, minus the context - otherwise the
# experiment changes two things at once (retrieval AND instructions)
BASELINE_SYSTEM_PROMPT = (
    "You are a financial analyst assistant. Answer the question about the "
    "company's 10-K filing. If you do not know the answer, say so. Answer "
    "concisely: give just the figure or fact asked for."
)


def load_vectorstore() -> Chroma:
    # reconnect to the store index.py built
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )


def retrieve(question: str, k: int = TOP_K) -> list[Document]:
    # embed the question, return the k nearest chunks
    store = load_vectorstore()
    return store.similarity_search(question, k=k)


def answer_rag(question: str, k: int = TOP_K) -> dict:
    # returns the chunks too, not just the answer - RAGAS needs them later
    # to judge faithfulness
    docs = retrieve(question, k=k)

    # join the chunk texts with a separator so the model can tell them apart
    chunk_texts = []
    for doc in docs:
        chunk_texts.append(doc.page_content)
    context = "\n\n---\n\n".join(chunk_texts)

    llm = ChatOpenAI(model=CHAT_MODEL, temperature=TEMPERATURE)
    messages = [
        ("system", RAG_SYSTEM_PROMPT.format(context=context)),
        ("human", question),
    ]
    response = llm.invoke(messages)

    return {
        "answer": response.content,
        "contexts": chunk_texts,
    }


def answer_baseline(question: str) -> str:
    # same model, same instructions, no context
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=TEMPERATURE)
    messages = [
        ("system", BASELINE_SYSTEM_PROMPT),
        ("human", question),
    ]
    response = llm.invoke(messages)
    return response.content


def main():
    # question from the command line, or ask for one
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
