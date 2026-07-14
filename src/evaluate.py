# evaluate.py - scores the answers. query.py answers, this file marks.
#
# For each question in data/qa_pairs.json:
#   1. get both answers (RAG + baseline, reusing query.py)
#   2. grade them:
#        - RAGAS: an LLM does the marking. Main metric.
#        - EM / F1 / contains: plain string+number checks, free, deterministic.
#   3. print a table, save everything to results/
#
# Note: 3 of the 4 RAGAS metrics judge the retrieved context. Baseline has no
# retrieval, so it only gets Answer Relevance + the string checks.

import os
import re
import json
import string
import collections

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,   # RAGAS's name for "Context Relevance"
    context_recall,
)

# reuse the actual pipeline from query.py - don't rebuild it here
from query import answer_rag, answer_baseline, EMBEDDING_MODEL, TEMPERATURE

load_dotenv()

QA_FILE = "data/qa_pairs.json"
RESULTS_DIR = "results"

# the marker. Deliberately stronger than the generator, so gpt-4o-mini isn't
# grading its own answers (self-preference bias). Generator stays gpt-4o-mini.
JUDGE_MODEL = "gpt-4o"

# which RAGAS metrics each system gets
RAG_METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]
BASELINE_METRICS = [answer_relevancy]


# --------------------------------------------------------------------------- #
# String/number checks: EM, F1, contains. No API calls, no cost.
# --------------------------------------------------------------------------- #
def normalize(text: str) -> str:
    # lowercase, strip punctuation, drop a/an/the, squash spaces
    text = text.lower()

    kept_chars = []
    for ch in text:
        if ch not in string.punctuation:
            kept_chars.append(ch)
    text = "".join(kept_chars)

    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, ground_truth: str) -> float:
    # 1.0 only if identical after cleaning
    return float(normalize(prediction) == normalize(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    # word overlap between answer and truth, 0 to 1
    pred_tokens = normalize(prediction).split()
    gt_tokens = normalize(ground_truth).split()

    if not pred_tokens or not gt_tokens:
        # empty only matches empty
        if pred_tokens == gt_tokens:
            return 1.0
        else:
            return 0.0

    common = collections.Counter(pred_tokens) & collections.Counter(gt_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)   # shared words / what the model said
    recall = overlap / len(gt_tokens)        # shared words / what the truth was
    return 2 * precision * recall / (precision + recall)


def numbers_in(text: str) -> list[float]:
    # pull the numbers out of text, KEEPING the decimal point:
    # "29,915" -> 29915.0, "$97.0" -> 97.0
    # (normalize() strips punctuation, which would make $97.0 and $9.70
    # come out identical - can't use it for money)
    found = []
    for match in re.findall(r"\d[\d,]*\.?\d*", text):
        cleaned = match.replace(",", "")
        try:
            found.append(float(cleaned))
        except ValueError:
            pass
    return found


def _close_enough(a: float, b: float) -> bool:
    # numbers within 0.5% count as the same fact. Absorbs rounding
    # (96.995 vs 97.0) but still fails real wrong answers
    # (164,000 vs 161,000 is 1.9% off)
    if a == b:
        return True
    largest = max(abs(a), abs(b))
    difference = abs(a - b)
    if difference / largest < 0.005:
        return True
    else:
        return False


def contains_answer(prediction: str, ground_truth: str) -> float:
    # does the key fact appear ANYWHERE in the answer?
    # - truth has numbers: at least one of them must show up (within
    #   tolerance). "at least one" because some ground truths give the same
    #   figure in two units, and a correct answer states one of them.
    # - no numbers ("Ernst & Young LLP"): substring check instead.
    # A correct full-sentence answer scores 1.0 - the case EM gets wrong.
    gt_numbers = numbers_in(ground_truth)
    if gt_numbers:
        pred_numbers = numbers_in(prediction)
        for value in gt_numbers:
            for pred_value in pred_numbers:
                if _close_enough(pred_value, value):
                    return 1.0
        return 0.0
    else:
        if normalize(ground_truth) in normalize(prediction):
            return 1.0
        else:
            return 0.0


# --------------------------------------------------------------------------- #
# Step 1 - ask both systems everything
# --------------------------------------------------------------------------- #
def run_systems(qa_pairs: list[dict]) -> list[dict]:
    records = []
    for i, qa in enumerate(qa_pairs, 1):
        question = qa["question"]
        gt = qa["ground_truth"]
        print(f"[{i}/{len(qa_pairs)}] {qa['id']}  {question}")

        rag = answer_rag(question)          # answer + the chunks it used
        baseline = answer_baseline(question)

        # one dict per question. EM/F1/contains are free, so compute them now
        records.append({
            "id": qa["id"],
            "type": qa["type"],
            "question": question,
            "ground_truth": gt,
            "rag_answer": rag["answer"],
            "contexts": rag["contexts"],
            "baseline_answer": baseline,
            "rag_em": exact_match(rag["answer"], gt),
            "rag_f1": f1_score(rag["answer"], gt),
            "rag_contains": contains_answer(rag["answer"], gt),
            "baseline_em": exact_match(baseline, gt),
            "baseline_f1": f1_score(baseline, gt),
            "baseline_contains": contains_answer(baseline, gt),
        })
    return records


# --------------------------------------------------------------------------- #
# Step 2 - RAGAS grading
# --------------------------------------------------------------------------- #
def build_judge():
    # the LLM + embeddings RAGAS marks with
    llm = LangchainLLMWrapper(ChatOpenAI(model=JUDGE_MODEL, temperature=TEMPERATURE))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=EMBEDDING_MODEL))
    return llm, embeddings


def run_ragas(records: list[dict]) -> None:
    llm, embeddings = build_judge()

    # RAG has context -> all four metrics
    rag_samples = []
    for r in records:
        sample = SingleTurnSample(
            user_input=r["question"],
            response=r["rag_answer"],
            retrieved_contexts=r["contexts"],
            reference=r["ground_truth"],
        )
        rag_samples.append(sample)
    print("\nScoring RAG with RAGAS (4 metrics)...")
    rag_df = evaluate(
        dataset=EvaluationDataset(samples=rag_samples),
        metrics=RAG_METRICS,
        llm=llm,
        embeddings=embeddings,
    ).to_pandas()

    # baseline -> Answer Relevance only
    base_samples = []
    for r in records:
        sample = SingleTurnSample(
            user_input=r["question"],
            response=r["baseline_answer"],
        )
        base_samples.append(sample)
    print("Scoring Baseline with RAGAS (Answer Relevance only)...")
    base_df = evaluate(
        dataset=EvaluationDataset(samples=base_samples),
        metrics=BASELINE_METRICS,
        llm=llm,
        embeddings=embeddings,
    ).to_pandas()

    # RAGAS keeps the rows in the order they were sent, so row i = record i
    for i, r in enumerate(records):
        rag_scores = {}
        for m in RAG_METRICS:
            rag_scores[m.name] = _safe_float(rag_df.iloc[i][m.name])
        r["rag_ragas"] = rag_scores

        baseline_scores = {}
        baseline_scores["answer_relevancy"] = _safe_float(base_df.iloc[i]["answer_relevancy"])
        r["baseline_ragas"] = baseline_scores


def _safe_float(value) -> float | None:
    # RAGAS returns NaN when it fails to score something - turn that into
    # None so the averages don't break
    try:
        f = float(value)
        # NaN is never equal to itself
        if f != f:
            return None
        else:
            return round(f, 3)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Step 3 - average, print, save
# --------------------------------------------------------------------------- #
def _mean(values: list) -> float | None:
    # average, skipping Nones
    nums = []
    for v in values:
        if isinstance(v, (int, float)):
            nums.append(v)

    if nums:
        return round(sum(nums) / len(nums), 3)
    else:
        return None


def agg(rows: list[dict]) -> dict:
    # one list per metric, then average each
    rag_faithfulness = []
    rag_answer_relevancy = []
    rag_context_precision = []
    rag_context_recall = []
    rag_em = []
    rag_f1 = []
    rag_contains = []
    baseline_answer_relevancy = []
    baseline_em = []
    baseline_f1 = []
    baseline_contains = []

    for r in rows:
        rag_faithfulness.append(r["rag_ragas"]["faithfulness"])
        rag_answer_relevancy.append(r["rag_ragas"]["answer_relevancy"])
        rag_context_precision.append(r["rag_ragas"]["context_precision"])
        rag_context_recall.append(r["rag_ragas"]["context_recall"])
        rag_em.append(r["rag_em"])
        rag_f1.append(r["rag_f1"])
        rag_contains.append(r["rag_contains"])
        baseline_answer_relevancy.append(r["baseline_ragas"]["answer_relevancy"])
        baseline_em.append(r["baseline_em"])
        baseline_f1.append(r["baseline_f1"])
        baseline_contains.append(r["baseline_contains"])

    return {
        "n": len(rows),
        "rag": {
            "faithfulness": _mean(rag_faithfulness),
            "answer_relevancy": _mean(rag_answer_relevancy),
            "context_precision": _mean(rag_context_precision),
            "context_recall": _mean(rag_context_recall),
            "em": _mean(rag_em),
            "f1": _mean(rag_f1),
            "contains": _mean(rag_contains),
        },
        "baseline": {
            "answer_relevancy": _mean(baseline_answer_relevancy),
            "em": _mean(baseline_em),
            "f1": _mean(baseline_f1),
            "contains": _mean(baseline_contains),
        },
    }


def summarise(records: list[dict]) -> dict:
    # overall first, then one block per question type
    summary = {"overall": agg(records), "by_type": {}}

    # distinct types, each kept once
    types = []
    for r in records:
        if r["type"] not in types:
            types.append(r["type"])
    types.sort()

    for qtype in types:
        rows_of_type = []
        for r in records:
            if r["type"] == qtype:
                rows_of_type.append(r)
        summary["by_type"][qtype] = agg(rows_of_type)

    return summary


def print_report(records: list[dict], summary: dict) -> None:
    print("\n" + "=" * 78)
    print("PER-QUESTION RESULTS")
    print("=" * 78)
    # "Has" = the contains check
    header = f"{'ID':<9}{'type':<10}{'Faith':>7}{'AnsRel':>8}{'CtxRel':>8}{'CtxRec':>8}{'R-EM':>6}{'R-F1':>6}{'R-Has':>7}{'B-EM':>6}{'B-F1':>6}{'B-Has':>7}"
    print(header)
    print("-" * len(header))
    for r in records:
        g = r["rag_ragas"]   # keeps the print line below readable
        print(
            f"{r['id']:<9}{r['type']:<10}"
            f"{_fmt(g['faithfulness']):>7}{_fmt(g['answer_relevancy']):>8}"
            f"{_fmt(g['context_precision']):>8}{_fmt(g['context_recall']):>8}"
            f"{r['rag_em']:>6.0f}{r['rag_f1']:>6.2f}{r['rag_contains']:>7.0f}"
            f"{r['baseline_em']:>6.0f}{r['baseline_f1']:>6.2f}{r['baseline_contains']:>7.0f}"
        )

    print("\n" + "=" * 78)
    print("AGGREGATES (RAG vs Baseline)")
    print("=" * 78)
    _print_agg("OVERALL", summary["overall"])
    for qtype, agg_result in summary["by_type"].items():
        _print_agg(f"type={qtype}", agg_result)


def _print_agg(label: str, agg: dict) -> None:
    rag, base = agg["rag"], agg["baseline"]
    print(f"\n{label}  (n={agg['n']})")
    print(f"  RAG      Faith={_fmt(rag['faithfulness'])}  AnsRel={_fmt(rag['answer_relevancy'])}"
          f"  CtxRel={_fmt(rag['context_precision'])}  CtxRec={_fmt(rag['context_recall'])}"
          f"  EM={_fmt(rag['em'])}  F1={_fmt(rag['f1'])}  Has={_fmt(rag['contains'])}")
    print(f"  Baseline AnsRel={_fmt(base['answer_relevancy'])}  EM={_fmt(base['em'])}  F1={_fmt(base['f1'])}"
          f"  Has={_fmt(base['contains'])}   (context metrics N/A - baseline has no retrieval)")


def _fmt(x) -> str:
    # dash when there's no score
    if x is None:
        return "  -  "
    else:
        return f"{x:.2f}"


# --------------------------------------------------------------------------- #
def main():
    with open(QA_FILE, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)
    print(f"Loaded {len(qa_pairs)} QA pairs from {QA_FILE}.\n")

    # ask -> grade -> average -> print
    records = run_systems(qa_pairs)
    run_ragas(records)
    summary = summarise(records)
    print_report(records, summary)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(f"{RESULTS_DIR}/apple_results.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    with open(f"{RESULTS_DIR}/apple_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved detailed results to {RESULTS_DIR}/apple_results.json")
    print(f"Saved aggregates to {RESULTS_DIR}/apple_summary.json")


if __name__ == "__main__":
    main()
