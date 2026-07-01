"""
llm_judge_eval.py
──────────────────────────────────────────────────────────────────────────
LLM-as-a-Judge evaluation agent for the AI Career Advisor project.

What this does
---------------
1. Pulls ground-truth (resume, job_description, ats_score, original_label)
   pairs from the HuggingFace dataset:
       0xnbk/resume-ats-score-v1-en
   (https://huggingface.co/datasets/0xnbk/resume-ats-score-v1-en)

2. For every sampled pair, re-uses the SAME scoring approach as
   job_finder.py (a Gemini call through LangChain) to produce a
   "predicted_score" / "predicted_label" — this stands in for what the
   app currently emits as `scores.job_relevance` / fit tier.

3. Feeds resume + job_description + prediction + ground truth to a
   SECOND, independent Gemini call that acts as the "judge": it decides
   whether the prediction is well justified relative to the ground
   truth and explains why (Agree / Partial / Disagree + reasoning
   quality 1-5 + short critique).

4. Aggregates everything (MAE, RMSE, Pearson correlation, tier
   accuracy, judge agreement rate, average reasoning quality) and
   writes ONE json file (default: evaluation_results.json) containing
   both the per-example results and the aggregate metrics.

Requirements (not yet in the project — add to requirements.txt)
-----------------------------------------------------------------
    datasets
    pandas

Usage
-----
    python llm_judge_eval.py --sample_size 30 --split validation \
        --output evaluation_results.json

Notes
-----
- Uses the exact same CHAT_MODEL / GEMINI_API_KEY plumbing as
  job_finder.py so it's consistent with the rest of the pipeline.
- Network access to huggingface.co is required to download the
  dataset (via the `datasets` library) — this script does not run
  inside this sandboxed session, it's meant to be run in your own
  project environment where GEMINI_API_KEY + internet are available.
- Calls are rate-limited with a small sleep + retry/back-off since the
  free Gemini tier throttles aggressively. Tune SLEEP_BETWEEN_CALLS /
  MAX_RETRIES for your quota.
"""

import os
import re
import json
import time
import argparse
from datetime import datetime, timezone
from statistics import mean

import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# ─────────────────────────────────────────────────────────────────────────
#  Setup — mirrors job_finder.py
# ─────────────────────────────────────────────────────────────────────────
load_dotenv()
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
CHAT_MODEL = "gemini-3.1-pro-preview"  # keep in sync with job_finder.py

chat_model = ChatGoogleGenerativeAI(
    google_api_key=GEMINI_API_KEY,
    model=CHAT_MODEL,
)

DATASET_ID = "0xnbk/resume-ats-score-v1-en"
SLEEP_BETWEEN_CALLS = 1.5   # seconds, be gentle with the API
MAX_RETRIES = 3


# ─────────────────────────────────────────────────────────────────────────
#  Prompts
# ─────────────────────────────────────────────────────────────────────────

PREDICT_PROMPT = PromptTemplate.from_template("""
You are an AI Career Advisor scoring how well a candidate resume fits a job description,
using an Application Tracking System (ATS)-style compatibility score.

Instructions:
- Return ONLY a valid JSON object. No markdown, no explanation, no ```json fences.
- "predicted_score" must be an integer between 0 and 100.
- "predicted_label" must be exactly one of: "No Fit", "Potential Fit", "Good Fit".
  (No Fit: score < 40, Potential Fit: 40-70, Good Fit: score > 70)

Resume:
{resume}

Job Description:
{job_description}

Return this schema exactly:
{{
    "predicted_score": integer,
    "predicted_label": string
}}
""")

JUDGE_PROMPT = PromptTemplate.from_template("""
You are an impartial evaluator (LLM-as-a-judge) auditing an AI Career Advisor's
resume-to-job fit assessment against a trusted ground-truth ATS score.

You are given:
- The resume and job description that were assessed.
- The ground-truth ATS score and fit label (treat this as correct).
- The model's predicted ATS score and fit label.

Your job:
- Decide whether the model's prediction is well justified given the ground truth.
- Do NOT simply say scores must be identical — allow for reasonable disagreement,
  but flag cases where the predicted label/tier or score is clearly unreasonable
  given the resume/job content or diverges heavily (>20 points) from ground truth
  without good reason.

Instructions:
- Return ONLY a valid JSON object. No markdown, no explanation, no ```json fences.
- "judge_verdict" must be exactly one of: "Agree", "Partial", "Disagree".
- "reasoning_quality" is an integer 1-5 rating of how well-justified the model's
  score/label is relative to the actual resume/job content (5 = excellent, 1 = poor).
- "explanation" is a short (1-3 sentence) justification for your verdict.

Ground Truth:
- ats_score: {gt_score}
- original_label: {gt_label}

Model Prediction:
- predicted_score: {pred_score}
- predicted_label: {pred_label}

Resume:
{resume}

Job Description:
{job_description}

Return this schema exactly:
{{
    "judge_verdict": string,
    "reasoning_quality": integer,
    "explanation": string
}}
""")


# ─────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────

def _extract_text(response) -> str:
    """Pull plain text out of a LangChain ChatGoogleGenerativeAI response,
    matching the parsing style already used in job_finder.py."""
    content = response.content
    if isinstance(content, str):
        return content
    return "".join(
        part["text"] for part in content
        if isinstance(part, dict) and part.get("type") == "text"
    )


def _call_json(prompt_template, variables, retries=MAX_RETRIES):
    """Invoke the chat model with a prompt, parse JSON, retry on failure."""
    chain = prompt_template | chat_model
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            response = chain.invoke(variables)
            raw = _extract_text(response).strip()
            raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()
            return json.loads(raw)
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(SLEEP_BETWEEN_CALLS * attempt)
    raise RuntimeError(f"Failed after {retries} retries: {last_err}")


def predict_fit(resume: str, job_description: str) -> dict:
    return _call_json(PREDICT_PROMPT, {
        "resume": resume,
        "job_description": job_description,
    })


def judge_prediction(resume, job_description, gt_score, gt_label, pred_score, pred_label) -> dict:
    return _call_json(JUDGE_PROMPT, {
        "resume": resume,
        "job_description": job_description,
        "gt_score": gt_score,
        "gt_label": gt_label,
        "pred_score": pred_score,
        "pred_label": pred_label,
    })


def parse_pair(text: str) -> tuple[str, str]:
    """Ground truth `text` column is formatted as `resume [SEP] job_description`."""
    for sep in ("[SEP]", " SEP ", "\nSEP\n"):
        if sep in text:
            resume, job_description = text.split(sep, 1)
            return resume.strip(), job_description.strip()
    # Fallback: no separator found, treat whole text as resume with no JD.
    return text.strip(), ""


def tier_from_score(score: float) -> str:
    if score < 40:
        return "No Fit"
    elif score <= 70:
        return "Potential Fit"
    return "Good Fit"


def pearson_corr(xs, ys) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = mean(xs), mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = sum((x - mx) ** 2 for x in xs) ** 0.5
    sy = sum((y - my) ** 2 for y in ys) ** 0.5
    if sx == 0 or sy == 0:
        return 0.0
    return cov / (sx * sy)


# ─────────────────────────────────────────────────────────────────────────
#  Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────

def load_ground_truth(split: str, sample_size: int, seed: int) -> pd.DataFrame:
    ds = load_dataset(DATASET_ID, split=split)
    df = ds.to_pandas()
    n = min(sample_size, len(df))
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def run_evaluation(sample_size: int, split: str, output_path: str, seed: int = 42):
    print(f"Loading {sample_size} ground-truth examples from '{DATASET_ID}' ({split} split)...")
    df = load_ground_truth(split=split, sample_size=sample_size, seed=seed)

    results = []
    for i, row in df.iterrows():
        resume, job_description = parse_pair(row["text"])
        gt_score = float(row["ats_score"])
        gt_label = str(row["original_label"])

        print(f"[{i+1}/{len(df)}] predicting...", end=" ")
        try:
            pred = predict_fit(resume, job_description)
            pred_score = int(pred["predicted_score"])
            pred_label = str(pred["predicted_label"])
        except Exception as e:  # noqa: BLE001
            print(f"PREDICTION FAILED: {e}")
            continue
        time.sleep(SLEEP_BETWEEN_CALLS)

        print("judging...", end=" ")
        try:
            judge = judge_prediction(
                resume, job_description, gt_score, gt_label, pred_score, pred_label
            )
        except Exception as e:  # noqa: BLE001
            print(f"JUDGE FAILED: {e}")
            continue
        time.sleep(SLEEP_BETWEEN_CALLS)

        abs_error = abs(pred_score - gt_score)
        tier_match = tier_from_score(gt_score) == pred_label

        results.append({
            "index": int(i),
            "ground_truth_score": gt_score,
            "ground_truth_label": gt_label,
            "predicted_score": pred_score,
            "predicted_label": pred_label,
            "absolute_error": round(abs_error, 2),
            "tier_match": tier_match,
            "judge_verdict": judge.get("judge_verdict"),
            "reasoning_quality": judge.get("reasoning_quality"),
            "judge_explanation": judge.get("explanation"),
        })
        print(f"done (gt={gt_score:.1f}, pred={pred_score}, verdict={judge.get('judge_verdict')})")

    if not results:
        raise RuntimeError("No examples were successfully evaluated — check API key / connectivity.")

    gt_scores = [r["ground_truth_score"] for r in results]
    pred_scores = [r["predicted_score"] for r in results]

    mae = mean(abs(p - g) for p, g in zip(pred_scores, gt_scores))
    rmse = mean((p - g) ** 2 for p, g in zip(pred_scores, gt_scores)) ** 0.5
    corr = pearson_corr(pred_scores, gt_scores)
    tier_accuracy = mean(1.0 if r["tier_match"] else 0.0 for r in results)
    judge_agreement_rate = mean(1.0 if r["judge_verdict"] == "Agree" else 0.0 for r in results)
    avg_reasoning_quality = mean(
        r["reasoning_quality"] for r in results if isinstance(r["reasoning_quality"], (int, float))
    )
    verdict_breakdown = {
        v: sum(1 for r in results if r["judge_verdict"] == v)
        for v in ("Agree", "Partial", "Disagree")
    }

    output = {
        "metadata": {
            "dataset": DATASET_ID,
            "dataset_split": split,
            "chat_model": CHAT_MODEL,
            "sample_size_requested": sample_size,
            "sample_size_evaluated": len(results),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "aggregate_metrics": {
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "pearson_correlation": round(corr, 3),
            "tier_accuracy": round(tier_accuracy, 3),
            "judge_agreement_rate": round(judge_agreement_rate, 3),
            "avg_reasoning_quality": round(avg_reasoning_quality, 2),
            "judge_verdict_breakdown": verdict_breakdown,
        },
        "per_example_results": results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=4)

    print("\n── Evaluation summary ──────────────────────────")
    print(f"  Examples evaluated : {len(results)}/{sample_size}")
    print(f"  MAE                : {mae:.2f}")
    print(f"  RMSE               : {rmse:.2f}")
    print(f"  Pearson corr.      : {corr:.3f}")
    print(f"  Tier accuracy      : {tier_accuracy:.1%}")
    print(f"  Judge agreement    : {judge_agreement_rate:.1%}")
    print(f"  Avg reasoning qual.: {avg_reasoning_quality:.2f}/5")
    print(f"  Verdict breakdown  : {verdict_breakdown}")
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-as-a-judge evaluation for AI Career Advisor scoring.")
    parser.add_argument("--sample_size", type=int, default=30, help="Number of examples to sample.")
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation"])
    parser.add_argument("--output", type=str, default="evaluation_results.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_evaluation(
        sample_size=args.sample_size,
        split=args.split,
        output_path=args.output,
        seed=args.seed,
    )
