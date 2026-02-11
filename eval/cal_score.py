#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

CATEGORIES = [
    "scientific_accuracy",
    "structural_correctness",
    "semantic_alignment",
]

def load_model_results(results_dir: str) -> Dict[str, Dict]:
    """
    Load all JSON files in directory.
    Each JSON file corresponds to one model.
    """
    results = {}
    path = Path(results_dir)

    for json_file in path.glob("*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            results[json_file.stem] = json.load(f)

    return results


def verdict_to_score(verdict: str) -> float:
    """
    Binary grading (paper-compatible special case):
    - Satisfied -> 1.0
    - Not Satisfied -> 0.0
    """
    v = verdict.lower().strip()
    if "satisfied" in v and "not" not in v:
        return 1.0
    return 0.0

def calculate_task_score(rubric_items: List[Dict]) -> Optional[float]:
    """
    Compute S_k for one task (all rubric items).
    """
    if not rubric_items:
        return None

    numerator = 0.0
    denominator = 0.0

    for item in rubric_items:
        weight = item.get("weight", 0)
        verdict = item.get("evaluation", {}).get("verdict", "Not Satisfied")
        m_i = verdict_to_score(verdict)

        numerator += weight * m_i
        if weight > 0:
            denominator += weight

    if denominator == 0:
        return None

    return numerator / denominator


def calculate_task_category_score(category_items: List[Dict]) -> Optional[float]:
    """
    Compute S_{k,c} for one task and one category.
    Same formula as S_k, but restricted to one category.
    """
    return calculate_task_score(category_items)

def score_single_model(model_data: Dict) -> Dict:
    """
    Compute:
    - task scores
    - overall score (macro average)
    - category scores (task-level macro average)
    """
    task_scores = []
    category_task_scores = defaultdict(list)

    for task_data in model_data.values():
        rubric_eval = task_data.get("rubric_eval", {})

        # --- Overall task score ---
        all_items = []
        for cat in CATEGORIES:
            all_items.extend(rubric_eval.get(cat, []))

        s_k = calculate_task_score(all_items)
        if s_k is not None:
            task_scores.append(s_k)

        # --- Category task scores ---
        for cat in CATEGORIES:
            cat_items = rubric_eval.get(cat, [])
            s_kc = calculate_task_category_score(cat_items)
            if s_kc is not None:
                category_task_scores[cat].append(s_kc)

    overall_score = (
        sum(task_scores) / len(task_scores)
        if task_scores else 0.0
    )

    category_scores = {}
    for cat in CATEGORIES:
        scores = category_task_scores.get(cat, [])
        category_scores[cat] = (
            sum(scores) / len(scores)
            if scores else 0.0
        )

    return {
        "overall": overall_score,
        "categories": category_scores,
        "num_tasks": len(task_scores),
    }

def print_score_table(all_model_scores: Dict[str, Dict]) -> None:
    """
    Pretty, aligned table printing (paper-ready).
    """

    headers = ["Model", "Overall"] + CATEGORIES

    # Prepare table rows
    rows = []
    for model, scores in all_model_scores.items():
        row = [
            model,
            f"{scores['overall']:.4f}",
        ]
        for cat in CATEGORIES:
            row.append(f"{scores['categories'][cat]:.4f}")
        rows.append(row)

    # Compute column widths (max of header / cell)
    col_widths = []
    for col_idx in range(len(headers)):
        max_width = max(
            len(headers[col_idx]),
            max(len(row[col_idx]) for row in rows)
        )
        col_widths.append(max_width)

    # Helpers
    def sep(char="-"):
        return "+".join(char * (w + 2) for w in col_widths)

    def format_row(row, align_left_first=True):
        formatted = []
        for i, (cell, width) in enumerate(zip(row, col_widths)):
            if i == 0 and align_left_first:
                formatted.append(f" {cell.ljust(width)} ")
            else:
                formatted.append(f" {cell.rjust(width)} ")
        return "|" + "|".join(formatted) + "|"

    # Print table
    print("\n" + sep("="))
    print(format_row(headers))
    print(sep("="))

    for row in rows:
        print(format_row(row))

    print(sep("="))

def main():
    results_dir = "./evaluation_results"

    model_results = load_model_results(results_dir)

    all_model_scores = {}
    for model_name, model_data in model_results.items():
        all_model_scores[model_name] = score_single_model(model_data)

    print_score_table(all_model_scores)


if __name__ == "__main__":
    main()
