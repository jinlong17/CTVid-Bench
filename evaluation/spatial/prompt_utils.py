#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Text-only prompt utilities for Spatial QA evaluation."""

import json
import re


def build_ocr_context(ocr_results: list) -> str:
    if not ocr_results:
        return "OCR detected text in the image:\n(no text detected)"
    lines = ["OCR detected text in the image:"]
    for item in ocr_results:
        content = item.get("content", "")
        bbox = item.get("bbox", [])
        if content:
            bbox_str = str(list(map(int, bbox))) if bbox else "[]"
            lines.append(f'- "{content}" at bbox {bbox_str}')
    return "\n".join(lines)


def build_system_prompt(question_type: str) -> str:
    base = (
        "You are a text-analysis assistant. "
        "You will be given a list of text strings extracted by OCR from an image. "
        "Your task is to answer the question based ONLY on the provided OCR text. "
        "Do NOT rely on common sense, world knowledge, or spelling patterns — "
        "answer strictly from what is present in the OCR output.\n\n"
        "Output Format: a single JSON object on one line:\n"
        '{"answer": "<your answer>", "reasoning": "<one sentence>"}\n\n'
        "Respond ONLY with a JSON object. Do not include anything else."
    )
    if question_type in ("fill_in_blank", "fill_in"):
        specific = (
            "For fill-in-the-blank questions, output ONLY the missing character(s). "
            "Do not output the full word."
        )
    elif question_type == "multiple_choice":
        specific = 'For multiple-choice questions, output ONLY the option letter (e.g. "A", "B", "C", or "D").'
    elif question_type == "true_false":
        specific = 'For true/false questions, output ONLY "True" or "False".'
    else:
        specific = "Answer the question concisely."
    return f"{base}\n{specific}"


def build_user_prompt(question_type: str, question: str, ocr_context: str) -> str:
    return (
        f"{ocr_context}\n\n"
        f"Question: {question}\n\n"
        "Answer based ONLY on the OCR text above. "
        'Respond with a JSON object: {"answer": "...", "reasoning": "..."}'
    )


def extract_json_answer(text: str) -> dict:
    try:
        text_clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
        match = re.search(r"\{[\s\S]*?\}", text_clean)
        if match:
            loaded = json.loads(match.group())
            if isinstance(loaded, dict):
                return loaded
        loaded = json.loads(text_clean.strip())
        if isinstance(loaded, list) and loaded and isinstance(loaded[0], dict):
            return loaded[0]
        if isinstance(loaded, dict):
            return loaded
    except Exception:
        pass
    answer_match = re.search(
        r'answer\s*[:：]\s*(.*?)(?=\s*(reasoning|$))', text, re.IGNORECASE | re.DOTALL
    )
    answer = answer_match.group(1).strip() if answer_match else ""
    return {"answer": answer + "__FAILED!@#", "reasoning": "Failed to parse"}
