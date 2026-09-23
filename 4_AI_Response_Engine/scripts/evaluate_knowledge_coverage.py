"""Evaluate source-grounded answer coverage across the complete ZENDS PDF catalog."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
for path in (
    ROOT / "3_RAG_Knowledge" / "code",
    ROOT / "4_AI_Response_Engine" / "code",
    ROOT / "5_Streamlit_Integration" / "code",
):
    sys.path.insert(0, str(path))

from document_loader import load_pdf_pages
from grounding import COUNTRIES, PRODUCT_NAMES, _product_facts, select_grounded_chunks
from llm import StaticAcknowledgementLLM
from response_engine import ZendsResponseEngine
from ui_helpers import OUT_OF_SCOPE_RESPONSE, apply_scope_guard, is_abstention


FEATURE_CASES = (
    ("What mobile services does ZENDS provide?", ("voice calling", "SMS", "5G mobile data")),
    ("What services are included with ZENDFiber Home?", ("fiber connectivity", "router and WiFi equipment", "24×7 technical support")),
    ("What does ZENDBiz Connect offer?", ("dedicated bandwidth", "MPLS connectivity", "priority troubleshooting")),
    ("What services does ZENDCloud provide?", ("virtual machines", "file storage", "backup and disaster recovery")),
    ("What IoT solutions does ZENDS provide?", ("sensor connectivity", "device management", "smart city integrations")),
)
POLICY_CASES = (
    ("How does ZENDS billing work?", ("monthly in advance", "consolidated invoices", "7 days")),
    ("What is the ZENDS refund policy?", ("7 days", "10%", "not refundable after activation")),
    ("How long is an enterprise contract?", ("minimum 12-month contract",)),
    ("Compare the ZENDS SLA for individual, business and enterprise customers.", ("98.5%", "99.5%", "99.9%")),
    ("Is ZENDS GDPR compliant?", ("GDPR compliant", "ISO 27001", "encrypted data at rest and in transit")),
    ("What is the fair usage limit?", ("1TB per month",)),
    ("What support tiers does ZENDS offer?", ("Standard", "Priority", "Enterprise Dedicated Support")),
    ("What discounts does ZENDS provide?", ("30%", "15%")),
)
MULTI_FACT_CASES = (
    ("What is the difference between Prepaid Basic and Prepaid Plus?", ("Prepaid Basic", "5GB", "Prepaid Plus", "20GB")),
    ("Compare ZENDFiber Home 100 Mbps and ZENDFiber Home 300 Mbps in India.", ("$18", "$15", "$30", "$27")),
    ("What are the individual and enterprise prices for Postpaid Gold in India?", ("Postpaid Gold", "$60", "$50")),
    ("What is Postpaid Platinum, how much does it cost in India, and what does it include?", ("unlimited data", "international calls", "$72", "$66")),
    ("What broadband plans does ZENDS offer in India and how much do they cost?", ("ZENDFiber Home 100 Mbps", "ZENDOffice Net 1G", "$18", "$90")),
    ("What support and discount options are available for enterprise customers?", ("Standard", "Enterprise Dedicated Support", "30%", "15%")),
)
OUT_OF_KNOWLEDGE_CASES = (
    "What is ZENDS customer care phone number?",
    "Who is the CEO of ZENDS?",
    "What is the weather in Salem?",
    "Write me a Python program.",
)
NUMBER_PATTERN = re.compile(r"\$?\d+(?:\.\d+)?(?:%|tb|gb|mbps|gbps)?", re.I)


def _numbers(text: str) -> set[str]:
    return {value.lower().lstrip("$") for value in NUMBER_PATTERN.findall(text)}


def _run_case(engine: ZendsResponseEngine, query: str, expected: tuple[str, ...], category: str) -> dict[str, Any]:
    raw = engine.respond(query)
    selected = select_grounded_chunks(query, str(raw["intent"]), raw["retrieved_context"])
    result = apply_scope_guard(query, raw)
    answer = str(result["recommended_response"])
    evidence_text = " ".join(str(chunk["text"]) for chunk in selected)
    if category == "product_pricing":
        retrieval_success = bool(selected) and all(value.lower() in evidence_text.lower() for value in expected[:2])
    elif category == "policy":
        retrieval_success = bool(selected) and any(chunk["metadata"].get("policy_category") for chunk in selected)
    else:
        retrieval_success = bool(selected) and all(value.lower() in evidence_text.lower() for value in expected)
    answer_success = (
        not result.get("abstention", False)
        and answer != OUT_OF_SCOPE_RESPONSE
        and not is_abstention(answer)
        and all(value.lower() in answer.lower() for value in expected)
    )
    unsupported_claim = not _numbers(answer).issubset(_numbers(evidence_text))
    return {
        "category": category,
        "query": query,
        "expected": list(expected),
        "retrieved_chunk_ids": [chunk["metadata"].get("chunk_id") for chunk in raw["retrieved_context"]],
        "selected_chunk_ids": [chunk["metadata"].get("chunk_id") for chunk in selected],
        "answer": answer,
        "retrieval_success": retrieval_success,
        "answer_success": answer_success,
        "abstained": bool(result.get("abstention", False)),
        "unsupported_claim": unsupported_claim,
    }


def main() -> None:
    import torch

    torch.set_num_threads(2)
    pdf_path = ROOT / "docs" / "ZENDS Communications.pdf"
    source_text = " ".join(page.text for page in load_pdf_pages(pdf_path))
    engine = ZendsResponseEngine.from_project_assets(llm=StaticAcknowledgementLLM())
    cases: list[dict[str, Any]] = []

    for product in PRODUCT_NAMES:
        facts = _product_facts(source_text, product)
        if not facts:
            raise RuntimeError(f"Could not extract authoritative PDF pricing for {product}.")
        for country in COUNTRIES:
            for customer_type in ("individual", "enterprise"):
                expected_price = facts[customer_type].get(country)
                if expected_price is None:
                    raise RuntimeError(f"Missing {customer_type} {country} source price for {product}.")
                query = f"What is the {customer_type} price of {product} in {country}?"
                expected = (product, f"${expected_price}", f"{customer_type} customers")
                cases.append(_run_case(engine, query, expected, "product_pricing"))

    for query, expected in FEATURE_CASES:
        cases.append(_run_case(engine, query, expected, "product_features"))
    for query, expected in POLICY_CASES:
        cases.append(_run_case(engine, query, expected, "policy"))
    for query, expected in MULTI_FACT_CASES:
        cases.append(_run_case(engine, query, expected, "multi_fact"))
    for query in OUT_OF_KNOWLEDGE_CASES:
        raw = engine.respond(query)
        result = apply_scope_guard(query, raw)
        answer = str(result["recommended_response"])
        cases.append({
            "category": "out_of_knowledge",
            "query": query,
            "expected": [],
            "retrieved_chunk_ids": [chunk["metadata"].get("chunk_id") for chunk in raw["retrieved_context"]],
            "selected_chunk_ids": [],
            "answer": answer,
            "retrieval_success": True,
            "answer_success": bool(result.get("abstention", False)) and is_abstention(answer),
            "abstained": bool(result.get("abstention", False)),
            "unsupported_claim": False,
        })

    positive = [case for case in cases if case["category"] != "out_of_knowledge"]
    summary = {
        "source": str(pdf_path.relative_to(ROOT)),
        "total_cases": len(cases),
        "pricing_cases": sum(case["category"] == "product_pricing" for case in cases),
        "feature_cases": sum(case["category"] == "product_features" for case in cases),
        "policy_cases": sum(case["category"] == "policy" for case in cases),
        "multi_fact_cases": sum(case["category"] == "multi_fact" for case in cases),
        "out_of_knowledge_cases": sum(case["category"] == "out_of_knowledge" for case in cases),
        "retrieval_success_rate": sum(case["retrieval_success"] for case in positive) / len(positive),
        "answer_success_rate": sum(case["answer_success"] for case in positive) / len(positive),
        "grounded_answer_rate": sum(case["answer_success"] and not case["unsupported_claim"] for case in positive) / len(positive),
        "positive_abstention_rate": sum(case["abstained"] for case in positive) / len(positive),
        "unsupported_claim_rate": sum(case["unsupported_claim"] for case in positive) / len(positive),
        "safe_unknown_rate": sum(case["answer_success"] for case in cases if case["category"] == "out_of_knowledge") / len(OUT_OF_KNOWLEDGE_CASES),
    }
    output = ROOT / "4_AI_Response_Engine" / "evaluation" / "knowledge_coverage_evaluation.json"
    output.write_text(json.dumps({"summary": summary, "cases": cases}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if summary["answer_success_rate"] < 1 or summary["unsupported_claim_rate"] > 0 or summary["safe_unknown_rate"] < 1:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
