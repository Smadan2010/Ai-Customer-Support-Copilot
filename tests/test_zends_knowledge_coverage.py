"""End-to-end coverage for source-grounded ZENDS product and policy answers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "4_AI_Response_Engine" / "code", ROOT / "5_Streamlit_Integration" / "code"):
    sys.path.insert(0, str(path))

from llm import StaticAcknowledgementLLM
from response_engine import ZendsResponseEngine
from ui_helpers import OUT_OF_SCOPE_RESPONSE, apply_scope_guard, is_abstention


@pytest.fixture(scope="module")
def engine() -> ZendsResponseEngine:
    return ZendsResponseEngine.from_project_assets(llm=StaticAcknowledgementLLM())


def ask(engine: ZendsResponseEngine, query: str) -> dict:
    return apply_scope_guard(query, engine.respond(query))


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("How much does Prepaid Basic cost in India?", ("Prepaid Basic", "5GB", "$36", "$28")),
        ("How much does Prepaid Plus cost in India?", ("Prepaid Plus", "20GB", "$48", "$29")),
        ("How much does Prepaid Unlimited cost in India?", ("Prepaid Unlimited", "$60", "$50")),
        ("What is the price of Postpaid Silver in the USA?", ("Postpaid Silver", "50GB", "$70", "$60")),
        ("What is the price of Postpaid Gold in the USA?", ("Postpaid Gold", "100GB", "$100", "$80")),
        ("How much does Postpaid Platinum cost in India?", ("Postpaid Platinum", "unlimited data", "$72", "$66")),
        ("How much does ZENDFiber Home 100 Mbps cost in India?", ("ZENDFiber Home 100 Mbps", "$18", "$15")),
        ("How much does ZENDFiber Home 300 Mbps cost in India?", ("ZENDFiber Home 300 Mbps", "$30", "$27")),
        ("How much does ZENDFiber Home 1 Gbps cost in India?", ("ZENDFiber Home 1 Gbps", "$48", "$42")),
        ("How much does ZENDCloud VM Basic cost in India?", ("ZENDCloud VM Basic", "$24", "$21")),
        ("How much does ZENDStorage 1TB cost in India?", ("ZENDStorage 1TB", "$12", "$9")),
        ("How much does ZENDSmart Parking cost in India?", ("ZENDSmart Parking", "$36", "$30")),
    ],
)
def test_required_product_prices_are_specific_and_grounded(
    engine: ZendsResponseEngine, query: str, expected: tuple[str, ...]
) -> None:
    result = ask(engine, query)
    answer = result["recommended_response"]
    assert result["abstention"] is False
    assert not is_abstention(answer)
    assert answer != OUT_OF_SCOPE_RESPONSE
    assert all(value in answer for value in expected)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("How does ZENDS billing work?", ("monthly in advance", "consolidated invoices", "7 days")),
        ("What is the ZENDS refund policy?", ("7 days", "10%", "not refundable after activation")),
        ("How long is an enterprise contract?", ("no long-term lock-in", "minimum 12-month contract")),
        ("Compare the ZENDS SLA for individual, business and enterprise customers.", ("98.5%", "99.5%", "99.9%")),
        ("Is ZENDS GDPR compliant?", ("GDPR compliant", "ISO 27001", "encrypted data at rest and in transit")),
        ("What is the fair usage limit?", ("1TB per month",)),
        ("What support tiers does ZENDS offer?", ("Standard", "Priority", "Enterprise Dedicated Support")),
        ("What discounts does ZENDS provide?", ("30%", "15%")),
    ],
)
def test_required_policies_preserve_all_conditions(
    engine: ZendsResponseEngine, query: str, expected: tuple[str, ...]
) -> None:
    result = ask(engine, query)
    answer = result["recommended_response"]
    assert result["abstention"] is False
    assert all(value in answer for value in expected)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("What does Postpaid Platinum include?", ("unlimited data", "international calls")),
        ("What services are included with ZENDFiber Home?", ("fiber connectivity", "router and WiFi equipment", "24×7 technical support")),
        ("What does ZENDBiz Connect offer?", ("dedicated bandwidth", "MPLS connectivity", "priority troubleshooting")),
        ("What services does ZENDCloud provide?", ("virtual machines", "file storage", "backup and disaster recovery")),
        ("What IoT solutions does ZENDS provide?", ("sensor connectivity", "device management", "smart city integrations")),
    ],
)
def test_required_capability_answers_use_the_correct_product_group(
    engine: ZendsResponseEngine, query: str, expected: tuple[str, ...]
) -> None:
    result = ask(engine, query)
    answer = result["recommended_response"]
    assert result["abstention"] is False
    assert all(value in answer for value in expected)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("What is the difference between Prepaid Basic and Prepaid Plus?", ("Prepaid Basic", "5GB", "Prepaid Plus", "20GB")),
        ("Compare ZENDFiber Home 100 Mbps and ZENDFiber Home 300 Mbps in India.", ("$18", "$15", "$30", "$27")),
        ("What are the individual and enterprise prices for Postpaid Gold in India?", ("Postpaid Gold", "$60", "$50")),
        ("What is Postpaid Platinum, how much does it cost in India, and what does it include?", ("unlimited data", "international calls", "$72", "$66")),
        ("What broadband plans does ZENDS offer in India and how much do they cost?", ("ZENDFiber Home 100 Mbps", "ZENDOffice Net 1G", "$18", "$90")),
        ("What support and discount options are available for enterprise customers?", ("Standard", "Enterprise Dedicated Support", "30%", "15%")),
    ],
)
def test_comparison_and_multi_fact_questions_combine_only_required_evidence(
    engine: ZendsResponseEngine, query: str, expected: tuple[str, ...]
) -> None:
    result = ask(engine, query)
    assert result["abstention"] is False
    assert all(value in result["recommended_response"] for value in expected)


@pytest.mark.parametrize(
    "query",
    [
        "What is ZENDS customer care phone number?",
        "Who is the CEO of ZENDS?",
        "What is the weather in Salem?",
        "Write me a Python program.",
    ],
)
def test_unknown_or_unrelated_questions_never_hallucinate(engine: ZendsResponseEngine, query: str) -> None:
    result = ask(engine, query)
    assert result["abstention"] is True
    assert is_abstention(result["recommended_response"])
