"""Grounding must preserve relationships, not just source vocabulary and numbers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "4_AI_Response_Engine" / "code"))

from grounding import is_grounded_answer


PRICE_EVIDENCE = [
    {
        "text": (
            "Postpaid Gold with 100GB is priced at $100 in the USA, $60 in India, $110 in Singapore, "
            "and $80 in Thailand for individual users, and $80 in the USA, $50 in India, "
            "$90 in Singapore, and $70 in Thailand for enterprise customers. "
            "ZENDFiber Home 100 Mbps is priced at $30 in the USA, $18 in India, $33 in Singapore, "
            "and $24 in Thailand for individual users, and $25 in the USA, $15 in India, "
            "$27 in Singapore, and $20 in Thailand for enterprise customers. "
            "ZENDCloud VM Pro is priced at $80 in the USA, $48 in India, $88 in Singapore, "
            "and $64 in Thailand for individual users, and $70 in the USA, $42 in India, "
            "$77 in Singapore, and $56 in Thailand for enterprise customers."
        ),
        "metadata": {"chunk_id": "catalog-prices"},
        "distance": 0.1,
    }
]


@pytest.mark.parametrize(
    ("answer", "valid"),
    [
        ("Postpaid Gold costs $100 for individual customers in the USA.", True),
        ("Postpaid Gold costs $80 for enterprise customers in the USA.", True),
        ("Postpaid Gold costs $80 for individual customers in the USA.", False),
        ("Postpaid Gold costs $100 for enterprise customers in the USA.", False),
        ("In India, Postpaid Gold costs $60 for individual customers.", True),
        ("In India, Postpaid Gold costs $100 for individual customers.", False),
        ("In India, Postpaid Gold costs $50 for enterprise customers.", True),
        ("In India, Postpaid Gold costs $60 for enterprise customers.", False),
        ("ZENDFiber Home 100 Mbps costs $33 in Singapore for individual customers.", True),
        ("ZENDFiber Home 100 Mbps costs $27 in Singapore for individual customers.", False),
        ("ZENDFiber Home 100 Mbps costs $27 in Singapore for enterprise customers.", True),
        ("ZENDFiber Home 100 Mbps costs $33 in Singapore for enterprise customers.", False),
        ("ZENDCloud VM Pro costs $64 in Thailand for individual customers.", True),
        ("ZENDCloud VM Pro costs $56 in Thailand for individual customers.", False),
        ("ZENDCloud VM Pro costs $56 in Thailand for enterprise customers.", True),
        ("ZENDCloud VM Pro costs $64 in Thailand for enterprise customers.", False),
        ("Postpaid Gold costs $18 in India for individual customers.", False),
        ("ZENDFiber Home 100 Mbps costs $60 in India for individual customers.", False),
    ],
)
def test_product_country_customer_type_and_price_stay_together(answer: str, valid: bool) -> None:
    assert is_grounded_answer(answer, PRICE_EVIDENCE) is valid


SLA_EVIDENCE = [{
    "text": "SLA: Individual users have 98.5% uptime, business users 99.5% uptime, and enterprise users 99.9% uptime.",
    "metadata": {"policy_category": "SLA"},
    "distance": 0.1,
}]


@pytest.mark.parametrize(
    ("answer", "valid"),
    [
        ("The individual SLA is 98.5%.", True),
        ("The individual SLA is 99.9%.", False),
        ("Business users have 99.5% uptime.", True),
        ("Business users have 98.5% uptime.", False),
        ("Enterprise users have 99.9% uptime.", True),
        ("Enterprise users have 98.5% uptime.", False),
    ],
)
def test_sla_percentage_stays_with_its_customer_tier(answer: str, valid: bool) -> None:
    assert is_grounded_answer(answer, SLA_EVIDENCE) is valid


POLICY_EVIDENCE = [
    {"text": "Refund: Full refund within 7 days if usage is less than 10%.", "metadata": {"policy_category": "Refund"}, "distance": 0.1},
    {"text": "Discounts: Bulk enterprise users can receive up to 30% discount. Annual payment discount is 15%.", "metadata": {"policy_category": "Discounts"}, "distance": 0.1},
    {"text": "Contracts: Individual users have no long-term lock-in. Enterprise customers have a minimum 12-month contract.", "metadata": {"policy_category": "Contracts"}, "distance": 0.1},
]


@pytest.mark.parametrize(
    ("answer", "valid"),
    [
        ("A full refund is available within 7 days when usage is below 10%.", True),
        ("A full refund is available when usage is below 15%.", False),
        ("The annual payment discount is 15%.", True),
        ("The annual payment discount is 30%.", False),
        ("Bulk enterprise users can receive a 30% discount.", True),
        ("Bulk enterprise users can receive a 15% discount.", False),
        ("Enterprise customers have a minimum 12-month contract.", True),
        ("Individual customers have a minimum 12-month contract.", False),
    ],
)
def test_policy_values_stay_with_their_category_and_tier(answer: str, valid: bool) -> None:
    assert is_grounded_answer(answer, POLICY_EVIDENCE) is valid
