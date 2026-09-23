"""Conservative evidence selection and response composition for Segment 4."""

from __future__ import annotations

import re
from typing import Any


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a", "an", "and", "are", "can", "do", "does", "for", "from", "how", "i", "in", "is", "it", "me", "my",
    "of", "on", "please", "the", "to", "what", "when", "where", "with", "you", "your", "zends", "customer",
}
GENERIC_TERMS = {
    "customer", "enterprise", "individual", "india", "price", "prices", "pricing", "service", "services",
    "support", "usa", "user", "users",
}
POLICY_INTENT = {"Billing": "Billing", "Refund": "Refund"}
POLICY_QUERY_TERMS = {
    "contract": "Contracts",
    "contracts": "Contracts",
    "discount": "Discounts",
    "discounts": "Discounts",
    "fair usage": "Fair Usage",
    "gdpr": "Data Privacy",
    "encrypted": "Data Privacy",
    "encryption": "Data Privacy",
    "privacy": "Data Privacy",
    "sla": "SLA",
    "support tiers": "Support Tiers",
}
SUPPORT_INTENTS = {"Technical", "Complaint"}
TECHNICAL_QUERY_TERMS = {
    "blocking", "cannot", "connect", "connection", "down", "error", "issue", "network", "outage", "problem",
    "problems", "setup", "trouble", "troubleshooting", "wifi", "internet", "dropping", "disconnecting", "fail",
    "failing", "working", "slow", "fiber", "installation",
}
SERVICE_QUERY_TERMS = {
    "broadband", "connection", "connectivity", "fiber", "internet", "network", "service", "wifi",
}
COMPLAINT_QUERY_TERMS = {"complain", "complaint", "disappointed", "escalate", "escalation", "unhappy", "unresolved"}
SUPPORT_EVIDENCE_TERMS = (
    "technical support", "troubleshooting", "network monitoring", "setup guidance", "installation", "fiber connectivity",
)
PRODUCT_GROUPS = {
    "Mobile Connectivity": {
        "aliases": ("mobile connectivity", "mobile services", "mobile plans", "prepaid", "postpaid", "mobile"),
        "evidence": ("mobile connectivity", "prepaid basic", "postpaid silver", "5g mobile data", "sim and esim"),
        "subject": "mobile connectivity services",
        "products": (
            "Prepaid Basic", "Prepaid Plus", "Prepaid Unlimited", "Postpaid Silver", "Postpaid Gold",
            "Postpaid Platinum",
        ),
    },
    "Home & Office Internet": {
        "aliases": ("home and office internet", "home office internet", "home broadband", "office internet", "zendfiber", "broadband", "internet"),
        "evidence": ("home office internet", "home broadband", "zendfiber", "fiber connectivity", "router and wifi"),
        "subject": "home and office internet services",
        "products": (
            "ZENDFiber Home 100 Mbps", "ZENDFiber Home 300 Mbps", "ZENDFiber Home 1 Gbps",
            "ZENDOffice Net 200", "ZENDOffice Net 500", "ZENDOffice Net 1G",
        ),
    },
    "Business Connectivity": {
        "aliases": ("business connectivity", "enterprise connectivity", "business internet", "zendbiz", "zendenterprise"),
        "evidence": ("business connectivity", "zendbiz", "zendenterprise", "dedicated bandwidth", "mpls connectivity"),
        "subject": "business connectivity services",
        "products": (
            "ZENDBiz Connect 100", "ZENDBiz Connect 500", "ZENDBiz Connect 1G",
            "ZENDEnterprise Ultra", "ZENDEnterprise Dedicated",
        ),
    },
    "Cloud & Data Center Services": {
        "aliases": ("cloud and data center services", "cloud data center", "cloud services", "cloud offerings", "cloud solutions", "zendcloud", "zendstorage", "zendarchive", "cloud"),
        "evidence": ("cloud and data center services", "zendcloud", "virtual machines", "file storage", "cloud networking", "cloud migration"),
        "subject": "cloud and data center services",
        "products": (
            "ZENDCloud VM Basic", "ZENDCloud VM Pro", "ZENDCloud VM Enterprise", "ZENDStorage 1TB",
            "ZENDStorage 10TB", "ZENDArchive Storage",
        ),
    },
    "IoT & Smart Solutions": {
        "aliases": ("iot and smart solutions", "iot smart solutions", "iot services", "smart solutions", "zendsmart", "zendindustrial", "zendfleet", "iot"),
        "evidence": ("iot and smart solutions", "zendsmart", "sensor connectivity", "device management", "smart city integrations"),
        "subject": "IoT and smart solutions",
        "products": (
            "ZENDSmart Traffic", "ZENDSmart Lighting", "ZENDSmart Parking", "ZENDIndustrial Sensor",
            "ZENDFleet IoT",
        ),
    },
}
PRODUCT_NAMES = tuple(
    product
    for configuration in PRODUCT_GROUPS.values()
    for product in configuration["products"]
)
PRODUCT_GROUP_BY_NAME = {
    product: group
    for group, configuration in PRODUCT_GROUPS.items()
    for product in configuration["products"]
}
COUNTRIES = ("USA", "India", "Singapore", "Thailand")
PRICE_QUERY_TERMS = {"cost", "costs", "how much", "price", "priced", "prices", "pricing"}
FEATURE_QUERY_TERMS = {"feature", "features", "include", "included", "includes", "offer", "offers", "provide", "provides", "service", "services"}
FACTUAL_RISK_TERMS = {
    "billing", "contract", "discount", "encryption", "gdpr", "price", "pricing", "refund", "service", "sla", "support",
    "tier", "zends",
}
INTERNAL_TERMS = {"chunk", "context", "document", "embedding", "rag", "retriev"}


def _terms(text: str) -> set[str]:
    return {term for term in TOKEN_PATTERN.findall(text.lower()) if term not in STOPWORDS}


def _meaningful_terms(text: str) -> set[str]:
    """Exclude generic telecom/pricing words from evidence relevance decisions."""
    return _terms(text) - GENERIC_TERMS


def _specific_phrases(query: str) -> set[str]:
    """Extract deterministic two- and three-token product/service phrases from the query."""
    tokens = TOKEN_PATTERN.findall(query.lower())
    phrases: set[str] = set()
    for length in (3, 2):
        for index in range(len(tokens) - length + 1):
            phrase_tokens = tokens[index:index + length]
            if any(token in STOPWORDS or token in GENERIC_TERMS for token in phrase_tokens):
                continue
            if any(token.startswith("zend") for token in phrase_tokens) or all(token.isalpha() or token.isdigit() for token in phrase_tokens):
                phrases.add(" ".join(phrase_tokens))
    return phrases


def _normalized_phrase_text(text: str) -> str:
    """Normalize punctuation variants such as '&' for deterministic alias matching."""
    return " ".join(TOKEN_PATTERN.findall(text.lower()))


def _contains_normalized_phrase(text: str, phrase: str) -> bool:
    return bool(re.search(rf"(?<!\w){re.escape(_normalized_phrase_text(phrase))}(?!\w)", _normalized_phrase_text(text)))


def _query_products(query: str) -> list[str]:
    """Return every exact source product named in the query, longest first."""
    return [product for product in PRODUCT_NAMES if _contains_normalized_phrase(query, product)]


def _products_in_text(text: str, *, group: str | None = None) -> list[str]:
    candidates = PRODUCT_GROUPS[group]["products"] if group else PRODUCT_NAMES
    return [product for product in candidates if _contains_normalized_phrase(text, product)]


def _is_price_question(query: str) -> bool:
    normalized = _normalized_phrase_text(query)
    return any(_contains_normalized_phrase(normalized, term) for term in PRICE_QUERY_TERMS)


def _is_feature_question(query: str) -> bool:
    normalized = _normalized_phrase_text(query)
    return any(_contains_normalized_phrase(normalized, term) for term in FEATURE_QUERY_TERMS)


def _is_comparison_question(query: str) -> bool:
    normalized = _normalized_phrase_text(query)
    return any(_contains_normalized_phrase(normalized, term) for term in ("compare", "comparison", "difference"))


def is_product_knowledge_query(query: str) -> bool:
    """Identify entity/group product questions independently of intent output."""
    return bool(_query_products(query) or _query_product_group(query)) and (
        _is_price_question(query) or _is_feature_question(query) or _is_comparison_question(query)
    )


def _chunk_position(chunk: dict[str, Any]) -> tuple[int, int] | None:
    matched = re.fullmatch(r"zends-p(\d+)-c(\d+)", str(chunk.get("metadata", {}).get("chunk_id", "")))
    return (int(matched.group(1)), int(matched.group(2))) if matched else None


def _query_product_group(query: str) -> str | None:
    normalized = _normalized_phrase_text(query)
    for group, configuration in PRODUCT_GROUPS.items():
        aliases = sorted(configuration["aliases"], key=len, reverse=True)
        if any(re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", normalized) for alias in aliases):
            return group
    return None


def _product_group_evidence_strength(document: str, group: str | None) -> int:
    if group is None:
        return 0
    normalized = _normalized_phrase_text(document)
    markers = PRODUCT_GROUPS[group]["evidence"]
    return sum(bool(re.search(rf"(?<!\w){re.escape(marker)}(?!\w)", normalized)) for marker in markers)


def _capability_sentence(document: str) -> str:
    matched = re.search(r"(?:^|(?<=[.!?])\s+)(Services include\s+[^.!?]+[.!?])", document, re.I)
    return matched.group(1) if matched else ""


def _policy_category(query: str, intent: str) -> str | None:
    """Resolve an explicit source policy heading from the query before intent fallback."""
    categories = _policy_categories(query, intent)
    return categories[0] if categories else None


def _policy_categories(query: str, intent: str) -> list[str]:
    """Resolve every explicitly requested policy while retaining intent fallback."""
    normalized = " ".join(query.lower().split())
    categories: list[str] = []
    for term, category in POLICY_QUERY_TERMS.items():
        if term in normalized and category not in categories:
            categories.append(category)
    if "support" in normalized and any(term in normalized for term in ("available", "option", "tier")):
        categories.append("Support Tiers")
    fallback = POLICY_INTENT.get(intent)
    if not categories and fallback:
        categories.append(fallback)
    return categories


def expected_policy_category(query: str, intent: str) -> str | None:
    """Expose policy routing without exposing evidence-selection internals."""
    return _policy_category(query, intent)


def select_grounded_chunks(query: str, intent: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select a small, entity-focused evidence set from retrieved source chunks."""
    query_terms = _meaningful_terms(query)
    specific_phrases = _specific_phrases(query)
    named_terms = {term for term in query_terms if term.startswith("zend") and len(term) > 4}
    expected_policies = _policy_categories(query, intent)
    raw_query_terms = set(TOKEN_PATTERN.findall(query.lower()))
    service_query_match = bool(raw_query_terms & SERVICE_QUERY_TERMS) or any(term.startswith("zend") for term in raw_query_terms)
    operational_query_match = bool(raw_query_terms & TECHNICAL_QUERY_TERMS)
    complaint_query_match = intent == "Complaint" and bool(raw_query_terms & COMPLAINT_QUERY_TERMS)
    support_query_match = service_query_match and (operational_query_match or complaint_query_match)
    query_product_group = _query_product_group(query)
    query_products = _query_products(query)

    # Explicit policy headings are authoritative and may be combined for a
    # genuine multi-policy question such as support tiers plus discounts.
    if expected_policies:
        policy_chunks = [chunk for chunk in chunks if chunk["metadata"].get("policy_category") in expected_policies]
        policy_chunks.sort(
            key=lambda chunk: (
                expected_policies.index(str(chunk["metadata"].get("policy_category"))),
                chunk["distance"],
                chunk["metadata"]["chunk_id"],
            )
        )
        if policy_chunks:
            return policy_chunks[:4]

    # Exact product entities are stronger than generic semantic similarity.
    # Keep one source chunk per requested product and, for pricing, its adjacent
    # continuation when the PDF's page-local chunk boundary split a price row.
    if query_products:
        selected: list[dict[str, Any]] = []
        for product in query_products:
            matches = [chunk for chunk in chunks if _contains_normalized_phrase(str(chunk["text"]), product)]
            matches.sort(key=lambda chunk: (chunk["distance"], chunk["metadata"]["chunk_id"]))
            if matches and matches[0] not in selected:
                selected.append(matches[0])
        if _is_price_question(query) or (_is_comparison_question(query) and _query_country(query)):
            positions = {_chunk_position(chunk): chunk for chunk in chunks if _chunk_position(chunk) is not None}
            for chunk in list(selected):
                position = _chunk_position(chunk)
                if position and (position[0], position[1] + 1) in positions:
                    continuation = positions[(position[0], position[1] + 1)]
                    if continuation not in selected:
                        selected.append(continuation)
        if _is_feature_question(query) and query_product_group:
            capabilities = [
                chunk for chunk in chunks
                if _product_group_evidence_strength(_capability_sentence(str(chunk["text"])), query_product_group) >= 2
            ]
            capabilities.sort(key=lambda chunk: (chunk["distance"], chunk["metadata"]["chunk_id"]))
            if capabilities and capabilities[0] not in selected:
                selected.append(capabilities[0])
        selected.sort(key=lambda chunk: (_chunk_position(chunk) or (999, 999), chunk["distance"]))
        if selected:
            return selected[:4]

    # Group-wide pricing questions need every retrieved pricing chunk for that
    # group; capability questions need only the group's source service sentence.
    if query_product_group:
        if _is_price_question(query):
            pricing_chunks = [
                chunk for chunk in chunks
                if "priced at" in str(chunk["text"]).lower()
                and _products_in_text(str(chunk["text"]), group=query_product_group)
            ]
            pricing_chunks.sort(key=lambda chunk: (_chunk_position(chunk) or (999, 999), chunk["distance"]))
            if pricing_chunks:
                return pricing_chunks[:4]
        if _is_feature_question(query):
            capabilities = [
                chunk for chunk in chunks
                if _product_group_evidence_strength(_capability_sentence(str(chunk["text"])), query_product_group) >= 2
            ]
            capabilities.sort(key=lambda chunk: (-_product_group_evidence_strength(_capability_sentence(str(chunk["text"])), query_product_group), chunk["distance"]))
            if capabilities:
                return capabilities[:1]

    scored: list[tuple[int, bool, bool, dict[str, Any]]] = []
    for chunk in chunks:
        document_text = str(chunk["text"])
        document_lower = document_text.lower()
        document_terms = _meaningful_terms(document_text)
        overlap = query_terms & document_terms
        named_match = bool(named_terms & document_terms)
        phrase_matches = [phrase for phrase in specific_phrases if phrase in document_lower]
        support_match = (
            intent in SUPPORT_INTENTS
            and support_query_match
            and any(phrase in document_lower for phrase in SUPPORT_EVIDENCE_TERMS)
        )
        product_group_strength = _product_group_evidence_strength(document_text, query_product_group)
        product_group_match = intent == "Product Inquiry" and product_group_strength >= 2
        product_match = bool(phrase_matches) or named_match
        score = (30 * len(phrase_matches)) + (20 if named_match else 0) + (12 if support_match else 0) + (4 if support_match and support_query_match else 0) + (10 * min(product_group_strength, 3)) + len(overlap)
        # Exact product/service phrases, named ZENDS terms, and technical
        # support evidence are meaningful on their own. Other evidence needs
        # two non-generic query terms to qualify.
        if product_match or support_match or product_group_match or len(overlap) >= 2:
            scored.append((score, product_match, support_match, chunk))
    scored.sort(key=lambda item: (-item[0], item[3]["distance"], item[3]["metadata"]["chunk_id"]))
    if not scored:
        return []
    # A second chunk must add a distinct relevant signal. This preserves a
    # product chunk plus a technical-support chunk, while rejecting broad
    # product/pricing context that merely repeats the same product phrase.
    _, first_product, first_support, first_chunk = scored[0]
    selected = [first_chunk]
    for _, product_match, support_match, chunk in scored[1:]:
        if (first_product and support_match and not first_support) or (first_support and product_match and not first_product):
            selected.append(chunk)
            break
    return selected


def safe_acknowledgement(model_text: str, sentiment: str) -> str:
    """Use an LLM tone sentence only when it contains no factual-risk language."""
    candidate = " ".join(str(model_text).replace("\n", " ").split())
    candidate_terms = _terms(candidate)
    factual_risk = any(term == risk or term.startswith(risk) for term in candidate_terms for risk in FACTUAL_RISK_TERMS)
    safe_model_phrases = {
        "i understand your concern.",
        "i understand your question.",
        "thank you for reaching out.",
        "thank you for your question.",
        "i am sorry this has been frustrating.",
    }
    if candidate.lower() in safe_model_phrases and not factual_risk:
        return candidate
    if sentiment == "Angry":
        return "I am sorry that this has been frustrating."
    if sentiment == "Happy":
        return "Thank you for reaching out."
    return "I understand your question."


def _evidence_text(evidence: list[dict[str, Any]]) -> str:
    return " ".join(str(chunk["text"]) for chunk in evidence)


def _source_sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]


def _single_policy_answer(category: str, text: str) -> str | None:
    if category == "Billing":
        monthly = re.search(r"Monthly billing in advance", text, re.I)
        invoices = re.search(r"Enterprise customers receive consolidated invoices", text, re.I)
        late = re.search(r"Late payment after (\d+) days may suspend services", text, re.I)
        parts = []
        if monthly:
            parts.append("ZENDS bills customers monthly in advance.")
        if invoices:
            parts.append("Enterprise customers receive consolidated invoices.")
        if late:
            parts.append(f"Payments overdue by more than {late.group(1)} days may lead to service suspension.")
        return " ".join(parts) or None
    if category == "Refund":
        refund = re.search(r"Full refund within (\d+) days if usage is less than (\d+)%", text, re.I)
        cloud = re.search(r"Cloud services are non-refundable after activation", text, re.I)
        parts = []
        if refund:
            parts.append(f"ZENDS offers a full refund within {refund.group(1)} days when usage is below {refund.group(2)}%.")
        if cloud:
            parts.append("Cloud services are not refundable after activation.")
        return " ".join(parts) or None
    if category in {"SLA", "Data Privacy", "Contracts", "Fair Usage", "Support Tiers", "Discounts"}:
        sentences = _source_sentences(re.sub(rf"^{re.escape(str(category))}:\s*", "", text, flags=re.I))
        return " ".join(sentences[:2]) or None
    return None


def _policy_answer(evidence: list[dict[str, Any]]) -> str | None:
    """Create short source-derived wording for one or more explicit policies."""
    answers = []
    for chunk in evidence:
        category = chunk["metadata"].get("policy_category")
        if category:
            answer = _single_policy_answer(str(category), str(chunk["text"]))
            if answer and answer not in answers:
                answers.append(answer)
    return " ".join(answers) or None


def _product_block(text: str, product: str) -> str | None:
    match = re.search(re.escape(product), text, re.I)
    if not match:
        return None
    end = len(text)
    for other in PRODUCT_NAMES:
        if other == product:
            continue
        following = re.search(re.escape(other), text[match.end():], re.I)
        if following:
            end = min(end, match.end() + following.start())
    return text[match.start():end].strip()


def _product_feature(block: str, product: str) -> str | None:
    priced = re.search(r"\bpriced at\b", block, re.I)
    if not priced:
        return None
    prefix = block[len(product):priced.start()].strip(" ,")
    prefix = re.sub(r"^with\s+", "", prefix, flags=re.I)
    prefix = re.sub(r"\bis\s*$", "", prefix, flags=re.I).strip(" ,")
    return prefix or None


def _country_prices(text: str) -> dict[str, str]:
    prices: dict[str, str] = {}
    pattern = re.compile(r"\$(\d+(?:\.\d+)?)\s+in\s+(?:the\s+)?(USA|India|Singapore|Thailand)", re.I)
    for value, country in pattern.findall(text):
        canonical = next(item for item in COUNTRIES if item.lower() == country.lower())
        prices[canonical] = value
    return prices


def _product_facts(text: str, product: str) -> dict[str, Any] | None:
    block = _product_block(text, product)
    if not block:
        return None
    pricing = re.search(
        r"priced at\s+(.*?)\s+for\s+individual\s+users,\s+and\s+(.*?)\s+for\s+enterprise\s+customers",
        block,
        re.I | re.S,
    )
    return {
        "product": product,
        "feature": _product_feature(block, product),
        "individual": _country_prices(pricing.group(1)) if pricing else {},
        "enterprise": _country_prices(pricing.group(2)) if pricing else {},
    }


def _customer_types(query: str) -> tuple[str, ...]:
    normalized = _normalized_phrase_text(query)
    requested = []
    if any(term in normalized for term in ("individual", "personal", "consumer")):
        requested.append("individual")
    if any(term in normalized for term in ("enterprise", "business customer", "business user")):
        requested.append("enterprise")
    return tuple(requested) or ("individual", "enterprise")


def _query_country(query: str) -> str | None:
    return next((country for country in COUNTRIES if _contains_normalized_phrase(query, country)), None)


def _feature_sentence(product: str, feature: str | None) -> str | None:
    if not feature:
        return None
    if re.fullmatch(r"\d+GB", feature, re.I):
        return f"{product} provides {feature} of data."
    return f"{product} includes {feature}."


def _product_price_answer(query: str, text: str) -> str | None:
    query_products = _query_products(query)
    group = _query_product_group(query)
    products = query_products or _products_in_text(text, group=group)
    country = _query_country(query)
    customer_types = _customer_types(query)
    answers: list[str] = []
    for product in products:
        facts = _product_facts(text, product)
        if not facts:
            continue
        feature_sentence = _feature_sentence(product, facts["feature"])
        if feature_sentence:
            answers.append(feature_sentence)
        if country:
            price_parts = [
                f"${facts[customer_type][country]} for {customer_type} customers"
                for customer_type in customer_types
                if country in facts[customer_type]
            ]
            if price_parts:
                answers.append(f"In {country}, {product} costs " + " and ".join(price_parts) + ".")
        else:
            for customer_type in customer_types:
                country_prices = facts[customer_type]
                if country_prices:
                    listing = ", ".join(f"${country_prices[item]} in {item}" for item in COUNTRIES if item in country_prices)
                    answers.append(f"For {customer_type} customers, {product} costs {listing}.")
    return " ".join(answers) or None


def _product_answer(query: str, evidence: list[dict[str, Any]]) -> str | None:
    """Summarize source-listed plans or product-group capabilities."""
    text = _evidence_text(evidence)
    normalized = query.lower()
    query_products = _query_products(query)
    if _is_price_question(query) or (_is_comparison_question(query) and _query_country(query)):
        price_answer = _product_price_answer(query, text)
        if price_answer:
            return price_answer
    if query_products and _is_feature_question(query):
        feature_answers = []
        for product in query_products:
            facts = _product_facts(text, product)
            answer = _feature_sentence(product, facts["feature"] if facts else None)
            if answer:
                feature_answers.append(answer)
        if feature_answers:
            return " ".join(feature_answers)
    if query_products and _is_comparison_question(query):
        feature_answers = []
        for product in query_products:
            facts = _product_facts(text, product)
            answer = _feature_sentence(product, facts["feature"] if facts else None)
            if answer:
                feature_answers.append(answer)
        if feature_answers:
            return " ".join(feature_answers)
    if "mobile" in normalized and any(word in normalized for word in ("plan", "offer", "option", "available")):
        plans = re.findall(r"\b(?:Prepaid\s+(?:Basic|Plus|Unlimited)|Postpaid\s+(?:Silver|Gold|Platinum))\b", text)
        unique_plans = list(dict.fromkeys(plans))
        if unique_plans:
            if len(unique_plans) == 1:
                return f"ZENDS offers the {unique_plans[0]} mobile plan."
            names = ", ".join(unique_plans[:-1]) + f", and {unique_plans[-1]}"
            return f"ZENDS offers {names} mobile plans."
    product_group = _query_product_group(query)
    capability_question = any(
        term in TOKEN_PATTERN.findall(normalized)
        for term in ("available", "capabilities", "offer", "offering", "offerings", "offers", "provide", "provides", "service", "services", "solution", "solutions")
    )
    if product_group and capability_question:
        capabilities = re.search(r"(?:^|(?<=[.!?])\s+)Services include\s+([^.!?]+)[.!?]", text, re.I)
        if capabilities:
            supported_capabilities = capabilities.group(1).strip().rstrip(",")
            subject = PRODUCT_GROUPS[product_group]["subject"]
            return f"ZENDS {subject} include {supported_capabilities}."
    return None


def deterministic_evidence_answer(query: str, intent: str, evidence: list[dict[str, Any]]) -> str | None:
    """Use compact source-derived fallback wording when a draft is unavailable."""
    if not evidence:
        return None
    # Explicit product and policy semantics take precedence over a forced
    # five-class intent prediction (for example, a comparison labelled Technical).
    factual_answer = _policy_answer(evidence) or _product_answer(query, evidence)
    if factual_answer:
        return factual_answer
    if intent in SUPPORT_INTENTS:
        text = _evidence_text(evidence).lower()
        support_capabilities = [
            ("24×7 technical support", "24×7 technical support"),
            ("network monitoring", "network monitoring"),
            ("setup guidance", "setup guidance"),
            ("troubleshooting", "troubleshooting assistance"),
            ("installation", "installation support"),
        ]
        supported = [label for phrase, label in support_capabilities if phrase in text]
        if supported:
            capabilities = ", ".join(supported[:-1]) + (f", and {supported[-1]}" if len(supported) > 1 else supported[0])
            product_group = _query_product_group(query)
            if product_group:
                subject = PRODUCT_GROUPS[product_group]["subject"]
                return f"ZENDS provides {capabilities} for its {subject}."
            return f"ZENDS provides {capabilities}."
        # Product/service descriptions alone are not troubleshooting steps.
        return None
    return None


PRICE_IN_ANSWER = re.compile(r"\$(\d+(?:\.\d+)?)")
SLA_IN_SOURCE = re.compile(
    r"\b(individual|business|enterprise)\s+users?\s+(?:have\s+)?(\d+(?:\.\d+)?)%\s+uptime\b",
    re.I,
)
PERCENT_IN_ANSWER = re.compile(r"(\d+(?:\.\d+)?)%")
CUSTOMER_TYPE_IN_ANSWER = re.compile(r"\b(individual|enterprise)\b", re.I)
POLICY_CLAIM_TERMS = {
    "Billing": ("bill", "billing", "invoice", "payment"),
    "Refund": ("refund", "refundable"),
    "Contracts": ("contract", "lock-in"),
    "SLA": ("sla", "uptime"),
    "Data Privacy": ("privacy", "gdpr", "encrypted", "iso"),
    "Fair Usage": ("fair usage", "usage cap", "unlimited plan"),
    "Support Tiers": ("support tier",),
    "Discounts": ("discount",),
}


def _evidence_price_tuples(source: str) -> set[tuple[str, str, str, str]]:
    """Extract complete product/country/customer/price relationships from evidence."""
    result: set[tuple[str, str, str, str]] = set()
    for product in _products_in_text(source):
        facts = _product_facts(source, product)
        if facts:
            for customer_type in ("individual", "enterprise"):
                for country, price in facts[customer_type].items():
                    result.add((product, country, customer_type, price))
        # A short, single-price source is also valid evidence for an LLM draft.
        block = _product_block(source, product) or ""
        for match in re.finditer(
            r"\$(\d+(?:\.\d+)?)\s+in\s+(?:the\s+)?(USA|India|Singapore|Thailand)\s+for\s+"
            r"(individual\s+users|enterprise\s+customers)", block, re.I,
        ):
            country = next(item for item in COUNTRIES if item.lower() == match.group(2).lower())
            customer_type = match.group(3).split()[0].lower()
            result.add((product, country, customer_type, match.group(1)))
    return result


def _unique_label(text: str, pattern: re.Pattern[str]) -> str | None:
    labels = {match.group(1).lower() for match in pattern.finditer(text)}
    return next(iter(labels)) if len(labels) == 1 else None


def _answer_price_tuples(answer: str, source_facts: set[tuple[str, str, str, str]]) -> list[tuple[str, str, str, str]] | None:
    """Bind every stated price to its local product, country, and customer type."""
    result: list[tuple[str, str, str, str]] = []
    source_products = {fact[0] for fact in source_facts}
    for sentence in _source_sentences(answer):
        if not PRICE_IN_ANSWER.search(sentence):
            continue
        products = [
            (match.start(), product)
            for product in PRODUCT_NAMES
            for match in re.finditer(rf"(?<!\w){re.escape(product)}(?!\w)", sentence, re.I)
        ]
        products.sort()
        if not products:
            if len(source_products) != 1:
                return None
            segments = [(next(iter(source_products)), sentence)]
        else:
            segments = [
                (product, sentence[start:products[index + 1][0] if index + 1 < len(products) else len(sentence)])
                for index, (start, product) in enumerate(products)
            ]
            if PRICE_IN_ANSWER.search(sentence[:products[0][0]]):
                return None
        sentence_country = _unique_label(sentence, re.compile(r"\b(USA|India|Singapore|Thailand)\b", re.I))
        for product, segment in segments:
            prices = list(PRICE_IN_ANSWER.finditer(segment))
            for index, match in enumerate(prices):
                before = segment[prices[index - 1].end() if index else 0:match.start()]
                after = segment[match.end():prices[index + 1].start() if index + 1 < len(prices) else len(segment)]
                customer_type = (
                    _unique_label(after, CUSTOMER_TYPE_IN_ANSWER)
                    or _unique_label(before, CUSTOMER_TYPE_IN_ANSWER)
                    or _unique_label(segment, CUSTOMER_TYPE_IN_ANSWER)
                )
                country = (
                    _unique_label(after, re.compile(r"\b(USA|India|Singapore|Thailand)\b", re.I))
                    or _unique_label(before, re.compile(r"\b(USA|India|Singapore|Thailand)\b", re.I))
                    or sentence_country
                )
                if country:
                    country = next(item for item in COUNTRIES if item.lower() == country)
                if not country or not customer_type:
                    compatible = {
                        (fact[1], fact[2]) for fact in source_facts
                        if fact[0] == product and fact[3] == match.group(1)
                        and (country is None or fact[1] == country)
                        and (customer_type is None or fact[2] == customer_type)
                    }
                    if len(compatible) != 1:
                        return None
                    country, customer_type = next(iter(compatible))
                result.append((product, country, customer_type, match.group(1)))
    return result


def _structured_relationships_valid(answer: str, evidence: list[dict[str, Any]]) -> bool:
    source = _evidence_text(evidence)
    source_prices = _evidence_price_tuples(source)
    if PRICE_IN_ANSWER.search(answer):
        claims = _answer_price_tuples(answer, source_prices)
        if claims is None or not claims or any(claim not in source_prices for claim in claims):
            return False

    source_slas = {tier.lower(): value for tier, value in SLA_IN_SOURCE.findall(source)}
    policy_text = {
        category: " ".join(str(chunk["text"]) for chunk in evidence if chunk.get("metadata", {}).get("policy_category") == category)
        for category in POLICY_CLAIM_TERMS
    }
    discount_source = policy_text["Discounts"]
    bulk_discount = re.search(r"Bulk enterprise users can receive up to (\d+(?:\.\d+)?)% discount", discount_source, re.I)
    annual_discount = re.search(r"Annual payment discount is (\d+(?:\.\d+)?)%", discount_source, re.I)
    contract_source = policy_text["Contracts"]
    enterprise_contract = re.search(r"Enterprise customers have a minimum (\d+)-month contract", contract_source, re.I)
    for sentence in _source_sentences(answer):
        percentages = list(PERCENT_IN_ANSWER.finditer(sentence))
        if source_slas and percentages and re.search(r"\b(sla|uptime|individual|business|enterprise)\b", sentence, re.I):
            tiers = list(re.finditer(r"\b(individual|business|enterprise)\b", sentence, re.I))
            for match in percentages:
                if not tiers:
                    if match.group(1) not in source_slas.values():
                        return False
                    continue
                closest = min(tiers, key=lambda tier: abs(tier.start() - match.start()))
                if source_slas.get(closest.group(1).lower()) != match.group(1):
                    return False
        if percentages and re.search(r"\bdiscount\b", sentence, re.I):
            bulk_claim = bool(re.search(r"\b(bulk|enterprise)\b", sentence, re.I))
            annual_claim = bool(re.search(r"\bannual\b", sentence, re.I))
            if bulk_claim != annual_claim:
                expected = bulk_discount if bulk_claim else annual_discount
                if expected is None or any(match.group(1) != expected.group(1) for match in percentages):
                    return False
            elif bulk_claim and annual_claim:
                # A combined sentence needs clause-level parsing; the source-derived
                # renderer handles it, so an ambiguous LLM draft fails closed.
                return False
        if enterprise_contract and re.search(r"\b\d+[- ]month\b", sentence, re.I):
            months = re.findall(r"\b(\d+)[- ]month\b", sentence, re.I)
            if any(month != enterprise_contract.group(1) for month in months):
                return False
            if re.search(r"\bindividual\b", sentence, re.I) and not re.search(r"\benterprise\b", sentence, re.I):
                return False
        # Numbers from one policy chunk cannot justify a claim about another.
        mentioned = [
            category for category, terms in POLICY_CLAIM_TERMS.items()
            if any(re.search(rf"\b{re.escape(term)}\b", sentence, re.I) for term in terms)
        ]
        if len(mentioned) == 1 and policy_text[mentioned[0]]:
            numbers = set(re.findall(r"\d+(?:\.\d+)?(?:%|tb|gb|mbps|gbps)?", sentence, re.I))
            policy_numbers = set(re.findall(r"\d+(?:\.\d+)?(?:%|tb|gb|mbps|gbps)?", policy_text[mentioned[0]], re.I))
            if not numbers.issubset(policy_numbers):
                return False
    return True


def is_grounded_answer(answer: str, evidence: list[dict[str, Any]]) -> bool:
    """Reject internal, contradictory, or numerically unsupported drafts."""
    candidate = " ".join(str(answer).split())
    if not candidate or len(candidate) > 700:
        return False
    candidate_terms = _terms(candidate)
    if any(term.startswith(internal) for term in candidate_terms for internal in INTERNAL_TERMS):
        return False
    source = _evidence_text(evidence).lower()
    number_pattern = re.compile(r"\$?\d+(?:\.\d+)?(?:%|tb|gb|mbps|gbps)?", re.I)
    candidate_numbers = {number.lower().lstrip("$") for number in number_pattern.findall(candidate)}
    source_numbers = {number.lower().lstrip("$") for number in number_pattern.findall(source)}
    if not candidate_numbers.issubset(source_numbers):
        return False
    if not _structured_relationships_valid(candidate, evidence):
        return False
    normalized_candidate = _normalized_phrase_text(candidate.replace("non-refundable", "non refundable"))
    normalized_source = _normalized_phrase_text(source.replace("non-refundable", "non refundable"))
    if (
        "non refundable after activation" in normalized_source
        and "refundable after activation" in normalized_candidate
        and "non refundable after activation" not in normalized_candidate
    ):
        return False
    if "no long term lock in" in normalized_source and "long term lock in" in normalized_candidate and "no long term lock in" not in normalized_candidate:
        return False
    source_terms = _terms(source)
    meaningful = candidate_terms - {"available", "information", "zends"}
    return len(meaningful & source_terms) >= 2


def compose_response(*, sentiment: str, priority: str, answer: str | None) -> str:
    """Return a concise grounded synthesis or a safe evidence abstention."""
    if not answer:
        if sentiment == "Angry":
            return "I’m sorry you’re experiencing this issue. The available ZENDS information does not provide enough detail to safely guide you further."
        return "The available ZENDS knowledge does not provide enough information to answer this question."
    if sentiment == "Angry" and not answer.lower().startswith(("i'm sorry", "i am sorry")):
        answer = f"I’m sorry this has been frustrating. {answer}"
    if priority == "High" and sentiment == "Angry":
        answer = f"{answer} I recognize this needs urgent attention."
    return answer
