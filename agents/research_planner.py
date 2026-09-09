"""
Stage 5-6: Research Planner agent. Builds a research plan and per-finding
hypotheses from the topic + detected findings, BEFORE any web search runs.
"""
from __future__ import annotations

from agents.llm_interface import LLM
from models.schemas import Finding, Hypothesis, ResearchPlan, ResearchQuestion

GENERIC_HYPOTHESIS_LIBRARY = [
    ("demand_side", "Demand-side conditions", "Change in aggregate or sector-specific demand."),
    ("supply_side", "Supply-side conditions", "Change in production, agricultural output, or input availability."),
    ("exchange_rate", "Exchange-rate movements", "Currency depreciation/appreciation affecting import prices."),
    ("monetary_policy", "Monetary policy", "Central bank policy-rate or liquidity decisions."),
    ("fiscal_policy", "Fiscal policy", "Government spending, taxation, or subsidy changes."),
    ("trade_policy", "Trade policy", "Tariffs, import/export restrictions, or trade agreements."),
    ("regulation", "Price regulation / administrative measures", "Government price controls or tariff-setting for regulated goods."),
    ("commodity_prices", "International commodity prices", "Global energy or food commodity price movements."),
    ("seasonal", "Seasonal factors", "Regular seasonal patterns (e.g., harvest cycles, holidays)."),
    ("base_effects", "Base effects", "Statistical effect of comparing against an unusual prior-year level."),
    ("expectations", "Expectations", "Shift in inflation or economic expectations among firms/households."),
]


def build_research_plan(topic: str, findings: list[Finding]) -> ResearchPlan:
    steps = [
        "Review the research topic and identify the core economic question.",
        "Summarize the key quantitative findings from the dataset.",
    ]
    questions: list[ResearchQuestion] = []
    qid = 0
    for f in findings:
        if f.importance == "low":
            continue
        qid += 1
        q = ResearchQuestion(
            id=f"Q{qid}",
            question=f"What historical and current policy or news developments could explain: {f.finding}",
            related_finding_ids=[f.id],
        )
        questions.append(q)
        steps.append(f"Investigate possible explanations for: {f.finding}")

    steps += [
        "Search for relevant Central Bank / government policy decisions in the affected period.",
        "Search for relevant current news covering the topic.",
        "Search for relevant international commodity-price or external developments where applicable.",
        "Match policy/news events against the timing of the observed data changes.",
        "Evaluate competing explanations against the collected evidence.",
        "Determine which tables and figures best communicate the key findings.",
        "Draft the economic interpretation and policy implications, using cautious causal language.",
    ]

    queries = build_search_queries(topic, findings)

    plan = ResearchPlan(topic=topic, steps=steps, questions=questions, search_queries=queries)

    # Optional LLM refinement of the plan's steps (keeps structure, improves wording).
    if LLM.enabled:
        system = ("You are a senior economist supervising a junior research assistant. Improve and reorder the "
                   "given research plan steps for clarity and rigor. Return a JSON list of strings only, same "
                   "approximate length, no commentary.")
        user = f"Topic: {topic}\nCurrent steps:\n" + "\n".join(f"- {s}" for s in steps)
        result = LLM.complete_json(system, user)
        if isinstance(result, list) and result and all(isinstance(x, str) for x in result):
            plan.steps = result

    return plan


def build_search_queries(topic: str, findings: list[Finding]) -> list[str]:
    queries: list[str] = [topic]
    top_findings = [f for f in findings if f.importance in ("high", "medium")][:6]

    policy_terms = ["central bank policy", "monetary policy decision", "government policy",
                     "tariff", "subsidy", "exchange rate", "import policy"]

    for f in top_findings:
        var_phrase = ", ".join(f.variables[:2]) if f.variables else ""
        base = f"{var_phrase} {f.period or ''}".strip()
        if base:
            queries.append(base)
        for term in policy_terms[:3]:
            queries.append(f"{var_phrase} {term} {f.period or ''}".strip())

    # de-duplicate, cap length
    seen, unique = set(), []
    for q in queries:
        q = " ".join(q.split())
        if q and q.lower() not in seen:
            seen.add(q.lower())
            unique.append(q)
    return unique[:14]


def generate_hypotheses(finding: Finding, topic: str) -> list[Hypothesis]:
    """Generate candidate explanations for a finding. Uses the generic
    hypothesis library, optionally refined/filtered by the LLM for topical
    relevance; never asserts any hypothesis is correct at this stage."""
    hyps: list[Hypothesis] = []
    if LLM.enabled:
        system = ("You are an economist generating candidate (not yet verified) explanations for an observed "
                   "data pattern. List 4-7 plausible, distinct hypotheses appropriate to the topic and finding. "
                   "Return JSON list of objects: {\"label\": str, \"description\": str}.")
        user = f"Research topic: {topic}\nObserved finding: {finding.finding}\nVariables: {finding.variables}"
        result = LLM.complete_json(system, user)
        if isinstance(result, list):
            for i, item in enumerate(result):
                if isinstance(item, dict) and item.get("label"):
                    hyps.append(Hypothesis(id=f"{finding.id}-H{i+1}", finding_id=finding.id,
                                            label=item["label"], description=item.get("description", "")))
    if not hyps:
        for i, (key, label, desc) in enumerate(GENERIC_HYPOTHESIS_LIBRARY):
            hyps.append(Hypothesis(id=f"{finding.id}-H{i+1}", finding_id=finding.id, label=label, description=desc))
    return hyps
