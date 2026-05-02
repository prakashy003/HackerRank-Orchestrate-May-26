"""
Triage agent.

For each support ticket:
  1. Retrieves the top relevant docs from the corpus via RAG.
  2. Calls Claude with a strict tool schema to produce structured output.
"""

import os
from anthropic import Anthropic
from retriever import Retriever

_client: Anthropic | None = None


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        _client = Anthropic()
    return _client


SYSTEM_PROMPT = """\
You are a support triage agent for three products: HackerRank, Claude (by Anthropic), and Visa.

For each ticket you must:
1. Classify the request type.
2. Identify the most relevant product area / support domain.
3. Decide whether to reply or escalate to a human.
4. If replying, write a helpful, grounded user-facing response.
5. Provide a concise justification for your decision.

─── REQUEST TYPE GUIDE ─────────────────────────────────────────────
Choose EXACTLY ONE:
• "bug"             — Broken behavior, errors, crashes, service outages, features not working
                      as designed. "Site is down", "getting an error", "not loading" = bug.
• "product_issue"   — Question about how an existing feature works, configuration help,
                      usage guidance, best-practice questions about existing features.
                      "How does X work?", "How do I configure Y?", "When should I use Z?" = product_issue.
• "feature_request" — Explicitly asking for a NEW capability that does not yet exist.
                      The user must clearly say they WANT something new added.
• "invalid"         — Spam, abusive content, completely off-topic, or clearly malicious.

─── ESCALATION RULES ───────────────────────────────────────────────
The key question is: does this require a HUMAN to take an action, or does the user just need INFORMATION?

ESCALATE when a human agent must personally act:
• Admin must restore access / reinstate a removed seat
• Finance must process a refund or investigate an unauthorised charge
• Security team must investigate a compromised account or active fraud
• Legal / compliance action required
• Site-wide outage that ops must respond to

REPLY (do NOT escalate) even for sensitive-sounding topics when the user is asking for:
• Where or how to report something (e.g. "Where do I report a stolen card?")
• What steps to take (e.g. "My card was stolen, what should I do?")
• Procedural information about an existing policy or process
• Contact details or links for next steps
• How to request something from support (account deletion, password reset, etc.)
If the support corpus contains directions, a support contact, or a link for the situation,
REPLY with that — you are providing information, not resolving the case yourself.

CRITICAL DISTINCTION — "replied" vs "escalated":
• "replied"   = You give the user information, steps, or contact details IN YOUR RESPONSE.
                Even "please contact support at X" or "fill in this form at Y" is a REPLY.
                The user receives a useful answer from you.
• "escalated" = You have NO actionable information to give. You hand the ticket off entirely.
                A human agent will contact the user with no message from you.
Use "escalated" ONLY when the corpus provides zero guidance and a human must act with
no information exchange. When in doubt, REPLY with whatever the corpus says.

ALSO reply (do NOT escalate) when:
• The user can self-resolve the issue with instructions from the docs
• The question is about how an existing feature works
• The user needs to contact support — tell them how to do so

─── GROUNDING RULES ────────────────────────────────────────────────
• Base your response ONLY on the provided support documentation.
• Do not invent policies, features, steps, or timelines not stated in the docs.
• If the provided docs clearly answer the question, REPLY — do not escalate out of caution.
• Only escalate when the docs genuinely do not cover the case AND a human must act.
• An accurate, doc-grounded reply is always better than an unnecessary escalation.
────────────────────────────────────────────────────────────────────
"""

TRIAGE_TOOL = {
    "name": "submit_triage",
    "description": "Submit the final structured triage result for a support ticket.",
    "input_schema": {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["replied", "escalated"],
                "description": "Whether the agent replies directly or escalates to a human.",
            },
            "product_area": {
                "type": "string",
                "description": "The most relevant support category or domain area (e.g. 'screen', 'billing', 'account-management', 'visa-consumer').",
            },
            "response": {
                "type": "string",
                "description": "User-facing answer grounded in the support corpus. If escalating, briefly explain why.",
            },
            "justification": {
                "type": "string",
                "description": "Concise internal explanation of the routing/answering decision.",
            },
            "request_type": {
                "type": "string",
                "enum": ["product_issue", "feature_request", "bug", "invalid"],
                "description": "Best-fit classification of the request.",
            },
        },
        "required": ["status", "product_area", "response", "justification", "request_type"],
    },
}


def triage_ticket(
    issue: str,
    subject: str,
    company: str,
    retriever: Retriever,
) -> dict:
    query = f"{subject} {issue}".strip()

    # Retrieve with company filter first; if company is unknown retrieve globally
    company_filter = company if company and company.lower() not in ("none", "") else None
    docs = retriever.retrieve(query, company=company_filter, n=8)

    # Always blend in a global search to catch cross-domain docs or missed hits
    extra = retriever.retrieve(query, company=None, n=5)
    seen = {d["filepath"] for d in docs}
    for d in extra:
        if d["filepath"] not in seen:
            docs.append(d)
            seen.add(d["filepath"])

    # Sort by score descending, keep top 8
    docs.sort(key=lambda d: d["score"], reverse=True)
    docs = docs[:8]

    # Build context block
    context = "\n\n---\n\n".join(
        f"[Doc {i} | {d['company']} / {d['product_area']} | score={d['score']}]\n{d['content']}"
        for i, d in enumerate(docs, 1)
    )

    user_message = (
        f"**Ticket**\n"
        f"Company: {company}\n"
        f"Subject: {subject or '(none)'}\n"
        f"Issue: {issue}\n\n"
        f"**Relevant support documentation**\n\n{context}\n\n"
        f"Analyse the ticket and call submit_triage with your answer."
    )

    response = _get_client().messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        tools=[TRIAGE_TOOL],
        tool_choice={"type": "tool", "name": "submit_triage"},
        messages=[{"role": "user", "content": user_message}],
        temperature=0,
    )

    # Extract the tool_use block (guaranteed by tool_choice={"type":"tool"})
    for block in response.content:
        if block.type == "tool_use" and block.name == "submit_triage":
            result = block.input
            # Normalise casing just in case
            result["status"] = str(result.get("status", "escalated")).lower()
            result["request_type"] = str(result.get("request_type", "product_issue")).lower()
            return result

    # Fallback — should not happen with tool_choice forced
    return {
        "status": "escalated",
        "product_area": "unknown",
        "response": "Unable to process this ticket automatically. Escalating to human support.",
        "justification": "Unexpected model response format.",
        "request_type": "product_issue",
    }
