"""
Critique engine: evaluates LLM responses for correctness, groundedness, and completeness.
Uses an LLM to score and analyze responses against source context.
"""

from app.services.llm_provider import call_llm
import json
import re

CRITIQUE_SYSTEM_PROMPT = """You are a rigorous answer evaluator. Given a question, an AI-generated response, and source context chunks, evaluate the response on these dimensions:

1. **Correctness** (0.0-1.0): Is the response factually accurate based on the context?
2. **Groundedness** (0.0-1.0): Does the response stick to information in the provided context, or does it hallucinate?
3. **Completeness** (0.0-1.0): Does the response address all aspects of the question using available context?

Also identify:
- **Issues**: Specific problems (hallucinations, misinterpretations, omissions)
- **Suggestions**: How the response could be improved

Respond ONLY in this JSON format:
{
    "correctness_score": 0.0,
    "groundedness_score": 0.0,
    "completeness_score": 0.0,
    "issues": ["issue1", "issue2"],
    "suggestions": ["suggestion1", "suggestion2"],
    "summary": "Brief overall assessment"
}"""


async def critique_response(
    query: str,
    response_text: str,
    context_chunks: list[str],
    model_id: str = "gpt-4o-mini",
) -> dict:
    """
    Run critique evaluation on a response.
    Returns parsed critique scores and feedback.
    """
    context_str = "\n\n---\n\n".join(context_chunks) if context_chunks else "(No context provided)"

    user_prompt = f"""**Question:** {query}

**AI Response to Evaluate:**
{response_text}

**Source Context Chunks:**
{context_str}

Evaluate the response. Return JSON only."""

    messages = [
        {"role": "system", "content": CRITIQUE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    result = await call_llm(model_id, messages, temperature=0.1, max_tokens=1500)
    text = result["text"]

    # Extract JSON from response (handle markdown fences)
    json_match = re.search(r"\{[\s\S]*\}", text)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            return {
                "correctness_score": min(1.0, max(0.0, parsed.get("correctness_score", 0))),
                "groundedness_score": min(1.0, max(0.0, parsed.get("groundedness_score", 0))),
                "completeness_score": min(1.0, max(0.0, parsed.get("completeness_score", 0))),
                "issues": parsed.get("issues", []),
                "suggestions": parsed.get("suggestions", []),
                "summary": parsed.get("summary", ""),
            }
        except json.JSONDecodeError:
            pass

    # Fallback if parsing fails
    return {
        "correctness_score": 0.0,
        "groundedness_score": 0.0,
        "completeness_score": 0.0,
        "issues": ["Failed to parse critique response"],
        "suggestions": [],
        "summary": text[:500],
    }
