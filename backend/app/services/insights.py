"""
Automated insights: summarization, topic extraction, reasoning chain analysis,
and cross-model comparison.
"""

from app.services.llm_provider import call_llm
import json
import re

INSIGHT_SYSTEM_PROMPT = """You are an analytical assistant that generates structured insights from AI model responses. Given a question and one or more model responses, produce:

1. **Summary**: A concise synthesis of the key information across responses
2. **Key Topics**: The main topics/concepts addressed (list of strings)
3. **Reasoning Quality**: Assessment of reasoning depth and coherence
4. **Model Comparison**: If multiple responses, compare their approaches and quality

Respond ONLY in this JSON format:
{
    "summary": "...",
    "key_topics": ["topic1", "topic2"],
    "reasoning_quality": "...",
    "model_comparison": "...",
    "metadata": {
        "agreement_level": "high|medium|low",
        "response_count": 0,
        "dominant_approach": "..."
    }
}"""


async def generate_insights(
    query: str,
    responses: list[dict],
    model_id: str = "gpt-4o-mini",
) -> dict:
    """
    Generate structured insights from one or more model responses.
    Each response dict should have: model_id, model_name, response (text).
    """
    responses_str = ""
    for i, r in enumerate(responses):
        responses_str += f"\n\n--- Response from {r.get('model_name', r.get('model_id', f'Model {i+1}'))} ---\n"
        responses_str += r.get("response", r.get("text", ""))

    user_prompt = f"""**Question:** {query}

**Model Responses:**
{responses_str}

Generate structured insights. Return JSON only."""

    messages = [
        {"role": "system", "content": INSIGHT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    result = await call_llm(model_id, messages, temperature=0.2, max_tokens=2000)
    text = result["text"]

    json_match = re.search(r"\{[\s\S]*\}", text)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            return {
                "summary": parsed.get("summary", ""),
                "key_topics": parsed.get("key_topics", []),
                "reasoning_quality": parsed.get("reasoning_quality", ""),
                "model_comparison": parsed.get("model_comparison", ""),
                "metadata": parsed.get("metadata", {}),
            }
        except json.JSONDecodeError:
            pass

    return {
        "summary": text[:500],
        "key_topics": [],
        "reasoning_quality": "Failed to parse",
        "model_comparison": "",
        "metadata": {},
    }
