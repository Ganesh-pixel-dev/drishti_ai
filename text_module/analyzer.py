import logging
import os
import json
import re
import math
from collections import Counter

logger = logging.getLogger(__name__)

# === LOCAL HEURISTIC AI-TEXT DETECTOR (No API needed) ===

# Classic ChatGPT patterns
CLASSIC_AI_PHRASES = [
    "it is important to note", "it's important to note", "in conclusion",
    "furthermore", "moreover", "additionally", "in today's world",
    "in the realm of", "it is worth noting", "plays a crucial role",
    "has become increasingly", "as we delve", "a testament to",
    "the landscape of", "navigating the", "let's explore",
    "dive into", "the power of", "at its core", "in an era",
    "stands as a", "harness the power", "it's no secret",
    "on the other hand", "having said that", "that being said",
    "in essence", "to put it simply", "needless to say",
    "without a doubt", "in summary", "to summarize",
    "it goes without saying", "as a matter of fact",
    "from healthcare to", "unprecedented ways",
    "remarkable accuracy", "transformative power", "ever-evolving",
    "game-changer", "cutting-edge", "groundbreaking",
    "delve into", "tapestry of", "multifaceted",
]

# Modern ChatGPT motivational/coaching patterns (2024-2025 style)
MODERN_AI_PHRASES = [
    "the truth is", "here's the thing", "let me be honest",
    "most people", "but you clearly", "and that's okay",
    "which is powerful", "real clarity", "consistent action",
    "your biggest gap", "once you fix that", "everything else will",
    "start to align", "not thinking", "wasting time overthinking",
    "staying comfortable", "settle for something", "want more than that",
    "act on it", "even if it's messy", "even if it's uncertain",
    "you don't need more", "you need consistent",
    "what you choose to do next", "depending on what you",
    "at this stage", "point in life where",
    "but here's the catch", "here's what most people miss",
    "the reality is", "if you're being honest",
    "that's not a bad thing", "but it's also", "which means",
    "the key is", "the difference between", "the good news is",
    "you already know", "deep down", "stop waiting for",
    "permission to", "own your", "take ownership",
    "level up", "show up every day", "build momentum",
    "compound over time", "long game", "short-term discomfort",
    "comfort zone", "growth happens when", "lean into",
    "embrace the uncertainty", "trust the process",
    "clarity comes from", "action creates clarity",
]

# Hedging/filler qualifiers AI overuses
HEDGING_WORDS = {
    "various", "numerous", "significant", "substantial",
    "considerable", "remarkable", "crucial", "essential",
    "fundamental", "comprehensive", "robust", "innovative",
    "powerful", "meaningful", "impactful", "transformative",
    "dynamic", "compelling", "authentic", "genuine",
    "ultimately", "essentially", "effectively", "incredibly",
}

# Contrasting conjunctions AI loves to chain
AI_CONJUNCTIONS = [
    "but also", "but you", "but it", "but here",
    "which is", "which means", "which can",
    "and that", "and once", "and eventually",
    "because real", "because the", "because once",
]


def _compute_word_repetition_score(words):
    """AI text reuses the same vocabulary more than humans."""
    if len(words) < 20:
        return 0.0
    unique_ratio = len(set(words)) / len(words)
    # Human text typically has 0.55-0.75 unique ratio
    # AI text tends toward 0.45-0.60 (more repetitive)
    if unique_ratio < 0.50:
        return 0.15
    elif unique_ratio < 0.55:
        return 0.10
    return 0.0


def _compute_sentence_complexity_score(sentences):
    """
    AI produces sentences with very consistent internal complexity.
    Measure comma density per sentence. AI uses commas very uniformly.
    """
    if len(sentences) < 3:
        return 0.0
    
    comma_counts = [s.count(',') for s in sentences]
    avg = sum(comma_counts) / len(comma_counts)
    
    if avg == 0:
        return 0.0
    
    variance = sum((c - avg) ** 2 for c in comma_counts) / len(comma_counts)
    std = math.sqrt(variance)
    cv = std / avg if avg > 0 else 1.0
    
    # AI has very consistent comma usage (low CV)
    if cv < 0.4 and avg >= 2:
        return 0.15
    elif cv < 0.6 and avg >= 1.5:
        return 0.08
    return 0.0


def _local_heuristic_check(text):
    """
    Pure offline heuristic check for AI-written text.
    Returns (is_ai: bool, confidence: float, reason: str)
    """
    text_lower = text.lower()
    words = text_lower.split()
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 5]
    
    score = 0.0
    reasons = []
    
    # 1. Classic AI phrase matches
    classic_hits = sum(1 for p in CLASSIC_AI_PHRASES if p in text_lower)
    if classic_hits >= 5:
        score += 0.30
        reasons.append(f"{classic_hits} classic AI phrases detected")
    elif classic_hits >= 3:
        score += 0.20
        reasons.append(f"{classic_hits} classic AI phrases detected")
    elif classic_hits >= 1:
        score += 0.08
    
    # 2. Modern ChatGPT motivational style
    modern_hits = sum(1 for p in MODERN_AI_PHRASES if p in text_lower)
    if modern_hits >= 4:
        score += 0.30
        reasons.append(f"{modern_hits} modern ChatGPT-style phrases detected")
    elif modern_hits >= 2:
        score += 0.18
        reasons.append(f"{modern_hits} modern AI coaching phrases detected")
    elif modern_hits >= 1:
        score += 0.08
    
    # 3. AI conjunction chaining
    conj_hits = sum(1 for c in AI_CONJUNCTIONS if c in text_lower)
    if conj_hits >= 3:
        score += 0.15
        reasons.append(f"{conj_hits} AI-typical conjunction chains found")
    elif conj_hits >= 2:
        score += 0.08
    
    # 4. Sentence length uniformity
    if len(sentences) >= 3:
        lengths = [len(s.split()) for s in sentences]
        avg_len = sum(lengths) / len(lengths)
        if avg_len > 0:
            variance = sum((l - avg_len) ** 2 for l in lengths) / len(lengths)
            cv = math.sqrt(variance) / avg_len
            if cv < 0.25:
                score += 0.15
                reasons.append("Suspiciously uniform sentence lengths")
            elif cv < 0.40:
                score += 0.08
    
    # 5. Long sentences (AI loves extremely long compound sentences)
    if sentences:
        long_sentences = sum(1 for s in sentences if len(s.split()) > 35)
        if long_sentences >= 2:
            score += 0.12
            reasons.append(f"{long_sentences} extremely long compound sentences (AI pattern)")
        elif long_sentences >= 1 and len(sentences) <= 5:
            score += 0.06
    
    # 6. No informal language in substantial text
    informal_markers = ["dont", "cant", "wont", "gonna", "wanna", "gotta",
                        "tbh", "imo", "lol", "haha", "btw", "nah", "yeah",
                        "kinda", "sorta", "idk", "omg", "lmao", "bruh", "bro",
                        "ngl", "fr", "lowkey", "highkey", "smh", "tf"]
    has_informal = any(m in text_lower.split() for m in informal_markers)
    # contractions like "don't" are fine for AI, check for actual slang
    if not has_informal and len(words) > 30:
        score += 0.08
        reasons.append("No informal/slang language detected")
    
    # 7. Hedging word density
    hedge_count = sum(1 for w in words if w in HEDGING_WORDS)
    hedge_ratio = hedge_count / len(words) if words else 0
    if hedge_ratio > 0.04:
        score += 0.15
        reasons.append(f"High density of qualifier words ({hedge_count} found)")
    elif hedge_ratio > 0.025:
        score += 0.08
    
    # 8. Word repetition score
    rep_score = _compute_word_repetition_score(words)
    if rep_score > 0:
        score += rep_score
        reasons.append("Low vocabulary diversity (repetitive word usage)")
    
    # 9. Sentence complexity uniformity (comma patterns)
    complexity_score = _compute_sentence_complexity_score(sentences)
    if complexity_score > 0:
        score += complexity_score
        reasons.append("Suspiciously uniform sentence complexity")
    
    # 10. Second-person motivational pattern ("you" overuse)
    you_count = text_lower.split().count("you") + text_lower.split().count("your") + text_lower.split().count("you're")
    you_ratio = you_count / len(words) if words else 0
    if you_ratio > 0.05 and len(words) > 30:
        score += 0.10
        reasons.append(f"Heavy 'you/your' usage ({you_count} instances)")
    
    # 11. Em-dash and semicolon usage (ChatGPT signature punctuation)
    special_punct = text.count('—') + text.count('–') + text.count(';')
    if special_punct >= 3 and len(sentences) <= 8:
        score += 0.10
        reasons.append(f"Heavy em-dash/semicolon usage ({special_punct})")
    elif special_punct >= 2:
        score += 0.05
    
    # Cap at 1.0
    score = min(score, 1.0)
    
    is_ai = score >= 0.35
    
    if not reasons:
        reasons.append("No strong AI indicators detected")
    
    return is_ai, score, "; ".join(reasons)


def analyze_text(text_content):
    results = {}
    is_ai, score, reason = _local_heuristic_check(text_content)
    results["is_ai"] = is_ai
    results["ai_confidence"] = round(score * 100, 1)
    results["explanation"] = reason
    results["method"] = "Phrase and style heuristics (unvalidated)"

    results.update(_web_search(text_content))
    return results


def _web_search(text):
    """Optional: send the first long sentence to serper.dev and list the top hits.

    A search engine returns results for almost any sentence, so hits are leads to check
    by hand, not proof of copying. Needs SERPER_API_KEY. Off by default.
    """
    import requests
    key = os.environ.get("SERPER_API_KEY")
    if not key:
        return {"search_message": "Web search is off. Set SERPER_API_KEY to enable it. "
                                  "It sends the first long sentence to serper.dev."}
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if len(s.strip().split()) > 10]
    if not sentences:
        return {"search_message": "No sentence long enough to search for."}
    try:
        resp = requests.post("https://google.serper.dev/search", timeout=10,
                             headers={"X-API-KEY": key, "Content-Type": "application/json"},
                             data=json.dumps({"q": sentences[0]}))
        resp.raise_for_status()
        hits = resp.json().get("organic", [])[:3]
    except (requests.RequestException, ValueError) as exc:
        logger.warning("Web search failed: %s", exc)
        return {"search_message": "The web search request failed."}
    return {"search_message": f"Search returned {len(hits)} result(s) for the first long sentence. "
                              "Check them by hand; a result is not proof of copying.",
            "search_sources": [{"title": h.get("title"), "url": h.get("link")} for h in hits]}
