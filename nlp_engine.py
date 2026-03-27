"""
nlp_engine.py  —  InterviewIQ NLP Analysis Engine  (v2 — fixed)
Pure-Python, zero external ML dependencies.

Fixes in v2:
  - Quantifiers: catches '35 percent', '8 engineers', '$1.2M', 'tripled' etc.
  - STAR triggers: expanded to real interview language ('I led', 'I proactively')
  - Relevance: substring matching so 'challenged' hits 'challenge'
  - Scoring calibration: solid answers now score 65–80 (not 30–40)
  - Confidence base raised, penalties capped — fairer scoring
  - Relevance weighted 20% (was 25%) — stops low-kw answers dragging overall
  - All dict keys guaranteed present — no frontend KeyError / 500
"""

import re
import statistics
from collections import Counter


# ═══════════════════════════════════════════════════════════════
# LEXICONS
# ═══════════════════════════════════════════════════════════════

FILLER_WORDS = {
    "um", "uh", "er", "ah", "like", "you know", "i mean",
    "basically", "literally", "actually", "honestly", "right",
    "so yeah", "kind of", "sort of", "stuff", "things", "whatever",
    "anyway", "i guess", "i think", "i feel like", "to be honest",
    "at the end of the day", "going forward", "touch base",
}

CONFIDENCE_POSITIVE = {
    "achieved", "led", "managed", "delivered", "built", "created",
    "designed", "implemented", "improved", "increased", "reduced",
    "launched", "developed", "executed", "resolved", "optimized",
    "established", "drove", "spearheaded", "accelerated", "transformed",
    "pioneered", "championed", "orchestrated", "exceeded", "surpassed",
    "successfully", "effectively", "efficiently", "significantly",
    "directly", "independently", "proactively", "strategically",
    "consistently", "measurably", "substantially", "coordinated",
    "facilitated", "streamlined", "scaled", "secured", "negotiated",
    "influenced", "mentored", "delegated", "initiated", "deployed",
}

CONFIDENCE_NEGATIVE = {
    "tried", "attempted", "maybe", "possibly", "might", "perhaps",
    "somewhat", "fairly", "just", "only", "little", "bit",
    "struggle", "unfortunately", "couldn't", "wasn't able",
    "hard to say", "i don't know", "not sure", "unsure",
}

POWER_VERBS = {
    "accelerated", "achieved", "built", "championed", "collaborated",
    "coordinated", "created", "delivered", "designed", "developed",
    "directed", "drove", "established", "executed", "expanded",
    "generated", "implemented", "improved", "increased", "initiated",
    "innovated", "launched", "led", "managed", "optimized",
    "orchestrated", "pioneered", "produced", "reduced", "resolved",
    "scaled", "streamlined", "strengthened", "transformed", "unified",
    "secured", "negotiated", "mentored", "spearheaded", "deployed",
    "automated", "migrated", "refactored", "architected", "integrated",
}

STAR_TRIGGERS = {
    "situation": [
        "situation", "when i", "at the time", "we were", "faced with",
        "in my previous", "in my last", "working at", "at my", "in my role",
        "our team was", "the company", "we had", "i was working",
        "previously", "we needed", "the background", "context",
    ],
    "task": [
        "task", "responsible for", "my role", "i needed to", "required to",
        "assigned", "goal was", "objective", "my job was", "i was asked",
        "i had to", "the challenge", "we needed to", "our goal",
        "my responsibility", "i was tasked", "the problem", "needed to",
    ],
    "action": [
        "i decided", "i took", "i implemented", "i created", "i worked",
        "i developed", "i reached out", "i collaborated", "specifically",
        "to address", "i then", "i led", "i built", "i designed",
        "i set up", "i introduced", "i proposed", "i initiated",
        "i coordinated", "i established", "i ran", "we implemented",
        "i proactively", "i immediately", "my approach", "i focused",
        "i made sure", "i ensured", "i organized", "i pushed",
    ],
    "result": [
        "result", "resulted in", "outcome", "as a result", "which led to",
        "impact", "improved", "increased", "reduced", "saved", "achieved",
        "ultimately", "in the end", "by the end", "we saw", "we delivered",
        "ended up", "this led to", "we launched", "shipped",
        "we hit", "we exceeded", "we surpassed", "the impact",
        "performance improved", "costs went down", "revenue grew",
    ],
}

CATEGORY_KEYWORDS = {
    "behavioral": {
        "required": [
            "team", "challenge", "situation", "result", "learned", "experience",
            "outcome", "lesson", "difficult", "obstacle", "achieved",
            "delivered", "worked", "project", "role",
        ],
        "power": [
            "collaborated", "leadership", "initiative", "conflict", "resolution",
            "cross-functional", "stakeholder", "deadline", "pressure", "impact",
            "proactive", "proactively", "mentored", "motivated", "aligned",
            "communication", "accountability", "feedback", "transparency",
        ],
    },
    "technical": {
        "required": [
            "solution", "implemented", "system", "performance", "code",
            "architecture", "built", "developed", "designed", "deployed",
            "problem", "approach", "technical", "engineering",
        ],
        "power": [
            "scalable", "optimized", "refactored", "debugging", "testing",
            "deployment", "infrastructure", "microservices", "api", "database",
            "automation", "monitoring", "observability", "reliability",
            "latency", "throughput", "security", "cloud",
        ],
    },
    "situational": {
        "required": [
            "would", "approach", "first", "consider", "communicate", "team",
            "step", "plan", "assess", "evaluate", "priority",
        ],
        "power": [
            "prioritize", "stakeholders", "align", "escalate", "mitigate",
            "framework", "process", "strategy", "feedback", "transparent",
            "risk", "tradeoff", "consensus", "collaborate", "outcome",
        ],
    },
    "common": {
        "required": [
            "experience", "skills", "role", "company", "background", "years",
            "worked", "built", "led", "developed", "created", "passionate",
        ],
        "power": [
            "expertise", "contributed", "value", "growth", "opportunity",
            "mission", "culture", "impact", "driven", "excited",
            "goal", "achieve", "deliver", "team", "technology",
        ],
    },
}

STOP_WORDS = {
    "i","me","my","myself","we","our","ours","you","your","he","him","his",
    "she","her","it","its","they","them","their","this","that","these","those",
    "am","is","are","was","were","be","been","being","have","has","had","do",
    "does","did","will","would","shall","should","may","might","must","can",
    "could","a","an","the","and","but","if","or","because","as","until",
    "while","of","at","by","for","with","about","between","into","through",
    "during","before","after","to","from","up","down","in","out","on","off",
    "over","under","again","then","here","there","when","where","how","all",
    "both","each","few","more","most","some","no","not","only","same","so",
    "than","too","very","just","also","well","really","quite","rather",
}


# ═══════════════════════════════════════════════════════════════
# TEXT UTILITIES
# ═══════════════════════════════════════════════════════════════

def tokenize(text):
    return re.findall(r"\b[a-z']+\b", text.lower())

def sentence_split(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]

def word_count(text):
    return len(text.strip().split())

def sentence_count(text):
    return max(1, len(sentence_split(text)))

def avg_sentence_length(text):
    sents = sentence_split(text)
    if not sents:
        return 0.0
    return round(statistics.mean(len(s.split()) for s in sents), 1)

def syllable_count(word):
    w = word.lower().strip(".,!?;:")
    if len(w) <= 3:
        return 1
    w = re.sub(r'(?:[^laeiouy]es|ed|[^laeiouy]e)$', '', w)
    w = re.sub(r'^y', '', w)
    return max(1, len(re.findall(r'[aeiouy]{1,2}', w)))

def flesch_kincaid_grade(text):
    words = text.split()
    sents = sentence_count(text)
    if not words or not sents:
        return 0.0
    syllables = sum(syllable_count(w) for w in words)
    asl = len(words) / sents
    asw = syllables / len(words)
    return round(max(0, 0.39 * asl + 11.8 * asw - 15.59), 1)

def type_token_ratio(text):
    tokens = [w for w in tokenize(text) if w not in STOP_WORDS and len(w) > 2]
    if not tokens:
        return 0.0
    return round(len(set(tokens)) / len(tokens), 3)


# ═══════════════════════════════════════════════════════════════
# ANALYSIS MODULES
# ═══════════════════════════════════════════════════════════════

def detect_filler_words(text):
    text_lower = text.lower()
    tokens = tokenize(text)
    found = {}

    for t in tokens:
        if t in FILLER_WORDS:
            found[t] = found.get(t, 0) + 1

    for fw in FILLER_WORDS:
        if " " in fw and fw in text_lower:
            found[fw] = text_lower.count(fw)

    total = sum(found.values())
    rate  = round(total / max(1, len(tokens)) * 100, 1)
    return {
        "found":   found,
        "total":   total,
        "rate":    rate,
        "penalty": min(25, total * 4),
    }


def score_confidence(text):
    tokens     = tokenize(text)
    token_set  = set(tokens)
    text_lower = text.lower()

    pos_hits = token_set & CONFIDENCE_POSITIVE
    neg_hits = token_set & CONFIDENCE_NEGATIVE

    passive_count = len(re.findall(
        r'\b(was|were|been|is|are)\s+\w+ed\b', text_lower))

    fp_active = len(re.findall(
        r'\bi\s+(?:built|led|created|managed|achieved|drove|designed|'
        r'delivered|implemented|coordinated|launched|developed|reduced|'
        r'improved|established|initiated|proactively|executed|resolved|'
        r'mentored|scaled|streamlined|secured|negotiated|championed|'
        r'spearheaded|deployed|automated|migrated|orchestrated)\b',
        text_lower))

    score  = 60
    score += min(20, len(pos_hits) * 3)
    score -= min(15, len(neg_hits) * 3)
    score -= min(10, passive_count * 3)
    score += min(12, fp_active * 3)

    fillers = detect_filler_words(text)
    score  -= fillers["penalty"] // 2

    return {
        "score":             max(0, min(100, score)),
        "positive_words":    sorted(pos_hits),
        "negative_words":    sorted(neg_hits),
        "passive_count":     passive_count,
        "first_person_active": fp_active,
    }


def score_clarity(text):
    avg_sl = avg_sentence_length(text)
    grade  = flesch_kincaid_grade(text)
    ttr    = type_token_ratio(text)
    wc     = word_count(text)

    if 14 <= avg_sl <= 22:
        sl_score = 100
    elif avg_sl < 8:
        sl_score = max(30, 50 + avg_sl * 5)
    elif avg_sl > 35:
        sl_score = max(25, 100 - (avg_sl - 22) * 2.5)
    else:
        sl_score = max(40, 100 - abs(avg_sl - 18) * 1.8)

    if 8 <= grade <= 13:
        grade_score = 100
    else:
        grade_score = max(35, 100 - abs(grade - 11) * 5)

    if 100 <= wc <= 400:
        wc_score = 100
    elif wc < 40:
        wc_score = max(15, wc * 1.5)
    elif wc > 600:
        wc_score = max(45, 100 - (wc - 400) * 0.08)
    else:
        wc_score = max(50, 100 - abs(wc - 250) * 0.25)

    score = int(sl_score * 0.35 + grade_score * 0.30 + wc_score * 0.35)
    return {
        "score":               max(0, min(100, score)),
        "avg_sentence_length": avg_sl,
        "grade_level":         grade,
        "vocab_richness":      round(ttr * 100, 1),
        "word_count":          wc,
        "sentence_count":      sentence_count(text),
    }


def score_relevance(text, category):
    text_lower = text.lower()
    cat_data   = CATEGORY_KEYWORDS.get(category, CATEGORY_KEYWORDS["common"])
    required   = cat_data["required"]
    power      = cat_data["power"]

    # Substring match — 'challenges' matches 'challenge'
    req_found   = [w for w in required if w in text_lower]
    power_found = [w for w in power    if w in text_lower]

    all_found = list(dict.fromkeys(req_found + power_found))

    req_score   = (len(req_found) / max(1, len(required))) * 100
    power_score = min(100, len(power_found) * 10)
    score       = int(req_score * 0.55 + power_score * 0.45)

    missing_req = [w for w in required[:6] if w not in text_lower][:4]
    suggested   = [w for w in power        if w not in text_lower][:5]

    return {
        "score":              max(0, min(100, score)),
        "keywords_found":     all_found[:12],
        "keywords_missing":   missing_req,
        "keywords_suggested": suggested,
        "category":           category,
    }


def detect_star_method(text):
    text_lower = text.lower()
    components = {}
    for comp, triggers in STAR_TRIGGERS.items():
        hits = [t for t in triggers if t in text_lower]
        components[comp] = {"found": bool(hits), "triggers": hits[:2]}

    found_count = sum(1 for v in components.values() if v["found"])
    return {
        "components":       components,
        "score":            int((found_count / 4) * 100),
        "complete":         found_count >= 3,
        "components_count": found_count,
    }


def detect_quantifiers(text):
    patterns = [
        r'\d+\s*%',
        r'\d+\s*percent',
        r'\$\s*[\d,.]+(?:\s*(?:million|billion|thousand|k))?',
        r'\d+\s*(?:million|billion|thousand)\b',
        r'\d+\s*(?:x\b|times\b)',
        r'\d+\s*(?:people|users|customers|clients|engineers|developers|members|employees)',
        r'\d+\s*(?:months?|weeks?|days?|years?|hours?|quarters?)',
        r'(?:doubled|tripled|halved|quadrupled)',
        r'(?:first|top\s*\d+)',
    ]
    found = []
    for p in patterns:
        for m in re.finditer(p, text.lower()):
            found.append(m.group().strip())

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for item in found:
        if item not in seen:
            seen.add(item)
            deduped.append(item)

    return {
        "found": deduped[:8],
        "count": len(deduped),
        "score": min(100, len(deduped) * 20),
    }


def score_depth(text, category):
    wc    = word_count(text)
    quant = detect_quantifiers(text)
    star  = detect_star_method(text)
    pvs   = set(tokenize(text)) & POWER_VERBS

    length_score = min(100, (wc / 200) * 100)
    quant_score  = quant["score"]
    star_score   = star["score"]
    power_score  = min(100, len(pvs) * 14)

    score = int(
        length_score * 0.20 +
        quant_score  * 0.25 +
        star_score   * 0.35 +
        power_score  * 0.20
    )
    return {
        "score":             max(0, min(100, score)),
        "quantifiers":       quant,
        "star_method":       star,
        "power_verbs_used":  sorted(pvs),
    }


def get_verdict(overall):
    if overall >= 82:
        return {"label": "Excellent",  "color": "#2ed573", "emoji": "🏆",
                "message": "Outstanding — you're interview-ready!"}
    elif overall >= 68:
        return {"label": "Strong",     "color": "#a8ff78", "emoji": "✅",
                "message": "Solid answer. Minor polish will make it shine."}
    elif overall >= 54:
        return {"label": "Good",       "color": "#00e5ff", "emoji": "👍",
                "message": "Decent response — a few targeted improvements will elevate it."}
    elif overall >= 38:
        return {"label": "Needs Work", "color": "#ffd32a", "emoji": "⚠️",
                "message": "The core is there — needs more structure and specifics."}
    else:
        return {"label": "Weak",       "color": "#ff4757", "emoji": "❌",
                "message": "Requires substantial development before your interview."}


def generate_tips(conf, clarity, rel, depth, fillers):
    strengths    = []
    improvements = []

    # Confidence
    if conf["score"] >= 72:
        if conf["positive_words"]:
            strengths.append(
                f"Confident, action-oriented language: "
                f"{', '.join(list(conf['positive_words'])[:4])}.")
    else:
        if conf["negative_words"]:
            improvements.append(
                f"Replace weak words ({', '.join(list(conf['negative_words'])[:3])}) "
                "with confident verbs like 'achieved', 'led', 'delivered'.")
        if conf["passive_count"] > 1:
            improvements.append(
                f"Reduce passive voice ({conf['passive_count']} instances) — "
                "say 'I implemented X' not 'X was implemented'.")

    # Fillers
    if fillers["total"] == 0:
        strengths.append("Clean delivery — zero filler words detected.")
    elif fillers["total"] <= 2:
        fw = ', '.join(f'"{w}"' for w in list(fillers["found"].keys())[:3])
        improvements.append(f"Minimise fillers ({fw}). Replace with a deliberate pause.")
    else:
        fw = ', '.join(f'"{w}"' for w in list(fillers["found"].keys())[:4])
        improvements.append(
            f"High filler usage ({fillers['total']} instances: {fw}). "
            "Practise pausing silently instead.")

    # Clarity
    if clarity["score"] >= 72:
        strengths.append(
            f"Clear, well-structured prose "
            f"(avg {clarity['avg_sentence_length']} words/sentence, grade {clarity['grade_level']}).")
    else:
        if clarity["avg_sentence_length"] > 28:
            improvements.append(
                f"Sentences average {clarity['avg_sentence_length']} words — aim for 14–22.")
        if clarity["word_count"] < 80:
            improvements.append(
                "Too brief. Aim for 120–300 words to fully demonstrate your experience.")
        elif clarity["word_count"] > 500:
            improvements.append(
                "Tighten to 180–350 words. Concise, impactful answers are more memorable.")

    # Quantifiers
    star = depth["star_method"]
    if depth["quantifiers"]["count"] >= 2:
        nums = ", ".join(str(x) for x in depth["quantifiers"]["found"][:3])
        strengths.append(
            f"Good use of {depth['quantifiers']['count']} quantifiable metrics ({nums}).")
    else:
        improvements.append(
            "Add specific numbers: %, team size, timelines, cost or revenue impact. "
            "Numbers make achievements credible.")

    # STAR
    if star["complete"]:
        strengths.append("STAR structure detected — Situation, Task, Action, Result all present.")
    else:
        missing = [k.upper() for k, v in star["components"].items() if not v["found"]]
        if missing:
            improvements.append(
                f"Missing STAR components: {', '.join(missing)}. "
                "Use STAR for a clear, memorable narrative.")

    # Power verbs
    pvs = depth["power_verbs_used"]
    if pvs:
        strengths.append(f"Strong power verbs used: {', '.join(list(pvs)[:4])}.")
    else:
        improvements.append(
            "Add impact verbs: 'spearheaded', 'orchestrated', 'delivered', 'scaled'.")

    # Relevance
    if rel["score"] >= 65:
        strengths.append("Response well-aligned with the question's core themes.")
    elif rel["keywords_missing"]:
        improvements.append(
            f"Weave in key themes: {', '.join(rel['keywords_missing'][:4])}.")

    return {"strengths": strengths[:4], "improvements": improvements[:4]}


# ═══════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def analyze_response(text, question, category="common"):
    """
    Full NLP analysis. Returns a guaranteed-complete dict —
    every key always present so the frontend never gets a 500.
    """
    if not text or not text.strip():
        return {"error": "Empty response"}

    text = text.strip()

    fillers = detect_filler_words(text)
    conf    = score_confidence(text)
    clarity = score_clarity(text)
    rel     = score_relevance(text, category)
    depth   = score_depth(text, category)

    # Weighted overall — relevance 20% to avoid unfairly dragging good answers
    overall = int(
        conf["score"]    * 0.28 +
        clarity["score"] * 0.27 +
        rel["score"]     * 0.20 +
        depth["score"]   * 0.25
    )
    overall = max(0, min(100, overall))

    verdict = get_verdict(overall)
    tips    = generate_tips(conf, clarity, rel, depth, fillers)
    wc      = clarity["word_count"]

    return {
        "overall": overall,
        "verdict": verdict,
        "scores": {
            "confidence": conf["score"],
            "clarity":    clarity["score"],
            "relevance":  rel["score"],
            "depth":      depth["score"],
        },
        "stats": {
            "word_count":       wc,
            "sentence_count":   clarity["sentence_count"],
            "avg_sentence_len": clarity["avg_sentence_length"],
            "grade_level":      clarity["grade_level"],
            "vocab_richness":   clarity["vocab_richness"],
            "reading_time_sec": max(1, round(wc / 2.5)),
        },
        "keywords": {
            "found":     rel["keywords_found"],
            "missing":   rel["keywords_missing"],
            "suggested": rel["keywords_suggested"],
        },
        "confidence_details": {
            "positive_words":    conf["positive_words"],
            "negative_words":    conf["negative_words"],
            "passive_voice":     conf["passive_count"],
            "active_openers":    conf["first_person_active"],
        },
        "fillers":     fillers,
        "star_method": depth["star_method"],
        "quantifiers": depth["quantifiers"],
        "power_verbs": depth["power_verbs_used"],
        "tips":        tips,
        "category":    category,
        "question":    question,
        # AI fields — always present (overwritten by app.py when API available)
        "ai_available": False,
        "ai_summary":   "",
        "ai_tips":      [],
        "ai_rewrite":   "",
    }
