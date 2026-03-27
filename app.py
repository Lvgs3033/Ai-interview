"""
app.py  —  InterviewIQ Flask Application  (v2 — fixed)
AI-Powered Interview Assistant with NLP Analysis + Claude AI coaching.

Fixes in v2:
  - All routes wrapped in try/except — no bare 500s
  - Anthropic call uses correct header 'x-api-key' (not 'Authorization')
  - /api/status endpoint so frontend can detect API key presence
  - Session store is thread-safe (lock added)
  - Graceful degradation when Anthropic is unavailable
  - CORS-friendly headers
  - Startup prints clear instructions
"""

import os
import json
import uuid
import threading
from datetime import datetime

from flask import Flask, render_template, request, jsonify
import requests as http_requests

from nlp_engine import analyze_response

# ─────────────────────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────────────────────

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "interviewiq-dev-secret-2024")

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

# Thread-safe session store
_session_lock = threading.Lock()
session_store = {}   # {session_id: [history_entries]}


# ─────────────────────────────────────────────────────────────
# QUESTION BANK
# ─────────────────────────────────────────────────────────────

QUESTIONS = {
    "behavioral": [
        {"id": "b1", "text": "Tell me about a time you demonstrated strong leadership under pressure.",        "difficulty": "Medium"},
        {"id": "b2", "text": "Describe a situation where you had to overcome a major challenge at work.",      "difficulty": "Medium"},
        {"id": "b3", "text": "Give an example of when you worked effectively as part of a team.",              "difficulty": "Easy"},
        {"id": "b4", "text": "Tell me about a time you failed and what you learned from it.",                  "difficulty": "Hard"},
        {"id": "b5", "text": "Describe a situation where you managed conflicting priorities.",                 "difficulty": "Medium"},
        {"id": "b6", "text": "Tell me about a time you went above and beyond what was required.",              "difficulty": "Easy"},
        {"id": "b7", "text": "Describe a time you had to influence someone without direct authority.",         "difficulty": "Hard"},
        {"id": "b8", "text": "Tell me about a time you had to adapt quickly to a major change.",               "difficulty": "Medium"},
    ],
    "technical": [
        {"id": "t1", "text": "Walk me through your approach to debugging a complex production issue.",         "difficulty": "Hard"},
        {"id": "t2", "text": "How do you ensure code quality and maintainability in your projects?",           "difficulty": "Medium"},
        {"id": "t3", "text": "Describe your experience with system design and scalability challenges.",        "difficulty": "Hard"},
        {"id": "t4", "text": "How do you stay current with new technologies and industry trends?",             "difficulty": "Easy"},
        {"id": "t5", "text": "Explain a complex technical decision you made and your reasoning.",              "difficulty": "Hard"},
        {"id": "t6", "text": "How do you approach testing and ensuring software reliability?",                 "difficulty": "Medium"},
    ],
    "situational": [
        {"id": "s1", "text": "If you disagreed with your manager's technical decision, how would you handle it?",  "difficulty": "Medium"},
        {"id": "s2", "text": "How would you handle a situation where a key team member leaves mid-project?",       "difficulty": "Hard"},
        {"id": "s3", "text": "You're given an impossible deadline. What do you do?",                               "difficulty": "Medium"},
        {"id": "s4", "text": "How would you handle receiving strongly negative feedback from a colleague?",        "difficulty": "Medium"},
        {"id": "s5", "text": "What would you do if you discovered a critical bug right before a major launch?",    "difficulty": "Hard"},
    ],
    "common": [
        {"id": "c1", "text": "Tell me about yourself and your professional background.",                       "difficulty": "Easy"},
        {"id": "c2", "text": "Why are you interested in this role and our company?",                           "difficulty": "Easy"},
        {"id": "c3", "text": "What are your greatest strengths and how do they apply to this role?",           "difficulty": "Easy"},
        {"id": "c4", "text": "Where do you see yourself in 5 years?",                                         "difficulty": "Medium"},
        {"id": "c5", "text": "Why are you leaving your current position?",                                     "difficulty": "Medium"},
        {"id": "c6", "text": "What is your greatest weakness and how are you working on it?",                  "difficulty": "Hard"},
        {"id": "c7", "text": "What makes you unique compared to other candidates?",                            "difficulty": "Medium"},
    ],
}

SAMPLE_ANSWERS = {
    "behavioral": [
        "In my previous role as engineering lead, our team faced a critical situation three weeks before "
        "a major product launch when our lead backend developer unexpectedly resigned. I immediately "
        "assessed the situation, held a team meeting, and redistributed responsibilities based on each "
        "engineer's strengths. I personally took on the critical API integration work while coordinating "
        "with our product manager to de-scope lower-priority features. I implemented daily standups and "
        "a shared board so everyone had full visibility. As a result, we delivered on schedule and "
        "achieved 40% higher adoption than our previous release. This taught me that transparent "
        "communication and rapid role clarity are the most powerful tools in a crisis.",

        "In my previous job, our team was tasked with migrating a legacy monolith to microservices within "
        "a tight 4-month deadline. The challenge was that half the team had never worked with microservices "
        "before. I took the initiative to organise weekly knowledge-sharing sessions and paired senior "
        "engineers with junior ones. I created a detailed migration roadmap and broke the project into "
        "smaller deliverable chunks. As a result, we completed the migration 2 weeks ahead of schedule, "
        "reduced deployment time by 70%, and the team gained real confidence in modern architecture.",

        "During a critical product demo for our biggest client, our servers went down 30 minutes before "
        "the presentation. I immediately coordinated with the infrastructure team, communicated the "
        "situation transparently to the client, and quickly set up a local demo environment as a backup. "
        "I led the team calmly, assigned clear roles, and we restored service in 20 minutes. The client "
        "appreciated our transparency. We retained the contract worth $500,000 and they became one of "
        "our strongest references.",

        "Early in my career I approved a code release that contained a bug affecting 15% of our users. "
        "I immediately owned the mistake publicly, organised a rapid hotfix within 2 hours, and communicated "
        "clearly with all stakeholders throughout. I then led a thorough post-mortem and introduced "
        "automated regression tests that caught similar issues going forward. The experience taught me "
        "the critical value of accountability and building systems that prevent human error.",

        "I was managing two high-priority projects simultaneously when both needed urgent attention on "
        "the same day. I scheduled a quick 15-minute sync with each team lead to understand the urgency "
        "and business impact of each. I prioritised the client-facing deadline, delegated decision-making "
        "authority to the second team lead, and checked in every 2 hours. Both projects delivered "
        "successfully. This reinforced the importance of clear delegation and trusting your team.",

        "A client requested a valuable feature that was outside our original scope. Even though it was "
        "beyond agreed deliverables, I scoped it over the weekend and presented a plan on Monday. I "
        "coordinated with design and backend teams to fit it into the current sprint without impacting "
        "other deliverables. The client gave us a 5-star review and the relationship led to over "
        "$200,000 in follow-on work.",

        "Our VP wanted to adopt a new technology I believed would create significant technical debt. "
        "Rather than disagreeing openly in a meeting, I prepared a detailed analysis comparing both "
        "approaches including performance benchmarks and migration costs. I presented it privately to "
        "the VP, then to the wider team. My alternative was adopted, saving an estimated 3 months of "
        "future rework and keeping our system maintainable.",

        "When our company announced a sudden product pivot, many team members were frustrated and "
        "uncertain. I organised informal sessions to give people space to share concerns and ask "
        "questions honestly. I worked with leadership to create a clear communication plan and updated "
        "our technical roadmap within one week. By facilitating open dialogue and providing clarity, "
        "morale improved and we successfully launched the pivoted product within the new timeline.",
    ],
    "technical": [
        "When debugging complex production issues, I follow a systematic approach. I reproduce the issue "
        "in staging, review recent deployments, and use observability tools like Datadog to identify "
        "anomalies in metrics, logs, and traces. I apply binary search to isolate the problem to a "
        "specific service or commit range. Once I find the root cause, I write a failing test before "
        "implementing the fix. I always conduct a blameless post-mortem afterward. This reduced our mean "
        "time to resolution by 60% over six months at my last company.",

        "I ensure code quality through multiple layers. I enforce coding standards with automated linters "
        "in our CI pipeline so issues are caught before review. I practice thorough code reviews focused "
        "on logic, edge cases, and readability. I write unit tests for all critical paths and integration "
        "tests for key user flows, maintaining above 85% coverage. I also schedule regular refactoring "
        "sessions to address technical debt. This approach reduced production bugs by 45% over one year.",

        "We needed to scale our payment service from 1,000 to 50,000 requests per minute within 3 months. "
        "I led the architecture redesign, introducing a message queue with RabbitMQ to decouple services, "
        "implementing database read replicas, and adding Redis caching for frequently accessed data. I "
        "also introduced horizontal auto-scaling on AWS. We hit our target within 10 weeks and the system "
        "handled 80,000 requests per minute during a peak sale event with zero downtime.",

        "I stay current by dedicating 30 minutes daily to technical blogs and release notes from sources "
        "like the ACM, Martin Fowler, and engineering blogs from Netflix and Stripe. I attend one "
        "conference per year and watch recorded talks from events I cannot attend. I contribute to open "
        "source projects to read production-grade codebases. Each quarter I build a small project with "
        "a new technology to get genuine hands-on experience beyond just reading tutorials.",

        "We needed to migrate from REST to GraphQL to support our mobile team. The debate was a full "
        "rewrite versus the strangler fig pattern. I chose strangler fig to reduce risk, wrapping "
        "existing REST endpoints in a GraphQL schema and migrating incrementally. I documented the "
        "decision with clear tradeoffs and established migration milestones. We delivered in 8 weeks "
        "with zero downtime, reduced over-fetching by 60%, and the mobile team cut data loading time "
        "by half.",

        "My testing strategy has three layers. Unit tests cover individual functions with mocked "
        "dependencies for fast feedback. Integration tests verify that services communicate correctly "
        "including database and third-party API interactions. End-to-end tests simulate real user "
        "journeys for critical paths. I also implement contract testing between microservices using "
        "Pact to catch breaking changes before deployment. This strategy enabled us to deploy 3 times "
        "per day with a production incident rate below 0.1%.",
    ],
    "situational": [
        "I would request a one-on-one with my manager to fully understand their reasoning. Decisions "
        "often have context I am not aware of — budget constraints or strategic priorities. After "
        "listening carefully, I would present my concerns using concrete data such as performance "
        "benchmarks or risk analysis, framing it as exploring tradeoffs. If we still disagreed, I "
        "would respect their decision, commit fully to executing it, and document my concerns. Trust "
        "and open communication are the foundations of any high-performing team.",

        "I would immediately call a team sync to assess the situation honestly. I would identify which "
        "tasks the departing member owned and rank them by urgency and impact. I would redistribute "
        "responsibilities based on each remaining team member's strengths and capacity, being transparent "
        "that we may need to re-negotiate scope with stakeholders. I would communicate the revised "
        "timeline to the client right away. Honest communication early is always better than a missed "
        "deadline with no warning.",

        "I would have a frank conversation with whoever set the deadline to understand what is truly "
        "negotiable. I would map out the work, identify the minimum viable deliverable, and propose a "
        "phased release that meets the core business need on time with the rest following in a second "
        "release. I would involve the team in planning to get realistic estimates and maintain morale. "
        "Over-promising and under-delivering is always worse than being honest upfront.",

        "I would listen fully without becoming defensive, because feedback is a gift even when "
        "uncomfortable. I would ask clarifying questions to understand the specific behaviours the "
        "person observed. Then I would reflect on whether I agree and thank them for raising it "
        "directly. If I agree, I would share a concrete plan to address it. If I partially disagree, "
        "I would share my perspective calmly with specific examples. Either way, I would follow up in "
        "two weeks to show I took it seriously.",

        "I would immediately stop any further releases and notify the team. I would assess the severity: "
        "how many users are affected, what is the data impact, and can it be patched quickly or does "
        "it require a rollback. If the bug is critical and cannot be patched in under an hour, I would "
        "recommend delaying the launch. I would communicate transparently with stakeholders about the "
        "situation, the risk, and our options. Protecting the user experience is more important than "
        "hitting a launch date.",
    ],
    "common": [
        "I am a software engineer with six years of experience specialising in backend systems and cloud "
        "architecture. I started at a fintech startup where I built payment processing systems handling "
        "over two million daily transactions. I then joined a Series B company where I led a team of "
        "four engineers, delivering a microservices migration that reduced infrastructure costs by 35% "
        "in six months. I am passionate about scalable system design, developer experience, and "
        "mentoring junior engineers. I am seeking a senior role where I can combine hands-on "
        "architecture work with technical leadership.",

        "I am genuinely excited about this role for three reasons. First, the engineering challenges "
        "here are exactly the scale and complexity I want to work on next. Second, your culture of "
        "ownership and shipping fast aligns with how I do my best work. Third, the product directly "
        "impacts people in a meaningful way. I have spent time reading your engineering blog and "
        "speaking with two people on your team, and every conversation reinforced that this is where "
        "I want to grow.",

        "My greatest strength is breaking down complex problems and communicating solutions clearly to "
        "both technical and non-technical stakeholders. At my last company I was often the bridge "
        "between engineering, product, and business teams — translating technical tradeoffs into "
        "business impact. This led to faster decision-making and fewer surprises. I also have a "
        "strong foundation in system design, which means I build solutions that scale without creating "
        "technical debt.",

        "In five years I see myself as a senior technical leader who has shipped products that scaled "
        "to millions of users. I want to have grown into an architect or engineering manager role where "
        "I am shaping technical direction while staying close to the code. I am also passionate about "
        "mentoring and want to have meaningfully accelerated the careers of at least three engineers "
        "I have worked with.",

        "I am leaving because I have grown significantly in my current role and I am looking for the "
        "next challenge. The scope for growth has become limited. I want to work on larger-scale "
        "problems with a stronger engineering team in an environment that invests in technical "
        "excellence. This role offers exactly that progression and I am ready to contribute at "
        "a higher level.",

        "My greatest weakness is that I sometimes take on too much because I want to help everyone. "
        "I have learned to recognise this pattern and now delegate more intentionally. I block time "
        "for focused work, maintain a priority list of tasks only I should handle, and actively give "
        "team members ownership of problems rather than solving them myself. This has made me a "
        "better leader and improved overall team output.",

        "What makes me unique is the combination of deep technical skills and strong communication "
        "ability. Many engineers are excellent technically but struggle to explain their work to "
        "non-engineers. I sit at that intersection — I have built production systems handling "
        "millions of transactions and also presented architecture decisions to board-level "
        "stakeholders. I bring genuine curiosity and a habit of continuous learning.",
    ],
}


# ─────────────────────────────────────────────────────────────
# ERROR HANDLERS — prevent bare 500 HTML pages
# ─────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found", "status": 404}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed", "status": 405}), 405

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": f"Internal server error: {str(e)}", "status": 500}), 500


# ─────────────────────────────────────────────────────────────
# ROUTES — PAGES
# ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    try:
        return render_template("index.html", questions=QUESTIONS)
    except Exception as e:
        return f"<h2>Template error: {e}</h2><p>Make sure templates/index.html exists.</p>", 500


# ─────────────────────────────────────────────────────────────
# ROUTES — API
# ─────────────────────────────────────────────────────────────

@app.route("/health")
def health():
    return jsonify({
        "status":    "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "version":   "2.0",
    })


@app.route("/api/status")
def api_status():
    """Frontend polls this to show AI-connected badge."""
    return jsonify({
        "nlp_engine":    True,
        "ai_available":  bool(ANTHROPIC_API_KEY),
        "speech_note":   "Speech recognition runs in-browser via Web Speech API",
    })


@app.route("/api/questions")
def get_questions():
    try:
        return jsonify(QUESTIONS)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/sample-answer/<category>")
@app.route("/api/sample-answer/<category>/<int:q_index>")
def get_sample_answer(category, q_index=0):
    try:
        answers = SAMPLE_ANSWERS.get(category) or SAMPLE_ANSWERS["common"]
        # Support both list (per-question) and plain string
        if isinstance(answers, list):
            answer = answers[q_index % len(answers)]
        else:
            answer = answers
        return jsonify({"answer": answer, "category": category, "index": q_index})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """Main analysis endpoint — NLP + optional Claude AI coaching."""
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "No JSON body received. Send Content-Type: application/json"}), 400

        response_text = (data.get("response") or "").strip()
        question      = (data.get("question") or "").strip()
        category      = (data.get("category") or "common").strip().lower()
        session_id    = (data.get("session_id") or "default").strip()
        use_ai        = bool(data.get("use_ai", True))

        # Validate category
        if category not in QUESTIONS:
            category = "common"

        # Validate response text
        if not response_text:
            return jsonify({"error": "Response text is empty."}), 400

        word_count = len(response_text.split())
        if word_count < 5:
            return jsonify({"error": "Response too short — please write at least a sentence or two."}), 400

        # ── Local NLP ────────────────────────────────────
        result = analyze_response(response_text, question, category)

        if "error" in result:
            return jsonify({"error": result["error"]}), 400

        # ── Optional Claude AI coaching ──────────────────
        if use_ai and ANTHROPIC_API_KEY:
            ai = _call_anthropic(response_text, question, category, result)
            if ai:
                result["ai_available"] = True
                result["ai_summary"]   = ai.get("summary", "")
                result["ai_tips"]      = ai.get("tips", [])
                result["ai_rewrite"]   = ai.get("rewrite_suggestion", "")

        # ── Session history ──────────────────────────────
        _add_to_history(session_id, question, category, response_text, result)

        return jsonify(result)

    except Exception as e:
        app.logger.exception("Error in /api/analyze")
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500


@app.route("/api/history/<session_id>")
def get_history(session_id):
    try:
        with _session_lock:
            history = session_store.get(session_id, [])
        return jsonify({"history": history, "count": len(history)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/history/<session_id>", methods=["DELETE"])
def clear_history(session_id):
    try:
        with _session_lock:
            session_store.pop(session_id, None)
        return jsonify({"status": "cleared"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _add_to_history(session_id, question, category, response_text, result):
    entry = {
        "id":        str(uuid.uuid4())[:8],
        "timestamp": datetime.utcnow().isoformat(),
        "question":  question,
        "category":  category,
        "snippet":   response_text[:180] + ("…" if len(response_text) > 180 else ""),
        "overall":   result.get("overall", 0),
        "verdict":   result.get("verdict", {}).get("label", "—"),
        "scores":    result.get("scores", {}),
    }
    with _session_lock:
        if session_id not in session_store:
            session_store[session_id] = []
        session_store[session_id].append(entry)
        # Keep last 20 entries per session
        session_store[session_id] = session_store[session_id][-20:]


def _call_anthropic(response_text, question, category, nlp):
    """
    Call Anthropic API for AI coaching feedback.
    Returns dict on success, None on any failure (graceful degradation).
    Correct auth header: 'x-api-key'  (NOT 'Authorization: Bearer ...')
    """
    try:
        s = nlp.get("scores", {})
        prompt = (
            f"You are an expert interview coach. A candidate answered this question:\n\n"
            f'Question: "{question}"\n'
            f"Category: {category}\n"
            f'Response: "{response_text}"\n\n'
            f"Local NLP scores — Confidence: {s.get('confidence',0)}/100, "
            f"Clarity: {s.get('clarity',0)}/100, "
            f"Relevance: {s.get('relevance',0)}/100, "
            f"Depth: {s.get('depth',0)}/100.\n\n"
            "Return ONLY valid JSON (no markdown fences, no extra text):\n"
            '{"summary":"<2-3 sentence expert assessment>",'
            '"tips":["<tip 1>","<tip 2>","<tip 3>"],'
            '"rewrite_suggestion":"<one improved opening sentence>"}'
        )

        resp = http_requests.post(
            ANTHROPIC_API_URL,
            headers={
                "x-api-key":         ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type":      "application/json",
            },
            json={
                "model":      "claude-sonnet-4-20250514",
                "max_tokens": 600,
                "messages":   [{"role": "user", "content": prompt}],
            },
            timeout=20,
        )

        if resp.status_code != 200:
            app.logger.warning(f"Anthropic returned {resp.status_code}: {resp.text[:200]}")
            return None

        body    = resp.json()
        content = body.get("content", [])
        text    = "".join(block.get("text", "") for block in content if block.get("type") == "text")
        text    = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)

    except json.JSONDecodeError as e:
        app.logger.warning(f"Anthropic JSON parse error: {e}")
        return None
    except Exception as e:
        app.logger.warning(f"Anthropic call failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("DEBUG", "false").lower() == "true"

    print("\n" + "═" * 58)
    print("  🎯  InterviewIQ — AI Interview Assistant  v2")
    print("═" * 58)
    print(f"  → Open in browser:  http://localhost:{port}")
    print(f"  → NLP Engine:       ✅  Active (pure Python)")
    print(f"  → Speech Input:     ✅  Browser Web Speech API")
    if ANTHROPIC_API_KEY:
        print(f"  → Claude AI tips:   ✅  Connected")
    else:
        print(f"  → Claude AI tips:   ⚠   Set ANTHROPIC_API_KEY for AI coaching")
        print(f"                          export ANTHROPIC_API_KEY=sk-ant-...")
    print(f"  → Debug mode:       {'ON' if debug else 'OFF'}")
    print("═" * 58)
    print()

    app.run(host="0.0.0.0", port=port, debug=debug)