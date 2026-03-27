#!/usr/bin/env python3
"""
cli_analyzer.py
Standalone CLI tool — runs NLP analysis directly in your terminal.
Usage:  python cli_analyzer.py
        python cli_analyzer.py --demo
"""

import sys
import argparse
import textwrap
from colorama import Fore, Back, Style, init as colorama_init
from nlp_engine import analyze_response

colorama_init(autoreset=True)

# ─────────────────────────────────────────────────────────
# COLOR HELPERS
# ─────────────────────────────────────────────────────────

def c(text, color): return f"{color}{text}{Style.RESET_ALL}"
def bold(text):     return f"{Style.BRIGHT}{text}{Style.RESET_ALL}"

def score_color(s):
    if s >= 80: return Fore.GREEN
    if s >= 60: return Fore.CYAN
    if s >= 40: return Fore.YELLOW
    return Fore.RED

def bar(score, width=30):
    filled = int((score / 100) * width)
    col = score_color(score)
    return col + "█" * filled + Fore.WHITE + "░" * (width - filled)

def divider(char="═", width=62):
    return Fore.CYAN + char * width + Style.RESET_ALL

# ─────────────────────────────────────────────────────────
# PRINT REPORT
# ─────────────────────────────────────────────────────────

def print_report(result: dict):
    v = result["verdict"]
    s = result["scores"]
    stats = result["stats"]
    kw = result["keywords"]
    tips = result["tips"]
    fillers = result["fillers"]
    star = result["star_method"]
    quant = result["quantifiers"]

    print()
    print(divider())
    print(bold(c("  🎯  INTERVIEWIQ — ANALYSIS REPORT", Fore.CYAN)))
    print(divider())

    # ── Overall ──────────────────────────────────────────
    ov_col = score_color(result["overall"])
    print()
    print(f"  {bold('OVERALL SCORE')}  {ov_col}{bold(str(result['overall']))}/100{Style.RESET_ALL}   "
          f"{c(v['emoji'] + ' ' + v['label'].upper(), ov_col)}")
    print(f"  {c(v['message'], Fore.WHITE)}")
    print()

    # ── Stats ─────────────────────────────────────────────
    print(divider("─"))
    print(f"  {c('RESPONSE STATS', Fore.CYAN)}")
    print(divider("─"))
    print(f"  Words: {bold(str(stats['word_count']))}   "
          f"Sentences: {bold(str(stats['sentence_count']))}   "
          f"Avg Sentence: {bold(str(stats['avg_sentence_len']))} words   "
          f"Grade Level: {bold(str(stats['grade_level']))}   "
          f"Speak Time: ~{bold(str(stats['reading_time_sec']))}s")
    print()

    # ── 4 Scores ──────────────────────────────────────────
    print(divider("─"))
    print(f"  {c('DIMENSION SCORES', Fore.CYAN)}")
    print(divider("─"))
    for label, key in [("Confidence", "confidence"), ("Clarity", "clarity"),
                        ("Relevance", "relevance"), ("Depth", "depth")]:
        sc = s[key]
        print(f"  {label:<14} {bar(sc, 28)}  {score_color(sc)}{bold(str(sc)):>3}{Style.RESET_ALL}/100")
    print()

    # ── Keywords ──────────────────────────────────────────
    print(divider("─"))
    print(f"  {c('KEYWORD ANALYSIS', Fore.CYAN)}")
    print(divider("─"))
    if kw["found"]:
        found_str = "  " + "  ".join(c(f"[✓ {k}]", Fore.GREEN) for k in kw["found"])
        print("  FOUND:    " + "  ".join(c(f"✓ {k}", Fore.GREEN) for k in kw["found"]))
    if kw["missing"]:
        print("  MISSING:  " + "  ".join(c(f"✗ {k}", Fore.RED) for k in kw["missing"]))
    if kw["suggested"]:
        print("  SUGGEST:  " + "  ".join(c(f"+ {k}", Fore.CYAN) for k in kw["suggested"]))
    print()

    # ── STAR Method ───────────────────────────────────────
    print(divider("─"))
    print(f"  {c('STAR METHOD ANALYSIS', Fore.CYAN)}")
    print(divider("─"))
    for comp, data in star["components"].items():
        icon = c("●", Fore.GREEN) if data["found"] else c("○", Fore.RED)
        status = c("FOUND", Fore.GREEN) if data["found"] else c("MISSING", Fore.RED)
        print(f"  {icon} {comp.upper():<12} {status}"
              + (f"  → '{data['triggers'][0]}'" if data.get("triggers") else ""))
    complete_str = c("✓ COMPLETE", Fore.GREEN) if star["complete"] else c("✗ INCOMPLETE (aim for 3+ components)", Fore.YELLOW)
    print(f"  Structure: {complete_str}")
    print()

    # ── Filler Words ──────────────────────────────────────
    print(divider("─"))
    print(f"  {c('LANGUAGE QUALITY', Fore.CYAN)}")
    print(divider("─"))
    if fillers["total"] == 0:
        print(f"  Filler words:  {c('None detected ✓', Fore.GREEN)}")
    else:
        filler_detail = ", ".join(f'"{w}"×{n}' for w, n in fillers["found"].items())
        print(f"  Filler words:  {c(str(fillers['total']) + ' total', Fore.YELLOW)} — {filler_detail}")
    if quant["count"] > 0:
        print(f"  Quantifiers:   {c(str(quant['count']) + ' found', Fore.GREEN)} — {', '.join(str(x) for x in quant['found'][:5])}")
    else:
        print(f"  Quantifiers:   {c('None — add numbers/percentages for impact', Fore.YELLOW)}")
    if result.get("power_verbs"):
        print(f"  Power verbs:   {c(', '.join(result['power_verbs'][:6]), Fore.CYAN)}")
    print()

    # ── Strengths ─────────────────────────────────────────
    print(divider("─"))
    print(f"  {c('STRENGTHS', Fore.GREEN)}")
    print(divider("─"))
    for tip in (tips["strengths"] or ["No clear strengths identified."]):
        print(f"  {c('✓', Fore.GREEN)} " + textwrap.fill(tip, width=58, subsequent_indent="    "))
    print()

    # ── Improvements ──────────────────────────────────────
    print(divider("─"))
    print(f"  {c('IMPROVEMENTS', Fore.YELLOW)}")
    print(divider("─"))
    for tip in (tips["improvements"] or ["No major improvements needed!"]):
        print(f"  {c('!', Fore.YELLOW)} " + textwrap.fill(tip, width=58, subsequent_indent="    "))
    print()
    print(divider())
    print()


# ─────────────────────────────────────────────────────────
# DEMO DATA
# ─────────────────────────────────────────────────────────

DEMO = {
    "question": "Tell me about a time you demonstrated leadership under pressure.",
    "category": "behavioral",
    "response": (
        "In my previous role as an engineering lead, we were three weeks from a major product launch "
        "when our lead backend developer unexpectedly resigned. I immediately assessed the situation "
        "and held a team meeting to redistribute responsibilities based on each engineer's strengths. "
        "I personally took on the critical API integration work while coordinating with our product "
        "manager to de-scope lower-priority features. I implemented daily standups and a shared Jira "
        "board so everyone had full visibility into progress. As a result, we delivered the launch on "
        "schedule, and the product achieved 40% higher adoption than our previous release. This "
        "experience taught me that transparent communication and rapid role clarity are the most "
        "powerful tools when navigating unexpected crises."
    )
}

# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="InterviewIQ CLI Analyzer")
    parser.add_argument("--demo", action="store_true", help="Run with a built-in demo response")
    args = parser.parse_args()

    print()
    print(divider())
    print(bold(c("  🎯  INTERVIEWIQ — AI Interview Assistant (CLI)", Fore.CYAN)))
    print(c("  NLP Analysis Engine · Pure Python · No ML Dependencies", Fore.WHITE))
    print(divider())

    if args.demo:
        print(f"\n  {c('DEMO MODE', Fore.YELLOW)} — Using sample behavioral answer\n")
        question = DEMO["question"]
        category = DEMO["category"]
        response = DEMO["response"]
        print(f"  {c('Question:', Fore.CYAN)} {question}")
        print(f"  {c('Category:', Fore.CYAN)} {category}")
        print(f"\n  {c('Response:', Fore.CYAN)}")
        for line in textwrap.wrap(response, width=60):
            print(f"    {line}")
        print()
    else:
        print(f"\n  {c('CATEGORIES:', Fore.CYAN)} behavioral | technical | situational | common\n")
        category = input(f"  {c('Category', Fore.CYAN)} [{c('common', Fore.WHITE)}]: ").strip() or "common"
        if category not in ("behavioral", "technical", "situational", "common"):
            print(f"  {c('Invalid category, defaulting to common.', Fore.YELLOW)}")
            category = "common"
        print()
        question = input(f"  {c('Question:', Fore.CYAN)} ").strip()
        if not question:
            question = "Tell me about yourself."
        print(f"\n  {c('Your response', Fore.CYAN)} (paste/type — press ENTER twice when done):\n")
        lines = []
        while True:
            try:
                line = input()
                if line == "" and lines and lines[-1] == "":
                    break
                lines.append(line)
            except EOFError:
                break
        response = "\n".join(lines).strip()

    if not response:
        print(c("  No response provided. Exiting.", Fore.RED))
        sys.exit(1)

    print(f"\n  {c('Analyzing response…', Fore.CYAN)}\n")
    result = analyze_response(response, question, category)
    print_report(result)


if __name__ == "__main__":
    main()
