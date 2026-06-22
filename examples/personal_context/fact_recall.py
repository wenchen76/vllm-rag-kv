# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fact-recall scorer for personal-context RAG answers.

ROUGE-L / cosine are coarse: they barely move when a single fact flips
("May 16" -> "May 6", "$480" -> "$920", "$1,440" -> "$340"), which is exactly
the failure mode KV reuse can introduce. This scorer extracts the *checkable
atoms* from the gold answer -- money, percentages, times, dates, ID/conf codes,
standalone numbers, and proper nouns -- and reports the fraction that survive
verbatim (modulo formatting) in the generated answer:

    fact_recall = |gold_facts found in generated| / |gold_facts|

It is reference-based (needs gold) and generation-only (no vLLM / logprobs), so
it runs on any generation dump. Matching is normalization-tolerant (case,
"$1,840"=="1840", "$48k"=="$48,000", "9:30 AM"=="9:30am", "16th"=="16") but
value-strict: a flipped number/date does NOT match. Numeric/code/date facts
match on whole-token boundaries (so "16" never matches inside "160"); proper
nouns match as substrings.

Limitations: regex extraction is approximate -- it can miss an oddly-worded fact
or admit a non-fact (e.g. a sentence-initial capitalized word). For the
vanilla-vs-r comparison this mostly cancels: both columns are scored against the
*same* extracted gold facts, so residual extraction noise is shared.
"""

from __future__ import annotations

import re

_MONTHS = (
    r"jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|january|february|"
    r"march|april|june|july|august|september|october|november|december"
)

# Capitalized tokens that are NOT facts on their own: pronouns, articles,
# conjunctions, common sentence-openers, titles, weekdays. Lowercased.
_STOP = {
    "the", "a", "an", "you", "your", "yours", "i", "we", "he", "she", "it",
    "they", "this", "that", "these", "those", "and", "or", "but", "for", "to",
    "of", "in", "on", "at", "by", "with", "as", "if", "so", "no", "yes", "not",
    "is", "are", "was", "were", "be", "been", "will", "would", "should", "can",
    "could", "may", "might", "do", "does", "did", "after", "before", "both",
    "also", "then", "there", "here", "your", "our", "his", "her", "their",
    "dr", "mr", "mrs", "ms", "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday", "mon", "tue", "tues", "wed", "thu", "thur",
    "thurs", "fri", "sat", "sun", "from", "until", "per", "via", "about",
    "am", "pm", "out", "me", "us", "ok", "what", "when", "where", "who",
}

_MONEY = r"[$€£]\s?\d[\d,]*(?:\.\d{1,2})?\s?[kKmM]?"
_PCT = r"\d+(?:\.\d+)?\s?%"
_TIME = r"\d{1,2}:\d{2}\s?(?:[ap]\.?m\.?)?"
_DATE = rf"(?:{_MONTHS})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?"
# token containing BOTH a letter and a digit: PJ-2244, 24V-122, SF-G-19284,
# NH7, UA582, IT-7821, D2740, 5mg, 10mg, 8.4kW...
_CODE = r"#?\b(?=[A-Za-z0-9-]*\d)(?=[A-Za-z0-9-]*[A-Za-z])[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*\b"
_NUM = r"\b\d{2,}\b"
_NAME = r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b"  # 1+ capitalized words


_MONTH_CANON = {
    "january": "jan", "february": "feb", "march": "mar", "april": "apr",
    "june": "jun", "july": "jul", "august": "aug", "september": "sep",
    "sept": "sep", "october": "oct", "november": "nov", "december": "dec",
}


def _normalize(text: str) -> str:
    """Lowercase + canonicalise month/money/time/date surface forms."""
    t = text.lower()
    t = t.replace("a.m.", "am").replace("p.m.", "pm")
    t = re.sub(r"(\d)\s+([ap]m)\b", r"\1\2", t)           # "9:30 am" -> "9:30am"
    t = re.sub(r"\b(\d{1,2})(st|nd|rd|th)\b", r"\1", t)   # "16th" -> "16"
    # "April"/"Sept" -> "apr"/"sep" so abbreviated gold matches spelled-out gen.
    t = re.sub(
        r"\b(" + "|".join(_MONTH_CANON) + r")\b",
        lambda m: _MONTH_CANON[m.group(1)],
        t,
    )
    t = t.replace(",", "")                                # "1,840" -> "1840"
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _expand_money(tok: str) -> str:
    """'$48k' -> '48000', '$1,840' -> '1840', '$2m' -> '2000000'."""
    s = tok.lower().lstrip("$€£").replace(",", "").replace(" ", "")
    m = re.match(r"^(\d+(?:\.\d+)?)([km])?$", s)
    if not m:
        return s
    val = float(m.group(1)) * {"k": 1_000, "m": 1_000_000}.get(m.group(2), 1)
    return str(int(val)) if val == int(val) else str(val)


def extract_facts(gold: str) -> set[tuple[str, str]]:
    """Return a set of ``(kind, value)`` facts; kind in {'token', 'name'}.

    Numeric/temporal spans are blanked as they're matched so a time like
    "9:30" can't also leak a bare "30" into the standalone-number pass.
    """
    facts: set[tuple[str, str]] = set()
    work = gold

    def take(pattern: str, kind: str, transform, flags: int = 0) -> None:
        nonlocal work

        def repl(m: re.Match) -> str:
            facts.add((kind, transform(m.group(0))))
            return " " * len(m.group(0))  # blank the matched span

        work = re.sub(pattern, repl, work, flags=flags)

    take(_MONEY, "token", _expand_money)
    take(_PCT, "token", lambda s: _normalize(s).replace(" ", ""))
    take(_TIME, "token", lambda s: _normalize(s).replace(" ", ""), re.I)
    take(_DATE, "token", _normalize, re.I)
    take(_CODE, "token", lambda s: _normalize(s).lstrip("#"))
    take(_NUM, "token", lambda s: s)

    for m in re.findall(_NAME, work):
        words = m.split()
        kept = [w for w in words if w.lower() not in _STOP]
        if kept:  # drop phrases that are entirely stop-words ("The", "You")
            facts.add(("name", " ".join(kept).lower()))
    return facts


def fact_recall(gold: str, generated: str):
    """Return ``(recall, found, missed)``.

    ``recall`` is None when the gold answer yields no extractable facts.
    ``found`` / ``missed`` are sorted fact-value lists for inspection.
    """
    facts = extract_facts(gold)
    if not facts:
        return None, [], []
    gen = _normalize(generated)
    found, missed = [], []
    for kind, val in facts:
        if kind == "name":
            hit = val in gen
        else:  # whole-token match: "16" must not match inside "160"
            hit = re.search(rf"(?<![a-z0-9]){re.escape(val)}(?![a-z0-9])", gen) \
                is not None
        (found if hit else missed).append(val)
    return len(found) / len(facts), sorted(found), sorted(missed)


if __name__ == "__main__":
    # Self-check on hand-picked (gold, vanilla, r) triples from real dumps,
    # including known factual flips, to confirm the scorer is sensitive where
    # cosine was not.
    CASES = [
        (
            "instance_023 ($1,440/$360 -> r flips to $340/$85; cos rated 0.62)",
            "3-bed cabin in Truckee Jun 21-23, $1,440 total split four ways = "
            "$360 per family. You bring meat + grill, Aunt Pat brings sides, "
            "cousin Sam brings drinks, parents bring dessert.",
            "The Tahoe Airbnb cost is $360 per family, and for the BBQ, ME is "
            "bringing meat and a grill, Pat is bringing sides, Sam is bringing "
            "drinks, and parents are bringing dessert.",
            "The Tahoe Airbnb cost is $340 total, and it's split among 4 "
            "families, which is $85 per family. ME is bringing meat, Pat sides, "
            "Sam drinks, parents dessert.",
        ),
        (
            "instance_007 ($480 -> r flips to $920 deductible)",
            "Bay Plumbing is coming Wed Apr 24, 8-10 AM window. Out of pocket: "
            "you'll pay the $480 estimate yourself.",
            "The plumber is coming on Wednesday, April 24th, between 8:00 and "
            "10:00 AM. You will need to pay the plumber $480 by check.",
            "The plumber is coming on Wed Apr 24, 08:00-10:00. You will pay the "
            "$920 deductible.",
        ),
        (
            "instance_003 (both correct: Cotogna / May 3 / 7:30pm / Priya / Marcus)",
            "Hannah's surprise dinner is Friday May 3, 7:30 PM at Cotogna "
            "(490 Pacific Ave). Priya and Marcus are still pending.",
            "Hannah's surprise birthday dinner is at Cotogna on Friday, May 3, "
            "at 7:30pm. Priya and Marcus are still pending.",
            "Hannah's surprise birthday dinner is at Cotogna on Friday, May 3, "
            "at 7:30pm. Priya and Marcus are still pending.",
        ),
        (
            "instance_046 (solar math, both correct: 32,400 / 9,720 / 22,680)",
            "Sunrun gross cost is $32,400. The 30% federal credit is $9,720, so "
            "net is $22,680.",
            "The total cost of the solar install after the federal credit is "
            "$22,680.",
            "The total cost is $22,680, calculated by subtracting the federal "
            "tax credit of $9,720 from the gross cost of $32,400.",
        ),
    ]
    for name, gold, van, r in CASES:
        rv, fv, mv = fact_recall(gold, van)
        rr, fr, mr = fact_recall(gold, r)
        print(f"\n{name}")
        print(f"  gold facts ({len(extract_facts(gold))}): "
              f"{sorted(v for _, v in extract_facts(gold))}")
        print(f"  vanilla recall = {rv:.2f}   missed={mv}")
        print(f"  r       recall = {rr:.2f}   missed={mr}")
