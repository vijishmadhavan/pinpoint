"""Prepare a benchmark manifest and starter query set for the 10k legal-text corpus."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

HEADER_PATTERNS = {
    "case_id": r"Case ID:\s*([^\n]+)",
    "title": r"Title:\s*([^\n]+)",
    "parties": r"Parties:\s*([^\n]+)",
    "date": r"Date:\s*([^\n]+)",
    "law_area": r"Law Area:\s*([^\n]+)",
    "sections": r"Sections:\s*([^\n]+)",
    "other_citation": r"Other Citation:\s*([^\n]+)",
    "case_number": r"Case Number:\s*([^\n]+)",
}

TOPIC_PATTERNS = [
    "section 138",
    "bail",
    "cenvat",
    "refund",
    "penalty",
    "classification",
    "service tax",
    "income tax",
    "customs",
]


def parse_case(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    parsed: dict[str, Any] = {
        "file": path.name,
        "path": str(path),
        "size": len(text),
        "preview": text[:500].replace("\n", " ").strip(),
    }
    for key, pattern in HEADER_PATTERNS.items():
        match = re.search(pattern, text)
        parsed[key] = match.group(1).strip() if match else ""

    lowered = text.lower()
    parsed["topics"] = [topic for topic in TOPIC_PATTERNS if topic in lowered]
    parsed["has_versus"] = "versus" in lowered
    return parsed


def build_manifest(corpus_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = []
    law_areas = Counter()
    topics = Counter()

    for path in sorted(corpus_path.glob("*.txt")):
        case = parse_case(path)
        manifest.append(case)
        if case["law_area"]:
            law_areas[case["law_area"]] += 1
        topics.update(case["topics"])

    summary = {
        "total_files": len(manifest),
        "law_areas": law_areas.most_common(),
        "topics": topics.most_common(),
    }
    return manifest, summary


def choose_starter_cases(manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_topic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in manifest:
        for topic in case["topics"]:
            by_topic[topic].append(case)

    starters: list[dict[str, Any]] = []
    desired_topics = ["section 138", "bail", "cenvat", "refund", "penalty", "service tax", "customs", "income tax"]
    seen_files = set()
    for topic in desired_topics:
        for case in by_topic.get(topic, []):
            if case["file"] in seen_files:
                continue
            starters.append(case)
            seen_files.add(case["file"])
            break
    return starters


def choose_expanded_cases(manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    desired = {
        "section 138": 3,
        "bail": 3,
        "cenvat": 3,
        "refund": 3,
        "penalty": 3,
        "service tax": 3,
        "customs": 3,
        "income tax": 4,
    }
    selected: list[dict[str, Any]] = []
    seen_files = set()
    for topic, target in desired.items():
        count = 0
        for case in manifest:
            if topic not in case["topics"]:
                continue
            if case["file"] in seen_files:
                continue
            selected.append(case)
            seen_files.add(case["file"])
            count += 1
            if count >= target:
                break
    return selected


def build_starter_queries(cases: list[dict[str, Any]]) -> dict[str, Any]:
    queries = []
    for case in cases:
        title = case.get("title", "")
        parties = case.get("parties", "")
        law_area = case.get("law_area", "")
        topics = case.get("topics", [])
        topic = topics[0] if topics else law_area.lower()
        slug = case["file"].replace(".txt", "")

        if topic == "section 138":
            query = "cheque dishonour section 138 case"
        elif topic == "bail":
            query = f"regular bail {law_area.lower()} enforcement case"
        elif topic == "cenvat":
            query = "cenvat credit dispute case"
        elif topic == "refund":
            query = "refund claim dispute case"
        elif topic == "penalty":
            query = "penalty dispute case"
        elif topic == "service tax":
            query = "service tax dispute case"
        elif topic == "customs":
            query = "customs dispute case"
        elif topic == "income tax":
            query = "income tax appeal case"
        else:
            query = f"{law_area.lower()} case"

        if parties:
            party_query = parties.split("VERSUS")[0].strip().replace("& ANR", "").replace("& ANR.", "").strip()
            if party_query:
                query = f"{query} {party_query}"

        queries.append(
            {
                "id": f"starter-{slug}",
                "query": query.lower(),
                "relevant": [case["file"]],
                "preferred_top": case["file"],
                "notes": f"title={title}; law_area={law_area}; topics={', '.join(topics)}",
            }
        )

    return {
        "name": "pinpoint-search-relevance-v3-legal-starter",
        "version": 1,
        "notes": [
            "Starter benchmark generated from the real legal corpus.",
            "These queries should be reviewed and expanded with harder paraphrase and ambiguity cases."
        ],
        "queries": queries,
    }


def choose_subset_cases(manifest: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Build a representative but overlap-heavy subset capped at `limit` files."""
    topic_targets = [
        ("income tax", 160),
        ("penalty", 70),
        ("refund", 60),
        ("cenvat", 60),
        ("service tax", 60),
        ("customs", 60),
        ("bail", 15),
        ("section 138", 15),
    ]
    selected: list[dict[str, Any]] = []
    seen = set()

    for topic, target in topic_targets:
        count = 0
        for case in manifest:
            if len(selected) >= limit:
                return selected
            if topic not in case["topics"]:
                continue
            if case["file"] in seen:
                continue
            selected.append(case)
            seen.add(case["file"])
            count += 1
            if count >= target:
                break

    if len(selected) < limit:
        for case in manifest:
            if len(selected) >= limit:
                break
            if case["file"] in seen:
                continue
            selected.append(case)
            seen.add(case["file"])

    return selected


def _query_templates(topic: str, case: dict[str, Any]) -> list[str]:
    law_area = case.get("law_area", "").lower()
    parties = case.get("parties", "")
    lead_party = parties.split("VERSUS")[0].strip().replace("& ANR", "").replace("& ANR.", "").strip()

    if topic == "section 138":
        templates = [
            "cheque dishonour section 138 case",
            "cheque bounce complaint section 138",
        ]
    elif topic == "bail":
        templates = [
            f"regular bail {law_area} case",
            "bail in enforcement matter",
        ]
    elif topic == "cenvat":
        templates = [
            "cenvat credit dispute",
            "input credit issue in excise or service tax",
        ]
    elif topic == "refund":
        templates = [
            "refund claim dispute",
            "refund denied because of procedure",
        ]
    elif topic == "penalty":
        templates = [
            "penalty dispute",
            "can penalty be set aside case",
        ]
    elif topic == "service tax":
        templates = [
            "service tax demand dispute",
            "taxable service issue case",
        ]
    elif topic == "customs":
        templates = [
            "customs dispute case",
            "import duty or customs issue",
        ]
    elif topic == "income tax":
        templates = [
            "income tax appeal case",
            "tax addition disallowance dispute",
        ]
    else:
        templates = [f"{law_area} case"]

    if lead_party:
        templates.append(f"{templates[0]} {lead_party.lower()}")
    return templates


def build_expanded_queries(cases: list[dict[str, Any]]) -> dict[str, Any]:
    queries = []
    for case in cases:
        topics = case.get("topics", [])
        topic = topics[0] if topics else case.get("law_area", "").lower()
        for idx, query in enumerate(_query_templates(topic, case), start=1):
            queries.append(
                {
                    "id": f"{case['file'].replace('.txt', '')}-{idx}",
                    "query": query,
                    "relevant": [case["file"]],
                    "preferred_top": case["file"],
                    "notes": (
                        f"title={case.get('title', '')}; law_area={case.get('law_area', '')}; "
                        f"topics={', '.join(topics)}"
                    ),
                }
            )

    return {
        "name": "pinpoint-search-relevance-v3-legal-expanded",
        "version": 1,
        "notes": [
            "Expanded legal benchmark generated from the real 10k corpus.",
            "Queries are starter heuristics and should be refined with manual judgments for harder ambiguity cases."
        ],
        "queries": queries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build manifest and starter benchmark files for the legal 10k corpus.")
    parser.add_argument("--corpus", required=True, help="Path to corpus folder")
    parser.add_argument("--out-dir", default="benchmarks/v3_legal", help="Output directory")
    parser.add_argument("--subset-limit", type=int, default=500, help="Max files for the representative subset")
    args = parser.parse_args()

    corpus = Path(args.corpus)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest, summary = build_manifest(corpus)
    starter_cases = choose_starter_cases(manifest)
    expanded_cases = choose_expanded_cases(manifest)
    subset_cases = choose_subset_cases(manifest, args.subset_limit)
    starter = build_starter_queries(starter_cases)
    expanded = build_expanded_queries(expanded_cases)

    (out_dir / "manifest.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=True) for item in manifest),
        encoding="utf-8",
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "starter_queries.json").write_text(json.dumps(starter, indent=2), encoding="utf-8")
    (out_dir / "expanded_queries.json").write_text(json.dumps(expanded, indent=2), encoding="utf-8")
    (out_dir / "subset_files.txt").write_text(
        "\n".join(case["path"] for case in subset_cases),
        encoding="utf-8",
    )
    (out_dir / "subset_summary.json").write_text(
        json.dumps(
            {
                "subset_limit": args.subset_limit,
                "selected_files": len(subset_cases),
                "law_areas": Counter(case["law_area"] for case in subset_cases if case["law_area"]).most_common(),
                "topics": Counter(topic for case in subset_cases for topic in case["topics"]).most_common(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Manifest written: {out_dir / 'manifest.jsonl'}")
    print(f"Summary written: {out_dir / 'summary.json'}")
    print(f"Starter queries written: {out_dir / 'starter_queries.json'}")
    print(f"Expanded queries written: {out_dir / 'expanded_queries.json'}")
    print(f"Subset files written: {out_dir / 'subset_files.txt'}")
    print(f"Subset summary written: {out_dir / 'subset_summary.json'}")
    print(f"Total files: {summary['total_files']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
