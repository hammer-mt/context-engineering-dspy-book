"""Write the compact provenance manifest for the final expanded dataset."""

from __future__ import annotations

import csv
from collections import Counter

import yaml

from .dataset import CHAPTER_DIR, DATA_PATH, as_bool, source_family


OUTPUT_PATH = CHAPTER_DIR / "data_sources_expanded.yaml"


def main() -> None:
    with DATA_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    humans = [row for row in rows if not as_bool(row["is_ai"])]
    pair_counts = Counter(row["source_id"] for row in humans)
    representative = {row["source_id"]: row for row in humans}
    sources = []
    for source_id in sorted(representative):
        row = representative[source_id]
        sources.append(
            {
                "id": source_id,
                "family": source_family(row),
                "title": row["source_title"],
                "author": row["source_author"],
                "source_url": row["source_url"],
                "license": row["license"],
                "human_passage_count": pair_counts[source_id],
            }
        )
    manifest = {
        "schema_version": 1,
        "dataset": "data/ai_vs_human_chapter06_expanded.csv",
        "selection_note": (
            "All human passages are real text harvested previously from pre-2022 tagged open-source "
            "documentation or December-2019 Wikipedia revisions. The original 74 rows are preserved. "
            "New AI rows are Sol-generated semantic rewrites selected with baseline-only screening "
            "before the final test lock."
        ),
        "wikipedia_license_url": "https://en.wikipedia.org/wiki/Wikipedia:Copyrights",
        "source_count": len(sources),
        "source_family_count": len({source["family"] for source in sources}),
        "sources": sources,
    }
    OUTPUT_PATH.write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
