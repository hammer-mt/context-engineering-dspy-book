"""Prepare the surplus real-human candidate pool without making API calls."""

from __future__ import annotations

import json

from .dataset import prepare_candidate_pool


def main() -> None:
    print(json.dumps(prepare_candidate_pool(), indent=2))


if __name__ == "__main__":
    main()
