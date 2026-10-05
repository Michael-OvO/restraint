"""Export facebook/AbstentionBench to JSONL. Runs in its OWN venv (python 3.11, datasets==3.6.0); no honest_agents.

The HF dataset is a loading script (`AbstentionBench.py`) that builds 20 source datasets (downloads from many hosts) and
applies their subsampling indices; `datasets>=4` no longer executes loading scripts, hence the separate venv.

    python abstentionbench_export_standalone.py --out abstentionbench.jsonl

Row: {dataset, index, question, reference_answers, should_abstain, metadata_json}. License: CC BY-NC 4.0.
"""

from __future__ import annotations

import argparse
import json


def main() -> int:
    import datasets

    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--repo", default="facebook/AbstentionBench")
    args = p.parse_args()

    builder = datasets.load_dataset_builder(args.repo, trust_remote_code=True)
    builder.download_and_prepare()  # AbstentionBench.download_and_prepare fills builder.datasets: {name: Dataset}
    n = 0
    with open(args.out, "w") as f:
        for name, ds in builder.datasets.items():
            for i, row in enumerate(ds):
                f.write(
                    json.dumps(
                        {
                            "dataset": name,
                            "index": i,
                            "question": row["question"],
                            "reference_answers": row["reference_answers"],
                            "should_abstain": bool(row["should_abstain"]),
                            "metadata_json": row["metadata_json"],
                        }
                    )
                    + "\n"
                )
                n += 1
            print(f"{name}: {len(ds)}")
    print(f"wrote {n} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
