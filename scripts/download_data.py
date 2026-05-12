"""Download / validate datasets using mirdata.

Usage:
    python scripts/download_data.py --dataset saraga --data-home data/raw/saraga
    python scripts/download_data.py --dataset maestro --data-home data/raw/maestro
    python scripts/download_data.py --dataset gtzan   --data-home data/raw/gtzan
"""

import argparse
import mirdata

DATASETS = {
    "saraga": "saraga_carnatic",
    "maestro": "maestro",
    "gtzan": "gtzan_genre",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASETS), required=True)
    parser.add_argument("--data-home", required=True)
    parser.add_argument("--partial", action="store_true", help="Download only annotations (no audio)")
    args = parser.parse_args()

    key = DATASETS[args.dataset]
    dataset = mirdata.initialize(key, data_home=args.data_home)

    if args.partial:
        dataset.download(partial_download=["annotations"])
    else:
        dataset.download()

    print(f"Validating {key}...")
    dataset.validate()
    print("Done.")


if __name__ == "__main__":
    main()
