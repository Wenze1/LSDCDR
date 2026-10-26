import argparse
import os

from utils.quick_start import quick_start


os.environ.setdefault("NUMEXPR_MAX_THREADS", "48")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LSDCDR.")
    parser.add_argument("--model", "-m", default="LSDCDR")
    parser.add_argument("--dataset", "-d", default="Amazon2014")
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["Clothing_Shoes_and_Jewelry", "Sports_and_Outdoors"],
    )
    args = parser.parse_args()

    quick_start(
        model=args.model,
        dataset=args.dataset,
        domains=args.domains,
        save_model=True,
    )
