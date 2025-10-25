import argparse
import sys
import logging
from model.data_process import preprocess_data

VERSION = "0.1.0"

def create_parser():
    parser = argparse.ArgumentParser(prog="mackathon", description="Mackathon CLI")
    parser.add_argument("file", nargs="?", help="path to input data file (e.g. CSV)")
    parser.add_argument("--head", "-n", type=int, default=5, help="print first N rows of the DataFrame")
    parser.add_argument("--version", action="store_true", help="show version and exit")
    return parser

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.version:
        print(VERSION)
        return 0

    if not args.file:
        parser.print_help()
        return 0

    try:
        df = preprocess_data(args.file)
        if df is None:
            print("preprocess_data returned None")
            return 1
        try:
            print(df.head(args.head).to_string(index=False))
        except Exception:
            # fallback if object isn't a pandas DataFrame or .to_string fails
            print(df)
    except Exception as e:
        logging.exception("Failed to preprocess file")
        print(f"Error: {e}")
        return 2

    return 0

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())