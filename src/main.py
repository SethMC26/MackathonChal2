import argparse
import sys
import logging
from model.data_process import create_linear_model, create_random_forest, predict_emissions, preprocess_data

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

        print("linear regression model")
        print(create_linear_model(df))
        
        # Create random forest model and store a copy of it for reference later so we dont recreate it
        model, mse, r2 = create_random_forest(df)
        print(f"Random Forest Model created. MSE: {mse}, R2: {r2}")

        print("Predicting emissions for state='CA', industry_sector='Energy', reporting_year=2020")
        print(predict_emissions(model, 'CA', 'Energy', 2020))

        print("Predicting emissions for state='TX', industry_sector='Transportation', reporting_year=2021")
        print(predict_emissions(model, 'TX', 'Transportation', 2021))

        print("Predicting emissions for state='NY', industry_sector='Residential', reporting_year=2022")
        print(predict_emissions(model, 'NY', 'Residential', 2022))

        print("Predicting emissions for state='FL', industry_sector='Agriculture', reporting_year=2019")
        print(predict_emissions(model, 'FL', 'Agriculture', 2019))

        print("Predicting emissions for state='IL', industry_sector='Manufacturing', reporting_year=2018")
        print(predict_emissions(model, 'IL', 'Manufacturing', 2018))

        print("Predicting emissions for state='WA', industry_sector='Construction', reporting_year=2023")
        print(predict_emissions(model, 'WA', 'Construction', 2023))

        print("Predicting emissions for state='NV', industry_sector='Mining', reporting_year=2024")
        print(predict_emissions(model, 'NV', 'Mining', 2024))

        print("Predicting emissions for state='OH', industry_sector='Healthcare', reporting_year=2015")
        print(predict_emissions(model, 'OH', 'Healthcare', 2015))

        
    except Exception as e:
        logging.exception("Failed to preprocess file")
        print(f"Error: {e}")
        return 2

    return 0

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())