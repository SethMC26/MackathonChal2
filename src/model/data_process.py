# general preprocessing functions for the data 
import logging
import pandas as pd

def preprocess_data(file: str) -> pd.DataFrame:
    try:
        # pass the path directly to pandas
        dataframe = pd.read_csv(file)
        return dataframe
    except FileNotFoundError:
        logging.warning("preprocess_data: file not found: %s", file)
        return None
    except PermissionError:
        logging.warning("preprocess_data: permission denied: %s", file)
        return None
    except Exception:
        logging.exception("preprocess_data: failed to read/parse file: %s", file)
        return None


    