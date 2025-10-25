# general preprocessing functions for the data 
import logging
from typing import Tuple
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

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

def __preprocess_for_model(data: pd.DataFrame) -> Tuple[pd.DataFrame]:
    # get features and target
    target_var = 'total_ghg_emissions_tonnes'
    feature_vars = ['state', 'industry_sector', 'reporting_year']
    
    # slice data
    x = data[feature_vars]
    y = data[target_var]
    
     # split into train and test
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42)

    # HANDLE non numeric data 
    # One-hot encode categoricals on TRAIN only
    x_train_d = pd.get_dummies(x_train, drop_first=False)
    
    # Recreate the same columns for TEST
    x_test_d = pd.get_dummies(x_test, drop_first=False)
    x_test_d = x_test_d.reindex(columns=x_train_d.columns, fill_value=0)

    return (x_train_d, x_test_d, y_train, y_test)

def create_linear_model(data: pd.DataFrame) -> Tuple[LinearRegression, float, float]:
    x_train, x_test, y_train, y_test = __preprocess_for_model(data)

    # create and train model
    model = LinearRegression()
    model.fit(x_train, y_train)

    # evaluate model
    preds = model.predict(x_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    logging.info(f"Linear Model - MSE: {mse}, R2: {r2}")

    return (model, mse, r2)

def create_random_forest(data: pd.DataFrame) -> Tuple[RandomForestRegressor, float, float]:
    x_train, x_test, y_train, y_test = __preprocess_for_model(data)

    # create and train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)

    # evaluate model
    preds = model.predict(x_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    logging.info(f"Random Forest Model - MSE: {mse}, R2: {r2}")

    return (model, mse, r2)