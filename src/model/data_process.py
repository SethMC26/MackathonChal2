# general preprocessing functions for the data 
import logging
from typing import Tuple, Optional
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from typing import Tuple
import math

dataframe = None

def preprocess_data(file: str) -> pd.DataFrame:
    global dataframe
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

def __preprocess_for_model(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    # get features and target
    target_var = 'total_ghg_emissions_tonnes'
    feature_vars = ['state', 'industry_sector', 'latitude', 'longitude', 'reporting_year']

    # slice data
    x = data[feature_vars]
    y = data[target_var]
    
    # split into train and test
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    # One-hot encode categoricals
    x_train_d = pd.get_dummies(x_train, drop_first=False)
    x_test_d = pd.get_dummies(x_test, drop_first=False)
    x_test_d = x_test_d.reindex(columns=x_train_d.columns, fill_value=0)

    return (x_train_d, x_test_d, y_train, y_test)

def create_linear_regression(data: pd.DataFrame) -> Tuple[LinearRegression, float, float]:
    x_train, x_test, y_train, y_test = __preprocess_for_model(data)

    # create and train model
    model = LinearRegression()
    model.fit(x_train, y_train)

    # evaluate model
    preds = model.predict(x_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    logging.info(f"Linear Model - MSE: {mse}, R2: {r2}")
    
    predsT = model.predict(x_train)
    mseT = mean_squared_error(y_train, predsT)
    r2T = r2_score(y_train, predsT)
    logging.info(f"Linear Model - MSE(train): {mseT}, R2(train): {r2T}")   

    return (model, mse, r2)

def linear_predict_emissions(model: LinearRegression, state: str, industry_sector: str, reporting_year: int, dataframe: pd.DataFrame) -> float:
    state_centroids = dataframe.groupby('state')[['latitude', 'longitude']].mean()

    input_df = pd.DataFrame([{
        'state': state,
        'industry_sector': industry_sector,
        'reporting_year': reporting_year,
        'latitude': state_centroids.loc[state, 'latitude'],
        'longitude': state_centroids.loc[state, 'longitude']
    }])

    # One-hot encode the input data
    input_d = pd.get_dummies(input_df, drop_first=False)

    # Align input columns with model training columns
    model_features = model.feature_names_in_
    input_d = input_d.reindex(columns=model_features, fill_value=0)

    prediction = model.predict(input_d)
    return prediction[0]

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
    
    predsT = model.predict(x_train)
    mseT = mean_squared_error(y_train, predsT)
    r2T = r2_score(y_train, predsT)
    logging.info(f"Random Forest Model - MSE(train): {mseT}, R2(train): {r2T}")    

    return (model, mse, r2)

def random_forest_predict_emissions(model: RandomForestRegressor, state: str, industry_sector: str, reporting_year: int, dataframe: pd.DataFrame) -> float:    
    state_centroids = dataframe.groupby('state')[['latitude', 'longitude']].mean()
    
    feature_vars = ['state', 'industry_sector', 'latitude', 'longitude', 'reporting_year']
    df_columns = pd.get_dummies(dataframe[feature_vars], drop_first=False).columns
    
    input_df = pd.DataFrame([{
        'state': state,
        'industry_sector': industry_sector,
        'reporting_year': reporting_year,
        'latitude': state_centroids.loc[state, 'latitude'],
        'longitude': state_centroids.loc[state, 'longitude']
    }])    

    # One-hot encode the input data
    input_d = pd.get_dummies(input_df, drop_first=False)
    input_d = input_d.reindex(columns=df_columns, fill_value=0)

    # Align input columns with model training columns
    model_features = model.feature_names_in_
    # input_d = input_d.reindex(columns=model_features, fill_value=0)

    prediction = model.predict(input_d)
    return prediction[0]

def evaluate_model(model, data: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
    """
    Evaluate a trained sklearn model on a holdout test split derived from `data`.
    Returns (mse, r2) or (None, None) on failure.
    """
    try:
        x_train_d, x_test_d, y_train, y_test = __preprocess_for_model(data)
        # align test matrix with model's expected features if possible
        if hasattr(model, "feature_names_in_"):
            features = list(model.feature_names_in_)
            x_test_d = x_test_d.reindex(columns=features, fill_value=0)
        preds = model.predict(x_test_d)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        return mse, r2
    except Exception:
        logging.exception("evaluate_model failed")
        return None, None