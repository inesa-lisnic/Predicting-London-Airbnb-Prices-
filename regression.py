import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Lasso, Ridge

def preprocess(X, y):
    '''Takes in features and target and implements all preprocessing steps for categorical and continuous features returning 
    train and test dataframes with targets'''
    
    #train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10, train_size = 0.7)
    
    # remove "object"-type features and SalesPrice from `X`
    X_train_cf = X_train[[column for column in X.columns if X[column].dtype != "object"]]
    X_test_cf = X_test[[column for column in X.columns if X[column].dtype != "object"]]
    
    # Scale the train and test data
    stdscaler = StandardScaler()
    stdscaler.fit(X_train_cf)
    
    X_train_scaled = pd.DataFrame(data = stdscaler.transform(X_train_cf), columns = X_train_cf.columns)
    X_test_scaled = pd.DataFrame(data = stdscaler.transform(X_test_cf), columns = X_test_cf.columns)
    
    # Create X_cat which contains only the categorical variables
    X_train_cat = X_train[[column for column in X.columns if X[column].dtype == "object"]]
    X_test_cat = X_test[[column for column in X.columns if X[column].dtype == "object"]]
    
    # OneHotEncode Categorical variables
    enc = OneHotEncoder(handle_unknown='ignore', dtype = "int64")
    enc.fit(X_train_cat)
    X_train_enc = enc.transform(X_train_cat)
    X_test_enc = enc.transform(X_test_cat)
    columns = enc.get_feature_names(input_features=X_train_cat.columns)
    X_train_enc = pd.DataFrame(X_train_enc.todense(), columns=columns)
    X_test_enc = pd.DataFrame(X_test_enc.todense(), columns=columns)
    
    # combine categorical and continuous features into the final dataframe
    X_train_all = pd.concat([X_train_scaled, X_train_enc], axis = 1)
    X_test_all = pd.concat([X_test_scaled, X_test_enc], axis = 1)
    
    return X_train_all, X_test_all, y_train, y_test



def run_linear(X_train, X_test, y_train, y_test):
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    
    train_R2 = linreg.score(X_train, y_train)
    test_R2 = linreg.score(X_test, y_test)
    train_mse = mean_squared_error(y_train, linreg.predict(X_train))
    test_mse = mean_squared_error(y_test, linreg.predict(X_test))
    coefs = linreg.coef_
    intercept = linreg.intercept_
    
    return train_R2, test_R2, train_mse, test_mse, coefs, intercept
    
def run_lasso(X_train, X_test, y_train, y_test):
    train_mse = []
    test_mse = []
    alphas = []
    train_R2 = []
    test_R2 = []
    coefs = []

    for alpha in np.linspace(0, 40, num=200):
        lasso = Lasso(alpha = alpha)
        lasso.fit(X_train, y_train)

        train_preds = lasso.predict(X_train)
        train_R2.append(lasso.score(X_train, y_train))
        train_mse.append(mean_squared_error(y_train, train_preds))

        test_preds = lasso.predict(X_test)
        test_R2.append(lasso.score(X_test, y_test))
        test_mse.append(mean_squared_error(y_test, test_preds))

        alphas.append(alpha)
        coefs.append(lasso.coef_)

    df_alpha = pd.DataFrame({"alpha":alphas, "Training_r^2": train_R2, "MSE_train": train_mse, "Testing_r^2": test_R2,
                             "MSE_test": test_mse, "Coefficients": coefs})
    return df_alpha.sort_values(by="Training_r^2", ascending = False)


def run_ridge(X_train, X_test, y_train, y_test):
    train_mse = []
    test_mse = []
    alphas = []
    train_R2 = []
    test_R2 = []
    coefs = []

    for alpha in np.linspace(0, 40, num=200):
        ridge = Ridge(alpha = alpha)
        ridge.fit(X_train, y_train)

        train_preds = ridge.predict(X_train)
        train_R2.append(ridge.score(X_train, y_train))
        train_mse.append(mean_squared_error(y_train, train_preds))

        test_preds = ridge.predict(X_test)
        test_R2.append(ridge.score(X_test, y_test))
        test_mse.append(mean_squared_error(y_test, test_preds))

        alphas.append(alpha)
        coefs.append(ridge.coef_)

    df_alpha = pd.DataFrame({"alpha":alphas, "Training_r^2": train_R2, "MSE_train": train_mse, "Testing_r^2": test_R2,
                             "MSE_test": test_mse, "Coefficients": coefs})
    return df_alpha.sort_values(by="Training_r^2", ascending = False)
