import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Lasso, Ridge

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score



def preprocess(X, y):
    '''
    Takes in features and target and implements all preprocessing steps for categorical and continuous features returning 
    train and test dataframes with targets
    '''
    
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
    """
    Runs linear regression then returns the R2, MSE and betas
    """
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    
    train_R2 = linreg.score(X_train, y_train)
    test_R2 = linreg.score(X_test, y_test)
    y_train_hat = linreg.predict(X_train)
    y_test_hat = linreg.predict(X_test)
    train_mse = mean_squared_error(y_train, y_train_hat)
    test_mse = mean_squared_error(y_test, y_test_hat)
    coefs = linreg.coef_
    intercept = linreg.intercept_
    
    result_dict = {"training_r^2": train_R2, "mse_train": train_mse, "testing_r^2": test_R2, "mse_test": test_mse}
    
    print(result_dict)
    sns.distplot(y_train)
    sns.distplot(y_train_hat)
    plt.show();
    sns.distplot(y_test)
    sns.distplot(y_test_hat)
    plt.show();
    
    return train_R2, test_R2, train_mse, test_mse, coefs, intercept
    
    
    
def run_lasso(X_train, X_test, y_train, y_test):
    """
    Runs LASSO regression then returns the alphas, R2, MSE and betas
    """
    train_mse = []
    test_mse = []
    alphas = []
    train_R2 = []
    test_R2 = []
    coefs = []
    intercept = []
    
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
        intercept.append(lasso.intercept_)
        
    df_alpha = pd.DataFrame({"alpha":alphas, "training_r^2": train_R2, "mse_train": train_mse, "testing_r^2": test_R2,
                             "mse_test": test_mse, "intercept": intercept, "coefficients": coefs})
    
    return df_alpha.sort_values(by="testing_r^2", ascending = False)




def run_ridge(X_train, X_test, y_train, y_test):
    """
    Runs Ridge regression then returns the alphas, R2, MSE and betas
    """
    train_mse = []
    test_mse = []
    alphas = []
    train_R2 = []
    test_R2 = []
    coefs = []
    intercept = []

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
        intercept.append(ridge.intercept_)

    df_alpha = pd.DataFrame({"alpha":alphas, "training_r^2": train_R2, "mse_train": train_mse, "testing_r^2": test_R2,
                             "mse_test": test_mse, "intercept": intercept, "coefficients": coefs})
    
    return df_alpha.sort_values(by="testing_r^2", ascending = False)



def lasso_coef(lasso_table, X_train, X_test, y_train, y_test):
    """
    Returns the coefs of Lasso model with the highest R^2
    """
    alpha = lasso_table.alpha.iloc[0]
    
    lasso = Lasso(alpha = alpha)
    lasso.fit(X_train, y_train)
    train_preds = lasso.predict(X_train)
    test_preds = lasso.predict(X_test)
    
    result_dict = {"alpha": alpha, "training_r^2": lasso.score(X_train, y_train), "mse_train": mean_squared_error(y_train, train_preds),
                   "testing_r^2": lasso.score(X_test, y_test), "mse_test": mean_squared_error(y_test, test_preds)}
    
    print(result_dict)
    sns.distplot(y_train)
    sns.distplot(train_preds)
    plt.show();
    sns.distplot(y_test)
    sns.distplot(test_preds)
    plt.show();
        
    best_model = pd.DataFrame({"predictor": list(X_train.columns), "coef":lasso.coef_})
    best_model["abs"] = abs(best_model["coef"])
    best_model = best_model.query("abs > 0.00001")
    print("This model has ", len(best_model), " features vs ", len(X_train.columns), " original features")
    print(best_model.drop("abs", axis = 1))



def ridge_coef(ridge_table, X_train, X_test, y_train, y_test):
    """
    Returns the coefs of Ridge model with the highest R^2
    """
    alpha = ridge_table.alpha.iloc[0]
    
    ridge = Ridge(alpha = alpha)
    ridge.fit(X_train, y_train)
    train_preds = ridge.predict(X_train)
    test_preds = ridge.predict(X_test)
    
    result_dict = {"alpha": alpha, "training_r^2": ridge.score(X_train, y_train), "mse_train": mean_squared_error(y_train, train_preds),
                   "testing_r^2": ridge.score(X_test, y_test), "mse_test": mean_squared_error(y_test, test_preds)}
    
    print(result_dict)
    sns.distplot(y_train)
    sns.distplot(train_preds)
    plt.show();
    sns.distplot(y_test)
    sns.distplot(test_preds)
    plt.show();
    
    best_model = pd.DataFrame({"predictor": list(X_train.columns), "coef":ridge_table.coefficients.iloc[0].reshape(len(X_train.columns),)})
    best_model["abs"] = abs(best_model["coef"])
    best_model = best_model.query("abs > 0.00001")
    print("This model has ", len(best_model), " features vs ", len(X_train.columns), " original features")
    print(best_model.drop("abs", axis = 1))



def run_poly(X_train, X_test, y_train, y_test, degree):
    """
    Returns Polynomial regression then returns the R2, MSE
    """
    
    poly=PolynomialFeatures(degree = degree)
    poly_x_train=poly.fit_transform(X_train)

    reg_poly=LinearRegression().fit(poly_x_train,y_train)
    poly_x_test=poly.transform(X_test)

    train_R2 = reg_poly.score(poly_x_train, y_train)
    test_R2 = reg_poly.score(poly_x_test, y_test)
    train_mse = mean_squared_error(y_train, reg_poly.predict(poly_x_train))
    test_mse = mean_squared_error(y_test, reg_poly.predict(poly_x_test))
    coefs = reg_poly.coef_
    intercept = reg_poly.intercept_

    result_dict = {"degree": degree, "training_r^2": train_R2, "mse_train": train_mse,
                   "testing_r^2": test_mse, "mse_test": test_mse}
    
    print(result_dict)
    sns.distplot(y_train)
    sns.distplot(reg_poly.predict(poly_x_train))
    plt.show();
    sns.distplot(y_test)
    sns.distplot(reg_poly.predict(poly_x_test))
    plt.show();