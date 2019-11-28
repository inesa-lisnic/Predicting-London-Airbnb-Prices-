## Predicting London Airbnb Prices

### Module 4 Project

### Team Members
        Grace Valmadrid
        Inesa Lisnic

### Goals

The purpose of this project is to investigate the London Airbnb market in order to help Airbnb hosts (and potential hosts) maximise their income and improve their ratings.

Data is taken from http://insideairbnb.com, an independent data prrovider for Airbnb.  

### Responsibilities

Workflow consists of 4 parts: Data Cleaning, Exploratory Data Analysis, Building Regression Model, Making the recommendations for Airbnb hosts.  Both of the team members wor

At Data Cleaning stage we dealt with null values, outliers and duplicate columns. Null or zero values were either replaced with mean or mode. 

The outliers were removed according to their positions on the frequency distribution, only those within zcore +/3 were retained.

At EDA stage we investigated the relationship between prices and other features using scatterplots and correlation matrices.

For the entire dataset, we built 4 regression models: Linear, Polynomial, Ridge and LASSO. Then we created subsets based on property type ('room_type'). We ran models for the 2 main property types: Homes/Apartments and  Private Rooms. 

At the last step we made our recommendations for the Airbnb hosts based on our results from regression models and EDA.

In order to optimize our work, we used a .py file where we built and stored the functions for regression models and for dealing with null values which are called in our Jupyter Notebook file.

### Files

Main notebook:
https://github.com/inesa-lisnic/Predicting-London-Airbnb-Prices-/blob/master/london_airbnb.ipynb

Custom functions for data cleaning, EDA and modelling
https://github.com/inesa-lisnic/Predicting-London-Airbnb-Prices-/blob/master/eda.py
https://github.com/inesa-lisnic/Predicting-London-Airbnb-Prices-/blob/master/regression.py

Linear and Polynomial models ran by Inesa
https://github.com/inesa-lisnic/Predicting-London-Airbnb-Prices-/blob/master/polynom%20regress.zip
