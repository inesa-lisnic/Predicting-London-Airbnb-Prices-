## Predicting London Airbnb Prices

### Module 4 Project

### Team Members
        Grace Valmadrid
        Inesa Lisnic

### Executive Summary

Airbnb is an online marketplace for people who want to rent their property for lodging. The purpose of this project is to investigate the London Airbnb market in order to help the hosts in estimating the rent price of their property and also to increase their income through improving the factors that greatly influence the rent price. Data is taken from http://insideairbnb.com, an independent data prrovider for Airbnb.  There are 85,273 listings as of September 2019.

The following regression methods were uses in price modelling:  Linear, Polynomial, Ridge and LASSO.

LASSO's best model resulted to an R^2 of 38% with only 50 features. The main features that affect the rent price the most are the following: 
        Neighbourhoods like Westminster, Kensington and Chelsea and the City of London increase the rent by £51, £43 and £18, respectively;
        Private rooms decreases it by £34 while an entire home/apartment type increases it by only £3;
        Boutique hotels and service apartments increase the rent by £27 and £18, respectively;
        Every additional guest a property can accommodate provides additional income of £14;
        An additional bedroom increases the price by £13.

Recommendations:
        Increase capacity and number of beds and amenities offered;
        Encourage guests to write good reviews;
        Maintain cleanliness and improve accuracy, responsiveness and check-in process.

### Files

Main notebook:
        https://github.com/inesa-lisnic/Predicting-London-Airbnb-Prices-/blob/master/london_airbnb.ipynb

Custom functions for data cleaning, EDA and modelling
        https://github.com/inesa-lisnic/Predicting-London-Airbnb-Prices-/blob/master/eda.py
        https://github.com/inesa-lisnic/Predicting-London-Airbnb-Prices-/blob/master/regression.py

Linear and Polynomial models ran by Inesa
        https://github.com/inesa-lisnic/Predicting-London-Airbnb-Prices-/blob/master/polynom%20regress.zip
