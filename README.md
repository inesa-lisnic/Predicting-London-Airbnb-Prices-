The purpose of this project is to investigate the London Airbnb market in order to help Airbnb hosts (and potential hosts) to increase their income based on their activity on this platform.

As a source we used http://insideairbnb.com from where we downloaded the London Airbnb database which was scrapped in September 2019. 

Our workflow consists of 4 parts: Data Cleaning, Exploratory Data Analysis, Creating the Regression Model, Making the recommendations for Airbnb hosts.

At Data Cleaning stage we dealt with null values,outliers and duplicates. We removed the rows with null values or replaced them with their column mean or mode with the condition to not affect the entire data.
The outliers were removed according to their positions on the frequency distribution, so we kept just the values corresponding with the zcore<|3|.
Also we removed columns that were identical in terms of values.

At EDA stage we investigated the relationship between prices and other parameters from the dataframe, plotting scatters and analysing their patterns.

For Regression Model we defined our target as the price values and checked his correlation with other numerical predictors, defining the top high correlated predictors. Then we checked collinearity between the main predictors and others from the database, removing those ‘secondary’ predictors with high correlation with the main predictors.

We created 3 regression models: Linear Regression model, Lasso Model and Ridge Model.
Then we sliced our database according to the properties types in 4 databases. 
We chose the datasets of the most representative two types of properties: Homes/Apartments and  Private Rooms. After this we made 3 models mentioned above for each of these 2 datasets having the same target - the price.

At the last step we made our recommendations for the Airbnb hosts based on our results from regression models and EDA.

In order to optimize our work, we used a .py file where we built and stored the functions for regression models and for dealing with null values which are called in our Jupyter Notebook file.

