# NYC_airbnb_proj  
* The dataset describes the listing activity and metrics in NYC, NY for 2019 which is from [Kaggle](https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data).
* Created Linear, Lasso, Random Forest, and Decision Tree Regressors to predict the rental price.
* Compare these models and reach the best model.

## Code and Resources Used 
**Dataset from :** https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data  
**Python Version :** Python 3.7.4  
**IDE :** Spyder, Jupyter Notebook  
**Packages :** pandas, numpy, matplotlib, seaborn, dataframe_image, WordCloud, sklearn  
**Reference websites :**
1. https://www.kaggle.com/duygut/airbnb-nyc-price-prediction  
2. https://seaborn.pydata.org/generated/seaborn.scatterplot.html
3. https://blog.csdn.net/zyb228/article/details/100897145
4. https://www.kaggle.com/shotashimizu/09-decisiontree-gridsearchcv

## Data info 
We got about 49000 records and 16 columns:
*	id
*	name
*	host_id
*	host_name
*	neighbourhood_group
*	neighbourhood
*	latitude
*	longitude
*	room_type 
*	price
*	minimum_nights
*	number_of_reviews
*	last_review 
*	reviews_per_month
*	calculated_host_listings_count
*	availability_365 

## Data Cleaning
Clean the data up so that it was usable for our model. I made the following changes and created the following variables:    
* Replace the missing data ('reviews_per_month' column) with mean, effected 10052 rows. 
* Removed 16 rows with missing data.  
* Made a column for the length of the 'name' column.  
* Eliminate the rental price is over 2000 dollars.
* Drop some columns which are irrelevant(name, host_id, latitude,longitude).
* After executing top of these actions, we still got 48729 records and 10 columns.

## Exploratory Data Analysis (EDA)
Below are a few tables and graphs I made. Try to find out the connections and relations in this dataset. 

**Correlations :**      
<img src="https://github.com/JohnnyHsieh1020/NYC_airbnb_proj/blob/main/images/Correlations.png" width=80%, heigh=80%>

**Neighbourhood group :**      
<img src="https://github.com/JohnnyHsieh1020/NYC_airbnb_proj/blob/main/images/neighbourhood_group.png">

**Neighborhood Top 20 :**      
<img src="https://github.com/JohnnyHsieh1020/NYC_airbnb_proj/blob/main/images/neighbourhood.png">

**Room Type :**      
<img src="https://github.com/JohnnyHsieh1020/NYC_airbnb_proj/blob/main/images/room_type.png">

**Price by Neighborhood group :**      
<img src="https://github.com/JohnnyHsieh1020/NYC_airbnb_proj/blob/main/images/price_by_ne_group.png">

**Word Cloud (Name) :**  
<img src="https://github.com/JohnnyHsieh1020/NYC_airbnb_proj/blob/main/images/wordcloud_name.png" width="600">

**Word Cloud (Neighbourhood) :**  
<img src="https://github.com/JohnnyHsieh1020/NYC_airbnb_proj/blob/main/images/wordcloud_neighbourhood.png" width="600">
 
## Model Building
1. By using LabelEncoder. It can transform data into a value between 0 and n_classes-1.
2. I split the data in a 80–20 ratio.
3. I used four different models and evaluated them using Mean Absolute Error(MAE).
    * Linear Regression: NMAE = -64.80  
    * Lasso Regression: NMAE = -64.79
    * Random Forest: NMAE = -60.36
    * Decision Tree Regressor: NMAE = -58.64
4. I also used GridsearchCV to find out the best group of parameters that can optimize the Random Forest and Decision Tree Regressor model.
    * GridsearchCV & Random Forest: NMAE =  -56.05
    * GridsearchCV & Decision Tree Regressor: NMAE =  -57.86

## Model performance
Below are the results. Using GridsearchCV to optimize Random Forest model has the best performance.  
* Linear Regression: MAE = 66.34
* Lasso Regression: MAE = 66.33
* Random Forest: MAE = 60.953719964861605
* GridsearchCV & Random Forest: MAE = 56.94
* Decision Tree Regressor: MAE = 59.96
* GridsearchCV & Decision Tree Regressor: MAE = 58.60
