# NYC_airbnb_proj  

## Code and Resources Used 
**Dataset from :** https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data  
**Python Version :** Python 3.7.4  
**IDE :** Spyder  

**Packages :** pandas, numpy, matplotlib, seaborn

**Reference websites :** 
1. https://www.kaggle.com/duygut/airbnb-nyc-price-prediction  
2. https://seaborn.pydata.org/generated/seaborn.scatterplot.html
3. https://blog.csdn.net/zyb228/article/details/100897145

## Data info 
We got about 48000 records and 14 columns:
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
*	Replace the missing data ('reviews_per_month' column) with mean, effected 10052 rows. 
*	Removed 16 rows with missing data.  
*	Made a column for the length of the 'name' column.  

## Exploratory Data Analysis (EDA)
Below are a few tables and graphs I made. Try to find out the connections and relations in this dataset. 

**Correlations :**      
<img src="https://github.com/JohnnyHsieh1020/NYC_airbnb_proj/blob/main/Correlations.png" width=80%, heigh=80%>

**Neighbourhood group :**      
<img src="https://github.com/JohnnyHsieh1020/NYC_airbnb_proj/blob/main/neighbourhood_group.png">

**Neighborhood Top 20 :**      
<img src="https://github.com/JohnnyHsieh1020/NYC_airbnb_proj/blob/main/neighbourhood.png">

**Room Type :**      
<img src="https://github.com/JohnnyHsieh1020/NYC_airbnb_proj/blob/main/room_type.png">

**Price by Neighborhood group :**      
<img src="https://github.com/JohnnyHsieh1020/NYC_airbnb_proj/blob/main/price_by_ne_group.png">

**Word Cloud (Name) :**  
<img src="https://github.com/JohnnyHsieh1020/NYC_airbnb_proj/blob/main/wordcloud_name.png" width="600">

**Word Cloud (Neighbourhood) :**  
<img src="https://github.com/JohnnyHsieh1020/NYC_airbnb_proj/blob/main/wordcloud_neighbourhood.png" width="600">
 
