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


 
