<h align="center">
## Mile High Climate Change:
## Analyzing and Modeling Climate Change in Denver
</h>
While climate change is a popular topic of discussion, public opinion is not often supported with accurate data. My goal for this project is to analyze the past 40 years of Denver climate data to see if there has been a statistically significant change in the climate and if so, to model and predict the future trend.

My data is being taken from NOAA’s National Centers for Environmental Information at https://www.ncdc.noaa.gov/. I have accessed and downloaded the land-based local climate data from DIA since Jan. 1, 1997 along with the data from Buckley AFB since Jan. 1, 1977. This data can be downloaded locally as a csv.

With ~260,000 rows I will need to decide which values are reasonable to impute and which should be discarded. No feature is complete for every row, so I will begin my dropping rows that are without the most prominent feature,    . There are also issues with features containing data of mixed-types. One instance of this is the precipitation column which allows “T” to be entered for trace precipitation. These values will have to be converted into meaningful numeric values to run them through my model.

My plan is to use KNN or another machine learning regressor to impute most of the missing data by looping through each column as the target. I am basing my custom inputer on https://github.com/hammerlab/fancyimpute.

If this change is in fact significant, I will then train a model to predict future change (I expect to use linear regression or a neural net).
If there is not evident climate change, I should still be able to express how I would create a climate model (and possibly go further back in time)

#### A/B Testing

I initially conducted a simple hypothesis test on the data to reject the null hypothesis that the climate has not changed significantly in this time period. I split the data into two 20-year periods (1977-1997 from Buckley and 1997-2017 from DIA). When analyzing hourly dry-bulb temperatures this split resulted in a p-value of 1.72e-56, allowing me to conclude that the temperature has in fact changed significantly.

For ease of computation I decided to focus on the DIA data since 1997. Below I plotted the daily departure from the average temperature over time and ran it through a linear regression. There is an obvious increasing tend in the data and I will include this in the model along with the seasonal periodicity.

![temp_dpt](images/temp_departure_lr.png)
