## Mile High Climate Change: Analyzing and Modeling Climate Change in Denver

![Climate Pic](images/Climate-change-1-1-938x450.jpg)

#### Problem

While climate change is a popular topic of discussion, public opinion is not often supported with accurate data. My goal for this project is to analyze the past 40 years of Denver climate data to see if there has been a statistically significant change in the climate and if so, to model and predict the future trend.

Climate models have continuously evolved throughout the recent years, implementing new features as their correlations are discovered. The current Earth System Models (ESMs) have improved upon the previous atmospheric and oceanographic models by including biochemical cycles as a predictor. While tremendous strides have been made in realizing these hard-to-detect trends, there are still more to be discovered. My hope is to utilize supervised machine learning models to predict climate trends, the benefit being that these models may be able to predict subtle trends without needing to know the source.

#### Data and Methods

My data is being taken from [National Centers for Environmental Information](https://www.ncdc.noaa.gov/). I have accessed and downloaded the land-based local climate data from DIA since Jan. 1, 1997 along with the data from Buckley AFB since Jan. 1, 1977. This data can be downloaded locally as a CSV and I've gone ahead and stored the raw data on an AWS S3 bucket.

No feature is complete for every row, so I began by cleaning the data to ensure I could utilize any available features in my model. There were many features with mixed data-types which needed to be translated into continuous data. One instance of this is the precipitation column which allows “T” to be entered for trace precipitation. In this specific instance, I replaced the "trace" values with 0. I went through each column where intuitive imputation was relevant.

For the rest of the features, I employed a k-Nearest Neighbors regression model to fill in missing values. This learned trends in the complete columns to go through each remaining column and predict the missing data.

<hr>

#### A/B Testing

I initially conducted a simple hypothesis test on the data to reject the null hypothesis that the climate has not changed significantly during this time period. I split the data into two 20-year periods (1977-1997 from Buckley and 1997-2017 from DIA). When analyzing hourly dry-bulb temperatures this split resulted in a p-value of 1.72e-56, allowing me to reject the null and conclude that the temperature has in fact changed significantly.

For ease of computation I decided to focus on the DIA data since 1997. Below I plotted the daily departure from the average temperature over time and ran it through a linear regression. There is an obvious increasing tend in the data and I will include this in the model along with the seasonal periodicity.

![temp_dpt](images/temp_departure_lr.png)

#### Gaussian Process Regression

To model this climate data, I chose to implement a [Gaussian Process Regressor](http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html#sklearn.gaussian_process.GaussianProcessRegressor). This allows me to return probabilities along with my predictions. For each predicted point, I have a confidence interval to illustrate the error in the prediction which is especially useful for timeseries predictions. Much of the variation in my model is based on modeling [CO2 Concentrations at Mauna Loa](http://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_co2.html).

##### Kernels
I chose to combine a few kernels to fully account for the signal in the data. A squared exponential kernel was chosen to model the exponential

My best model predicting on the years 2015 and 2016 scored an R^2 value of 0.451.
