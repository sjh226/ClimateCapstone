import pandas as pd
import numpy as np
import math
import re
pd.options.mode.chained_assignment = None


def camel_to_snake(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def clean_data(dataframe):
    # clean dataframe by dropping meaningless columns or those with <500 values
    dataframe = dataframe.drop(['MonthlyAverageRH', 'MonthlyDewpointTemp', \
                                'MonthlyWetBulbTemp', 'MonthlyAvgHeatingDegreeDays',\
                                'MonthlyAvgCoolingDegreeDays', 'MonthlyAverageWindSpeed',\
                                'MonthlyGreatestPrecip', 'MonthlyGreatestPrecipDate',\
                                'STATION', 'STATION_NAME', 'ELEVATION', 'LATITUDE',\
                                'LONGITUDE', 'MonthlyMaximumTemp', 'MonthlyMinimumTemp',\
                                'MonthlyMeanTemp', 'MonthlyStationPressure',\
                                'MonthlySeaLevelPressure', 'MonthlyTotalSnowfall',\
                                'MonthlyTotalLiquidPrecip', 'MonthlyGreatestSnowfall',\
                                'MonthlyGreatestSnowfallDate', 'MonthlyDaysWithGT90Temp',\
                                'MonthlyDaysWithLT32Temp', 'MonthlyDaysWithGT32Temp',\
                                'MonthlyDaysWithLT0Temp', 'MonthlyDaysWithGT001Precip',\
                                'MonthlyDaysWithGT010Precip', 'MonthlyDaysWithGT1Snow',\
                                'MonthlyMaxSeaLevelPressureValue',\
                                'MonthlyMaxSeaLevelPressureDate',\
                                'MonthlyMaxSeaLevelPressureTime',\
                                'MonthlyMinSeaLevelPressureValue',\
                                'MonthlyMinSeaLevelPressureDate',\
                                'MonthlyMinSeaLevelPressureTime',\
                                'MonthlyTotalHeatingDegreeDays',\
                                'MonthlyTotalCoolingDegreeDays',\
                                'MonthlyDeptFromNormalHeatingDD',\
                                'MonthlyDeptFromNormalCoolingDD',\
                                'MonthlyTotalSeasonToDateHeatingDD',\
                                'MonthlyTotalSeasonToDateCoolingDD',\
                                'MonthlyGreatestSnowDepth',\
                                'MonthlyGreatestSnowDepthDate',\
                                'REPORTTPYE', 'DAILYWeather',\
                                'HOURLYPRSENTWEATHERTYPE', 'DAILYSunrise',\
                                'DAILYSunset', 'HOURLYSKYCONDITIONS',\
                                'HOURLYDRYBULBTEMPC', 'HOURLYWETBULBTEMPC',\
                                'HOURLYDewPointTempC', 'HOURLYWindDirection'],\
                                axis=1)
    # possible climate change indicators
    climate_df = dataframe[['MonthlyDeptFromNormalMaximumTemp',\
                            'MonthlyDeptFromNormalMinimumTemp',\
                            'MonthlyDeptFromNormalAverageTemp',\
                            'MonthlyDeptFromNormalPrecip']]
    dataframe = dataframe.drop(['MonthlyDeptFromNormalMaximumTemp',\
                                'MonthlyDeptFromNormalMinimumTemp',\
                                'MonthlyDeptFromNormalAverageTemp',\
                                'MonthlyDeptFromNormalPrecip'], axis=1)
    # dates
    dataframe['DATE'] = pd.to_datetime(dataframe['DATE'])
    date_df = dataframe.pop('DATE')
    # convert camelcase to snakecase for columns
    dataframe.columns = [camel_to_snake(name) for name in dataframe.columns]
    dataframe.columns = ['hourly_visibility', 'hourly_dry_bulb_temp_f',\
                         'hourly_wet_bulb_temp_f'] + list(dataframe.columns[3:])
    return dataframe, climate_df, date_df

def fill_nans(df):
    # drop rows with fewer than 10 non-na values (78% of data)
    df = df.dropna(thresh=10)
    # drop rows with NaN in largest column
    # df = df.dropna(subset=['hourly_dry_bulb_temp_f'])
    # fill hourly visibility NaNs with mean
    df['hourly_visibility'].loc[:] = df['hourly_visibility'].astype(str)\
                                     .str.rstrip('sV').replace('*', 'nan')
    mean_vis = np.mean(df[df['hourly_visibility'] != 'nan']['hourly_visibility']\
               .astype(float))
    df['hourly_visibility'] = df['hourly_visibility'].str.replace('nan',str(mean_vis))\
                              .astype(float)

    df['hourly_precip'].fillna('0.00', inplace=True)
    df['hourly_precip'] = df['hourly_precip'].str.rstrip('s').replace('T', '0.00')
    df['hourly_precip'] = df['hourly_precip'].astype(float)
    
    return df

if __name__ == '__main__':
    df = pd.read_csv('965113.csv')
    df, climate_df, date_df = clean_data(df)
    df = fill_nans(df)
