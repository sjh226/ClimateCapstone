import pandas as pd
import numpy as np
import math
import re
from sklearn.neighbors import KNeighborsRegressor

pd.options.mode.chained_assignment = None


def camel_to_snake(name):
    '''
    INPUT: string in camel case
    OUTPUT: string in snake case
    '''

    st = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', st).lower()

def clean_data(dataframe):
    # clean dataframe by dropping meaningless columns or those with <500 values
    dataframe = dataframe.drop(['MonthlyAverageRH', 'MonthlyDewpointTemp', \
                                'MonthlyWetBulbTemp', 'MonthlyAvgHeatingDegreeDays',\
                                'MonthlyAvgCoolingDegreeDays', 'MonthlyAverageWindSpeed',\
                                'MonthlyGreatestPrecip', 'MonthlyGreatestPrecipDate',\
                                'STATION', 'STATION_NAME', 'ELEVATION', 'LATITUDE',\
                                'LONGITUDE', 'MonthlyMaximumTemp', 'MonthlyMinimumTemp',\
                                'MonthlyStationPressure',\
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
                                'HOURLYDewPointTempC', 'HOURLYWindDirection',\
                                'DAILYMaximumDryBulbTemp', 'DAILYMinimumDryBulbTemp',\
                                'PeakWindDirection', 'DAILYSnowDepth',\
                                'HOURLYSeaLevelPressure', 'DAILYAverageDryBulbTemp',\
                                'HOURLYAltimeterSetting', 'DAILYAverageRelativeHumidity',\
                                'DAILYAverageDewPointTemp', 'DAILYAverageWetBulbTemp',\
                                'DAILYAverageStationPressure', 'DAILYAverageSeaLevelPressure',\
                                'DAILYAverageWindSpeed', 'DAILYPeakWindSpeed',\
                                'DAILYSustainedWindDirection', 'MonthlyMeanTemp',\
                                'DAILYSustainedWindSpeed', 'DAILYPrecip'],\
                                axis=1)
    # possible climate change indicators
    climate_df = dataframe[['DATE',\
                            'MonthlyDeptFromNormalMaximumTemp',\
                            'MonthlyDeptFromNormalMinimumTemp',\
                            'MonthlyDeptFromNormalAverageTemp',\
                            'MonthlyDeptFromNormalPrecip',\
                            'DAILYDeptFromNormalAverageTemp']]
    climate_df.dropna(thresh=6, inplace=True)
    dataframe = dataframe.drop(['MonthlyDeptFromNormalMaximumTemp',\
                                'MonthlyDeptFromNormalMinimumTemp',\
                                'MonthlyDeptFromNormalAverageTemp',\
                                'MonthlyDeptFromNormalPrecip',\
                                'DAILYDeptFromNormalAverageTemp'], axis=1)
    # dates
    dataframe['DATE'] = pd.to_datetime(dataframe['DATE'])
    climate_df['DATE'] = pd.to_datetime(climate_df['DATE'])
    # convert camelcase to snakecase for columns
    dataframe.columns = [camel_to_snake(name) for name in dataframe.columns]
    dataframe.columns = ['date', 'hourly_visibility', 'hourly_dry_bulb_temp_f',\
                         'hourly_wet_bulb_temp_f'] + list(dataframe.columns[4:])
    climate_df.columns = [camel_to_snake(name) for name in climate_df.columns]
    return dataframe, climate_df

def clean_type(df):
    # convert columns to consistent types (float or datetime)
    columns = ['hourly_visibility', 'hourly_dry_bulb_temp_f', 'hourly_wet_bulb_temp_f',\
               'hourly_wet_bulb_temp_f', 'hourly_dew_point_temp_f',\
               'hourly_relative_humidity', 'hourly_wind_speed',\
               'hourly_station_pressure', 'hourly_pressure_change',\
               'hourly_precip',\
               'daily_snowfall', 'daily_heating_degree_days',\
               'daily_cooling_degree_days', 'date']
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def fill_nans(df):
    # drop rows with fewer than 8 non-na values (81% of data)
    df = df.dropna(thresh=8)

    # fill precip nans to 0.00 (including 'trace' values)
    df['hourly_precip'].fillna(0.00, inplace=True)
    df['daily_snowfall'].fillna(0.00, inplace=True)

    # fill wind gust with the wind speed
    df['hourly_wind_gust_speed'].fillna(df['hourly_wind_speed'], inplace=True)
    # use KNN to impute missing values
    # after each run of the KNN Regressor loop, the imputed column is added
    # into the valid training columns
    columns = ['hourly_wind_gust_speed', 'hourly_relative_humidity',\
               'hourly_dry_bulb_temp_f', 'hourly_wind_speed',\
               'hourly_visibility', 'hourly_wet_bulb_temp_f',\
               'hourly_station_pressure', 'hourly_pressure_tendancy',\
               'hourly_pressure_change', 'daily_heating_degree_days',\
               'daily_cooling_degree_days']
    complete_columns = ['date', 'hourly_precip', 'daily_snowfall']
    for col in columns:
        print('Imputing column', col)
        KNN = KNeighborsRegressor(weights='distance')
        X = df[df[col].notnull()][complete_columns].values
        X_pred = df[df[col].isnull()][complete_columns]
        y = df[df[col].notnull()].pop(col).values
        KNN.fit(X, y)
        y_pred = KNN.predict(X_pred.values)
        df[col].fillna(dict(zip(X_pred.index, y_pred)), inplace=True)
        complete_columns.append(col)

    return df

def to_bucket(f, bucket, write_name):
    '''
    Write files to s3 bucket.

    INPUT: f - file to write
           bucket - bucket to write to
           write_name - name for S3
    '''
    # Specify the service
    s3 = boto3.resource('s3')
    data = open(f, 'rb')
    # s3.create_bucket(Bucket=bucket)
    s3.Bucket(bucket).put_object(Key=write_name, Body=data)
    print('Success! {} added to {} bucket'.format(write_name, bucket))

def make_bucket(bucket_name):
    s3 = boto3.resource('s3')
    s3.create_bucket(Bucket=bucket_name)


if __name__ == '__main__':
    # df = pd.read_csv('965113.csv')
    make_bucket('climate_data')
    to_bucket('965113.csv', 'climate_data', '20yr.csv')
    df = pd.read_csv('https://s3.amazonaws.com/climate_data/20yr.csv')

    df, climate_df = clean_data(df)
    df = clean_type(df)

    df1 = df.head(1000)
    df = fill_nans(df)
