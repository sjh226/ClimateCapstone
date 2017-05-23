import pandas as pd
import numpy as np
import math
import re
from sklearn.neighbors import KNeighborsClassifier

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
                                'PeakWindDirection'],\
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
    # date_df = dataframe.pop('DATE')
    # convert camelcase to snakecase for columns
    dataframe.columns = [camel_to_snake(name) for name in dataframe.columns]
    dataframe.columns = ['date', 'hourly_visibility', 'hourly_dry_bulb_temp_f',\
                         'hourly_wet_bulb_temp_f'] + list(dataframe.columns[4:])
    return dataframe, climate_df

def clean_type(df):
    # convert columns to consistent types (float or datetime)
    columns = ['hourly_visibility', 'hourly_dry_bulb_temp_f', 'hourly_wet_bulb_temp_f',\
               'hourly_wet_bulb_temp_f', 'hourly_dew_point_temp_f',\
               'hourly_relative_humidity', 'hourly_wind_speed',\
               'hourly_station_pressure', 'hourly_pressure_change',\
               'hourly_sea_level_pressure', 'hourly_precip',\
               'hourly_altimeter_setting', 'daily_average_dry_bulb_temp',\
               'daily_dept_from_normal_average_temp', 'daily_precip',\
               'daily_snowfall', 'daily_snow_depth',\
               'daily_peak_wind_speed', 'daily_heating_degree_days',\
               'daily_cooling_degree_days', 'date']
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def fill_nans(df):
    # drop rows with NaN in largest column aside from 'date'
    df = df.dropna(subset=['hourly_relative_humidity'])
    # drop rows with fewer than 10 non-na values (78% of data)
    # df = df.dropna(thresh=10)
    ## fill hourly visibility NaNs with mean
    # df['hourly_visibility'].loc[:] = df['hourly_visibility'].astype(str)\
    #                                  .str.rstrip('sV').replace('*', 'nan')
    # mean_vis = np.mean(df[df['hourly_visibility'] != 'nan']['hourly_visibility']\
    #            .astype(float))
    # df['hourly_visibility'] = df['hourly_visibility'].str.replace('nan',str(mean_vis))\
    #                           .astype(float)

    # try KNN to impute values
    # https://github.com/chrisalbon/notes_on_data_science_machine_learning_and_artificial_intelligence
    columns = ['hourly_dry_bulb_temp_f']
    complete_columns = ['hourly_dew_point_temp_f', 'hourly_relative_humidity', \
                        'date']
    for col in columns:
        KNN = KNeighborsClassifier(weights='distance')
        X = df[df[col].notnull()][complete_columns].values
        X_pred = df[df[col].isnull()][complete_columns].values
        y = df[df[col].notnull()].pop(col).values
        KNN.fit(X, y)
        y_pred = KNN.predict(X_pred)
        df[col] = df[col].fillna(y_pred)
        complete_columns.append(col)

    # fill precip nans to 0.00
    # df['hourly_precip'].fillna('0.00', inplace=True)

    # # use monthly averages to fill nans
    # mean_temps = np.array([])
    # for idx, temp in enumerate(df['monthly_mean_temp']):
    #     if not np.isnan(temp):
    #         if len(mean_temps) > 1:
    #             mean_temps = np.vstack((mean_temps, np.array([idx, temp])))
    #         else:
    #             mean_temps = np.hstack((mean_temps, np.array([idx, temp])))
    # for row in df.head().iterrows():
    #     idx = row[0]
    #     row[1]['hourly_dry_bulb_temp_f']

    return df

if __name__ == '__main__':
    df = pd.read_csv('965113.csv')
    df, climate_df = clean_data(df)
    df = clean_type(df)

    df1 = df[['date','hourly_visibility','hourly_dry_bulb_temp_f',\
              'hourly_wet_bulb_temp_f','hourly_dew_point_temp_f',\
              'hourly_relative_humidity','hourly_wind_speed',\
              'hourly_wind_gust_speed','hourly_station_pressure']].head(1000)

    df1 = fill_nans(df1)
