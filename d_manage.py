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
    # how do I keep values as NaN when converting?
    columns = ['hourly_visibility', 'hourly_dry_bulb_temp_f', 'hourly_wet_bulb_temp_f',\
               'hourly_wet_bulb_temp_f', 'hourly_dew_point_temp_f',\
               'hourly_relative_humidity', 'hourly_wind_speed',\
               'hourly_station_pressure', 'hourly_pressure_change',\
               'hourly_sea_level_pressure', 'hourly_precip',\
               'hourly_altimeter_setting', 'daily_average_dry_bulb_temp',\
               'daily_dept_from_normal_average_temp', 'daily_precip',\
               'daily_snowfall', 'daily_snow_depth',\
               'daily_peak_wind_speed', 'daily_heating_degree_days',\
               'daily_cooling_degree_days']
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    # strings = df['hourly_dry_bulb_temp_f'].astype(str)
    # for string in strings:
    #     new.append(re.sub('\D\W', '', s))
    return df


def fill_nans(df):
    # drop rows with fewer than 10 non-na values (78% of data)
    df = df.dropna(thresh=10)
    # drop rows with NaN in largest column
    # df = df.dropna(subset=['hourly_dry_bulb_temp_f'])
    ## fill hourly visibility NaNs with mean
    # df['hourly_visibility'].loc[:] = df['hourly_visibility'].astype(str)\
    #                                  .str.rstrip('sV').replace('*', 'nan')
    # mean_vis = np.mean(df[df['hourly_visibility'] != 'nan']['hourly_visibility']\
    #            .astype(float))
    # df['hourly_visibility'] = df['hourly_visibility'].str.replace('nan',str(mean_vis))\
    #                           .astype(float)

    df['hourly_precip'].fillna('0.00', inplace=True)
    df['hourly_precip'] = df['hourly_precip'].str.rstrip('s').replace('T', '0.00')
    df['hourly_precip'] = df['hourly_precip'].astype(float)

    # use monthly averages to fill nans
    mean_temps = np.array([])
    for idx, temp in enumerate(df['monthly_mean_temp']):
        if not np.isnan(temp):
            if len(mean_temps) > 1:
                mean_temps = np.vstack((mean_temps, np.array([idx, temp])))
            else:
                mean_temps = np.hstack((mean_temps, np.array([idx, temp])))
    for row in df.head().iterrows():
        idx = row[0]
        row[1]['hourly_dry_bulb_temp_f']

    # try KNN to impute values
    # https://github.com/chrisalbon/notes_on_data_science_machine_learning_and_artificial_intelligence


    return df

if __name__ == '__main__':
    df = pd.read_csv('965113.csv')
    df, climate_df = clean_data(df)
    # df = fill_nans(df)
    df = clean_type(df)
