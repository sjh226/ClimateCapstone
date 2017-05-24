import matplotlib.pyplot as plt
from sklearn import preprocessing
from d_manage import clean_data, clean_type, fill_nans


def plot_departure(df):
    plt.close()
    y = df['daily_dept_from_normal_average_temp'].values
    x = df['date'].values
    min_max_scaler = preprocessing.MinMaxScaler()
    x = min_max_scaler.fit_transform(x)
    plt.plot(x, y)
    plt.title('Daily Departure from Avg Temp over Time')
    plt.savefig('temp_departure.png')


if __name__ == '__main__':
    df = pd.read_csv('965113.csv')
    df, climate_df = clean_data(df)
    df = clean_type(df)

    df2 = df[df['daily_dept_from_normal_average_temp'].notnull()]\
            [['date', 'daily_dept_from_normal_average_temp']]
    plot_departure(df2)
