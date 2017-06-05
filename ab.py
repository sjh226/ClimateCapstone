import pandas as pd
import numpy as np
import scipy.stats as stats


def hyp_test(df):
    a_samp = df[df['date'] < 852102000000000000]['hourly_dry_bulb_temp_f']
    b_samp = df[df['date'] >= 852102000000000000]['hourly_dry_bulb_temp_f']

    t_cal = (b_samp.mean() - a_samp.mean()) / \
        ((((a_samp.std() ** 2)/len(a_samp)) + \
        ((b_samp.std() ** 2)/len(b_samp))) ** .5)
    t, p = stats.ttest_ind(a_samp, b_samp, equal_var=False)
    print('Resulting t-value: {}\nand p-value: {}\nand calculated t: {}'\
            .format(t, p, t_cal))


if __name__ == '__main__':
    df = pd.read_pickle('40yr_df.pkl').sort_values('date')
    hyp_test(df)
    # this results in a t-value of 15.9 and p-value of 1.72e-56.
    # clearly there is a significant difference between hourly temperatures
    # between 1977-1997 and 1997-2017
