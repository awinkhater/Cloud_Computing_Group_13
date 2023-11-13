import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.model_selection import train_test_split
from statsmodels.tsa.holtwinters import ExponentialSmoothing
#Read a Dataset
DF=pd.read_csv('CDS.csv',parse_dates=['Date/Time'], index_col='Date/Time')

#Handling Missing Days
def find_missing_days(df):
    df.index = pd.to_datetime(df.index)
    start_date = df.index.min()
    end_date = df.index.max()
    expected_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    actual_dates = pd.Series(df.index.date).unique()
    missing_days = expected_dates.difference(actual_dates)

    return missing_days

missing_days = find_missing_days(DF)
# k=0
# for missing_day in missing_days:
#     k+=1
#     print(k,missing_day)

#Cutting the data at the date where the sampling becomes uneven
end_date = pd.to_datetime('2019-01-06')
DF = DF.loc[DF.index <= end_date]
print("CDF",len(DF))

############% Basic Cleaning
#No Null values
#No encoding
#No upsamplng/downsampling

#Drift Method:
null_count_before = DF['Peak Direction'].isnull().sum()
print(f'Number of null values in "Peak Direction" before drift correction: {null_count_before}')

def drp(df):
    df.index = pd.to_datetime(df.index)
    for col in df.columns:
        # phv finda placeholders (-99.90) and null values
        phv = (df[col] == -99.9) | df[col].isnull()

        # Iterates rows with placeholder/null values in the current column
        for i, row in df[phv].iterrows():
            start_time = i - pd.to_timedelta('1D')
            end_time = i + pd.to_timedelta('1D')
            fv = df[(df.index >= start_time) & (df.index <= end_time) & ~phv]

            if not fv.empty:
                mv = fv[col].mean()
                #Using df.at syntax to access a single point in the df
                df.at[i, col] = mv
    return df


df=drp(DF)
df.index = pd.to_datetime(df.index)

start_datetime = '2017-02-27 10:00'
end_datetime = '2017-02-28 01:00'

selected_columns = ['Peak Direction', 'SST']

# Calculate the average of the start and end datetime values for the selected columns
average_values = df.loc[[start_datetime, end_datetime], selected_columns].mean()

# Update the values
df.loc[(df.index > start_datetime) & (df.index < end_datetime), selected_columns] = average_values.values

#Created the final clean dataframe: df