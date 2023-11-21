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

#ADDING SOME PLOTS:
def ACF_prep (data,lags):
    mu=0
    for m in data:
        mu+=m
    mu=(mu/len(data))
    denom=0
    numerator=0
    List=[]
    Nlist=[]
    for d in data:
        denom+= (d-mu)**2
    for l in range (0, lags+1):
        numerator=0
        for i in range (len(data)):
            if i+l < len(data):
                numerator+=(data[i+l]-mu)*(data[i]-mu)

        ac= numerator/denom
        List.append(ac)
        Nlist.append(ac)
    Nlist.reverse()
    Nlist= Nlist+List[1:]
    LAGLIST = list(range(-lags, lags + 1))
    LD=pd.DataFrame(Nlist, index=LAGLIST)
    LD.columns=['AC Value']
    return(LD)
def ACF_stem(data, lags, title):
    LD=ACF_prep(data,lags)
    plt.figure()
    (markers, stemlines, baseline) = plt.stem(LD.index,LD['AC Value'], markerfmt='o')
    plt.setp(markers, color='red')
    m = 1.96 / (np.sqrt(len(data)))
    plt.title(title)
    plt.xlabel("# of Lags")
    plt.ylabel('AutoCorrelation value')
    plt.axhspan(-m, m, alpha=0.2, color='blue')
    plt.show()
ACF_stem(df['Hmax'],96,"Waves ACF Plot")

# #ACF looks like slight Daily Seasonality
# ################################%Correlation Matrix
# sns.set_style('darkgrid')
# sns.heatmap(data=df.corr(), annot=True)
# plt.title('Waves Correlation Matrix')
# plt.show()
# # ##############################Rolling Variance and Mean
def RMVplot(df, col):
    log=[]
    rv=[]
    rm=[]
    for i in df[col]:
        log.append(i)
        rv.append(np.var(log))
        rm.append(np.mean(log))
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(rm, label=f'{col} Rolling Mean')
    plt.xlabel('Time ->')
    plt.ylabel('Value')
    plt.legend()
    plt.xticks([])
    plt.title(f'Rolling Mean of {col}')
    plt.grid(True)
    plt.subplot(2,1,2)
    plt.plot(rv, label=f'{col} Rolling Variance')
    plt.xlabel('Time ->')
    plt.ylabel('Value')
    plt.legend()
    plt.xticks([])
    plt.title(f'Rolling Variance of {col}')
    plt.tight_layout()
    plt.grid(True)
    plt.show()
# RMVplot(df,'Hmax')
#Stationarity Test
from statsmodels.tsa.stattools import adfuller
def ADF_Cal(x):
    result = adfuller(x)
    print("ADF Statistic: %f" %result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

print(ADF_Cal(df['Hmax']))