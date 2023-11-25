#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns


# In[3]:


#Read a Dataset
DF = pd.read_csv('New_CDF.csv',parse_dates=['Date/Time'], index_col='Date/Time')

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
df.to_csv('cleaned_1_df.csv', index=False)


# In[7]:


#Plot 1
sns.set() # Seaborn-Stile verwenden
df[df.Hs > 0].pivot_table("Hs", index = df.index).plot(figsize=(25,10))
plt.ylabel("Significant Wave Height")
plt.xlabel("Time")
# displaying the title
plt.title("Changes in Significant Wave Height over Time")
plt.savefig("plot1.png")

#Plot 2
sns.set() # Seaborn-Stile verwenden
df[df.SST > 0].pivot_table("SST", index = df.index).plot(figsize=(25,10))
plt.ylabel("Sea Surface Temperature")
plt.xlabel("Time")
# displaying the title
plt.title("Changes in Sea Surface Temperature over Time")
plt.savefig("plot2.png")

#Plot 3
sns.set() # Seaborn-Stile verwenden
df[df.Hmax > 0].pivot_table("Hmax", index = df.index).plot(figsize=(25,10))
plt.ylabel("Maximum Wave Height")
plt.xlabel("Time")
# displaying the title
plt.title("Changes in Maximum Wave Height over Time")
plt.savefig("plot3.png")

#plot 4 - 6

#Plot 4
df_1 = df.copy()
df_1['Date_Time'] = df_1.index
df_1['month'] = pd.to_datetime(df_1['Date_Time']).dt.month
df_1['year'] = pd.to_datetime(df_1['Date_Time']).dt.year
df_1_avg = df_1.groupby(['year','month'], as_index=False).mean()
df_1_2017 = df_1_avg[df_1_avg['year'] == 2017]
df_1_2017 = df_1_2017.reset_index()
df_1_2018 = df_1_avg[df_1_avg['year'] == 2018]
df_1_2018 = df_1_2018.reset_index()
index = ['Jan', 'Feb', 'Mar','Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
plot4_avg = pd.DataFrame({'2017': df_1_2017['Hmax'].tolist(), '2018': df_1_2018['Hmax'].tolist()}, index=index)
ax = plot4_avg.plot.bar(rot=0, color={"2017": "green", "2018": "red"})
plt.ylabel("Maximum Wave Height")
plt.xlabel("Months")
# displaying the title
plt.title("Comparing changes in Maximum Wave Height in 2017 / 2018")
plt.savefig("plot4.png")

#Plot 5
plot5_avg = pd.DataFrame({'2017': df_1_2017['SST'].tolist(), '2018': df_1_2018['SST'].tolist()}, index=index)
ax = plot5_avg.plot.bar(rot=0, color={"2017": "green", "2018": "red"})
plt.ylabel("Sea Surface Temperature")
plt.xlabel("Months")
# displaying the title
plt.title("Comparing changes in Sea Surface Temperature in 2017 / 2018")
plt.savefig("plot5.png")

#Plot 6
plot6_avg = pd.DataFrame({'2017': df_1_2017['Hs'].tolist(), '2018': df_1_2018['Hs'].tolist()}, index=index)
ax = plot6_avg.plot.bar(rot=0, color={"2017": "green", "2018": "red"})
plt.ylabel("Significant Wave Height")
plt.xlabel("Months")
# displaying the title
plt.title("Comparing changes in Significant Wave Height in 2017 / 2018")
plt.savefig("plot6.png")

#Plot 7
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
    plt.savefig("plot7.png")
    plt.show()
    

ACF_stem(df['Hmax'],96,"Waves ACF Plot")

#Plot 8
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
    plt.savefig("plot8.png")
    plt.show()

RMVplot(df,'Hmax')


#Plot 9 - 12

def pearson_corr(x, y):
    x_diff = x - np.mean(x)
    y_diff = y - np.mean(y)
    return np.dot(x_diff, y_diff) / (np.sqrt(sum(x_diff ** 2)) * np.sqrt(sum(y_diff ** 2)))

sns.set()


# In[21]:


#Plot 9
sns.scatterplot(data = df, x = "SST", y = "Hs", s=10)
plt.xlabel("Temperature (°C)")
plt.ylabel("Wave Height (m)")
plt.title(f"Significant Wave Height vs. Sea Surface Temperature. Correlation: {pearson_corr(df.Hs, df.SST):.2f}")
#plt.savefig("WH_SST.png", format="png", dpi=300)
plt.savefig("plot9.png")
plt.show()


#Plot 10
ax = plt.axes(projection="polar")
ax.set_theta_direction(-1)
ax.set_theta_zero_location("N")
ax.set_thetagrids([0, 90, 180, 270], labels=["N", "E", "S", "W"])
ax.set_rgrids([1, 2, 3, 4, 5, 6, 7], angle=0)
ax.scatter(df["Peak Direction"]*(np.pi / 180), df.Hmax,  s=5)
plt.title(f"Maximum Wave Height vs. Direction")
#plt.savefig("WH_D.png", format="png", dpi=300)
plt.savefig("plot10.png")
plt.show()


#Plot 11
sns.scatterplot(data = df, x="Tp", y="Hmax", s=10)
plt.xlabel("Period (s)")
plt.xlim(df.Tp.min()-1, df.Tp.max()+1)
plt.ylabel("Wave Height (m)")
plt.title(f"Maximum Wave Height vs. Peak Energy Wave Period. Correlation: {pearson_corr(df.Hmax, df.Tp):.2f}")
#plt.savefig("WH_Tp.png", format="png", dpi=300)
plt.savefig("plot11.png")
plt.show()


#Plot 12
sns.scatterplot(data = df, x=df.Tz, y=df.Hmax, s=10)
plt.xlabel("Period (s)")
plt.xlim(df.Tp.min()-1, df.Tp.max()+1)
plt.ylabel("Wave Height (m)")
plt.title(f"Maximum Wave Height vs. Zero Upcrossing Wave Period. Correlation: {pearson_corr(df.Hmax, df.Tz):.2f}")
#plt.savefig("WH_Tz.png", format="png", dpi=300)
plt.savefig("plot12.png")
plt.show()

