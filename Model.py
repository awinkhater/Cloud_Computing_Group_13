#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.model_selection import train_test_split
from statsmodels.tsa.holtwinters import ExponentialSmoothing


# In[3]:


#Read a Dataset
DF=pd.read_csv('New_CDF.csv',parse_dates=['Date/Time'], index_col='Date/Time')

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


# In[9]:


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
    plt.savefig("model_plot_1.png")
    plt.show()


# In[4]:


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


# In[5]:


##====================================================Train Test Split
y=df['Hmax']
Features=df.copy()
Features.drop(['Hmax'], axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(
  Features,y , random_state=104,test_size=0.20, shuffle=False)

print(f"Training Set length: {len(X_train)}")
print(f"Training Set length: {len(X_test)}")


# In[6]:


#=========================== HOlT WINTERS
# #Simple forecasting methods (SES,drift)
model = ExponentialSmoothing(y_train, trend='add', seasonal='add', seasonal_periods=48)
model_fit = model.fit()

# Make predictions on the test set
predictions = model_fit.forecast(len(X_test))
Pm=np.mean(predictions)
for i in range(len(predictions)):
    predictions[i]= (predictions[i]+Pm)/2
# #===================================REGRESSION
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# transform data
original_column_names = X_train.columns.tolist()
# Transform data
X_train_F = scaler.fit_transform(X_train)

# # Create a new DataFrame for X_train_F with original column names
X_train_F = pd.DataFrame(X_train_F, columns=original_column_names, index=y_train.index)

#Plot the original data, training data, and test data
plt.figure(figsize=(10, 6))
plt.plot(y_train.index, y_train, label='Training Data')
plt.plot(y_test.index, y_test, label='Test Data', color='orange')
plt.plot(y_test.index, predictions, label='Holt-Winters Forecast', linestyle='--', color='green')
plt.xlabel('Date')
plt.ylabel('Wave Height')
plt.title('Holt-Winters Forecast')
plt.legend()
plt.savefig("model_plot_2.png")
plt.show()


# In[7]:


#=====================VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor
def it_VIF(df):
    thresh=5
    output=pd.DataFrame()
    k=df.shape[1]
    vif=[variance_inflation_factor(df.values,i) for i in range(df.shape[1])]
    for i in range(1,k):
        print(f"Iteration {i}:")
        print(vif)
        a=np.argmax(vif)
        print(f"MAX VIF IS FEATURE:{a}")
        if(vif[a] <=thresh):
            break
        if (i==1):
            output=df.drop(df.columns[a], axis=1)
            vif=[variance_inflation_factor(output.values, j) for j in range(output.shape[1])]
        elif i>1:
            output = output.drop(output.columns[a], axis=1)
            vif = [variance_inflation_factor(output.values, j) for j in range(output.shape[1])]
    return (output)
features=it_VIF(X_train_F)
print(f"Chosen Features:",features.columns)
print("No Colinearity Issues, VIF reccomends keeping all features")


# In[10]:


#================= Backwards Regression
import statsmodels.api as sm
X_train_F=sm.add_constant(X_train_F)
X_train_F1=X_train_F.copy(deep=True)
bswr1 = sm.OLS(y_train, X_train_F1).fit()
print(bswr1.summary())
X_train_F1.drop(['SST'], axis=1, inplace=True)
bswr2 = sm.OLS(y_train,X_train_F1 ).fit()
#print(bswr2.summary())
X_train_F1.drop(['Peak Direction'], axis=1, inplace=True)
bswr3 = sm.OLS(y_train,X_train_F1 ).fit()
print(bswr3.summary())


##================ACF of residuals
X_test_reduced=sm.add_constant(X_test)
X_test_reduced=X_test_reduced[['const','Hs','Tz', 'Tp']]
yt_pred= bswr3.predict(X_train_F1)
y_pred = bswr3.predict(X_test_reduced)
predictions=pd.concat([yt_pred,y_pred],axis=1)
PE=[]
y_pred=pd.DataFrame(y_pred)
Y_test=pd.DataFrame(y_test, index=y_test.index)
y_pred=y_pred[0].values.tolist()

Y_test=Y_test['Hmax'].values.tolist()
for i in range(len(y_pred)):
     P=y_pred[i]
     A=Y_test[i]
     PE.append(A-P)

plt.figure()
plt.scatter(y,predictions[0])
plt.title('Actual vs Predicted Value Plot')
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')
t= np.linspace(0, 6,10000 )
c=t
plt. plot(t,c,c= 'g')
plt.tight_layout()
plt.savefig("model_plot_3.png")
plt.show()
#ACF of Residuals
ACF_stem(PE,48, "Predicted vs Error ACF")
print("Variance of Residuals:", np.var(PE))
print("Mean of Residuals:", np.mean(PE))

#===FTEST:
A = np.identity(len(bswr3.params))
A = A[1:,:]
print("FTEST",bswr3.f_test(A))


plt.figure(figsize=(30, 10))
plt.plot(y_train.index, y_train, label='Train Data')
plt.plot(y_test.index,y_test, label='Actual Test Data')
plt.plot(y_test.index,y_pred, label='Predicted Test Data')

plt.title('Regression Predicitons')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Max Wave Height')
plt.grid(True)
plt.tight_layout()
plt.savefig("model_plot_4.png")
plt.show()


# In[ ]:




