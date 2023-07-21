#!/usr/bin/env python
# coding: utf-8

# # Milk Production - Time Series Forecasting

# ***
# _**Importing the required libraries & packages**_

# In[1]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pandas.plotting import autocorrelation_plot
from math import sqrt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from pmdarima import auto_arima
import pickle
import warnings
warnings.filterwarnings('ignore')


# _**Changing The Default Working Directory Path & Reading the Dataset using Pandas Command along with that changing the index column with `Month` column and also changing the data type of `Month` column as <span style="color:red">DateTime</span>**_

# In[2]:


os.chdir('C:\\Users\\Shridhar\\Desktop\\Milk Production Project')
df = pd.read_csv('monthly-milk-production.csv',parse_dates = ['Month'],index_col = 'Month')


# ## Exploratory Data Analysis(EDA)

# _**Checking the data type of the column in the dataset**_

# In[3]:


df.dtypes


# _**Getting the shape of the dataset**_

# In[4]:


df.shape


# _**Checking for the null values in the column from the dataset**_

# In[5]:


df.isna().sum()


# _**Getting the summary of various descriptive statistics for the numeric column in the dataset**_

# In[6]:


df.describe()


# ## Data Visualization

# _**Plotting the line graph to show the data trend in the dataset and saving the graph as PNG file**_

# In[7]:


df.plot(figsize =(10, 5))
plt.title('Monthly Milk Production')
plt.savefig('Monthly Milk Production.png')
plt.show()


# _**Plotting the histogram and KDE line graph to show the distribution of data in the dataset and saving the graph as PNG file**_

# In[8]:


fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1, sharex = False, sharey = False, figsize = (10, 5))
df.hist(ax = ax1)
df.plot(kind = 'kde', ax = ax2 )
plt.title('Data Distribution of Milk Production')
plt.savefig('Data Distribution of Milk Production.png')
plt.show()


# _**Plotting the graph with "Seasonal Decompose" function to show the Data Description, Trend, Seasonal, Residuals and saving the graph as PNG file**_

# In[9]:


plt.rcParams['figure.figsize'] = 10, 5
decomposition = seasonal_decompose(df['Milk Production'], period = 12, model = 'additive')
decomposition.plot()
plt.savefig('Trend, Seasonal, Residual Graph.png')
plt.show()


# _**Plotting the graphs with Auto-Correlation and Partial Auto-Correlation of the data from the dataset and saving the graphs as PNG file**_

# In[10]:


fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1, sharex = False, sharey = False, figsize = (10, 5))
ax1 = plot_acf(df['Milk Production'], lags = 10, ax = ax1)
ax2 = plot_pacf(df['Milk Production'], lags = 10, ax = ax2)
plt.subplots_adjust(hspace = 0.5)
plt.savefig('ACF & PACF.png')
plt.show()


# ## Data Tranmsformation

# _**The `adf_check()` function performs an Augmented Dickey-Fuller test on a time series. The test is used to determine whether a time series is stationary or not. If the p-value of the test is less than or equal to 0.05, then there is strong evidence against the null hypothesis, and the series is considered to be stationary. Otherwise, the series is considered to be non-stationary.
# The function takes a time series as input and returns the results of the test. The results are printed to the console, along with a message indicating whether the series is stationary or not.**_

# In[11]:


def adf_check (time_series):
    result = adfuller (time_series)
    print ('Augmented Dickey Fuller Test :')
    labels = ['ADF Test Statistics', 'P Value', 'Number of Lags Used','Number of Observations']
    for value, label in zip(result, labels):
        print (label +' : '+ str (value))
    if result [1] <= 0.05:
        print ('Strong evidence against the null hypothesis, hence REJECT null hypothesis and the series is Stationary ')
    else:
        print ('Weak evidence against the null hypothesis, hence ACCEPT null hypothesis and the series is Not Stationary ')


# _**Performing the Augmented Dickey-Fuller test on the original data in the dataset to find whether the time series is stationary or not**_

# In[12]:


adf_check(df['Milk Production'])


# _**Since the Time Series is Not Stationary, the dataset is transformed as a new DataFrame with First Differene and Seasonal First Difference to make it as a Stationary Series**_

# In[13]:


df1 = df.diff().diff(12).dropna()


# _**Performing again the Augmented Dickey-Fuller test on the new transformed data from the dataset to find whether the time series is stationary or not**_

# In[14]:


adf_check(df1['Milk Production'])


# _**Plotting the line graph to show the data trend in the transformed data from the dataset and saving the graph as PNG file**_

# In[15]:


df1.plot(figsize =(10, 5))
plt.title('Monthly Milk Production(Transformed)')
plt.savefig('Monthly Milk Production(Transformed).png')
plt.show()


# _**Plotting the graph with pandas plotting autocorrelation_plot to show the difference between the Stationary Data and Non-Stationary Data and saving it as PNG file**_

# In[16]:


fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1, sharex = False, sharey = False, figsize = (10,5))
ax1 = autocorrelation_plot(df['Milk Production'], ax = ax1)
ax1.set_title('Non-Stationary Data')
ax2 = autocorrelation_plot(df1['Milk Production'], ax = ax2)
ax2.set_title('Stationary_Data')
plt.subplots_adjust(hspace = 0.5)
plt.savefig('Autocorrelation_plot of Stationary & Non-Stationary.png')
plt.show()


# ## Model Fitting

# _**Getting the p value and q value for the model fitting using `auto_arima` function by passing through some needed parameters, the best model is evaluated by least Akaike Information Criterion[AIC]**_ 

# In[17]:


model = auto_arima(df['Milk Production'], d = 1, D = 1, seasonal = True, m = 12, max_order = 6,
                     start_p = 0, start_q = 0, test = 'adf', trace = True)


# _**Defining the summary of the model fitted with `auto_arima` function, here getting various information such as Akaike Information Criterion[AIC], Bayesian Information Criterion[BIC}, Hannan-Quinn Information Criterion[HQIC], Log Likelihood etc. from which we can evaluate the model**_

# In[18]:


model.summary()


# _**Splitting the dataset in training data(85%) and test data(15%)**_

# In[19]:


train = df[:int(0.85*len(df))]
test = df[int(0.85*len(df)):]


# _**Getting the shapes of training data and test data, so that we can able to know the exact observations in training and test data**_

# In[20]:


train.shape, test.shape


# _**Fitting the model in SARIMAX model with the best value got from auto_arima model in the training data and getting the summary of the fitted model**_

# In[21]:


model = SARIMAX(train['Milk Production'], order = (1,1,0), seasonal_order = (0,1,1,12))
result = model.fit()
result.summary()


# _**Plotting the Diagnostic plot for the fitted model to show the best fit of the model and saving it as PNG file**_

# In[22]:


result.plot_diagnostics(figsize = (15,5))
plt.subplots_adjust(hspace = 0.5)
plt.savefig('Diagnostic Plot of Best Model')
plt.show()


# _**Predicting the values using test data and renaming it as "Predictions"**_

# In[23]:


predictions = result.predict(len(train), len(train) + len(test) - 1, typ = 'levels').rename('Predictions')


# _**Comparing the predicted value with actual value in the test data**_

# In[24]:


for i in range(len(predictions)):
    print(f"predicted = {predictions[i]:<6.5}, expected = {test['Milk Production'][i]}")


# _**Plotting the line graph with the Predicted value and Test Data value and saving the graph as PNG file**_

# In[25]:


test['Milk Production'].plot(figsize = (12,6))
predictions.plot()
plt.title('Comparison of Predicted & Actual Test Data value')
plt.legend()
plt.savefig('Comparison of Predicted & Actual Test Data value.png')
plt.show()


# ## Model Evaluation

# _**Evaluating the model with the following metrics such as R2 Score, Mean Squared Error, Root Mean Squared Error, Mean Absolute Error and Mean Absolute Percentage Error for the predicted value and test data value**_

# In[26]:


print('Evaluation Results for Test Data : \n')
print(' Percenatge of R2 Score : {} %'.format(100*(r2_score(test['Milk Production'],predictions))),'\n')
print(' Mean Squared Error : ',mean_squared_error(test['Milk Production'],predictions),'\n')
print(' Root Mean Squared Error : ',sqrt(mean_squared_error(test['Milk Production'],predictions)),'\n')
print(' Mean Absolute Error : ',mean_absolute_error(test['Milk Production'],predictions),'\n')
print(' Mean Absolute Percentage Error : {0:.2f} %'.format(100*mean_absolute_percentage_error(test['Milk Production'],predictions)),'\n')


# ## Model Testing

# _**Creating the pickle file with the best model that gives high R2 score for the test data**_

# In[27]:


pickle.dump(result,open('Best Model.pkl','wb'))


# _**Loading the pickle file and predicting the whole data for testing**_

# In[28]:


final_model = pickle.load(open('Best Model.pkl','rb'))
fpred = final_model.predict(0, 167, typ = 'levels')


# _**Evaluating the model with the following metrics such as R2 Score, Mean Squared Error, Root Mean Squared Error, Mean Absolute Error and Mean Absolute Percentage Error for the predicted value and whole data**_

# In[29]:


print('Evaluation Results for whole Data : \n')
print(' Percenatge of R2 Score : {} %'.format(100*(r2_score(df['Milk Production'],fpred))),'\n')
print(' Mean Squared Error : ',mean_squared_error(df['Milk Production'],fpred),'\n')
print(' Root Mean Squared Error : ',sqrt(mean_squared_error(df['Milk Production'],fpred)),'\n')
print(' Mean Absolute Error : ',mean_absolute_error(df['Milk Production'],fpred),'\n')
print(' Mean Absolute Percentage Error : {0:.2f} %'.format(100*mean_absolute_percentage_error(df['Milk Production'],fpred)),'\n')


# ## Forecasting

# _**Forecasting the result for the future dates using the loaded model**_

# In[30]:


forecast = final_model.predict(start = '1976-01-01', end = '1980-12-01')


# _**Plotting the line graph with given data and predicted future data and saving it as PNG file**_

# In[31]:


plt.plot(df, color = 'red', label = 'Actual',alpha = 1)
plt.plot(forecast, color = 'green', label = 'Predicted', alpha = 0.7)
plt.xlabel('Month')
plt.ylabel('Quantity')
plt.legend()
plt.title('Actual Data with Forecast Data')
plt.savefig('Actual Data with Forecast Data.png')
plt.show()


# _**Making the forecasted value as dataframe, concating it with dataframe and exporting the DataFrame to [Comma Seperated Value]csv file**_

# In[32]:


forecast_df=pd.DataFrame(forecast)
forecast_df.rename(columns={'predicted_mean': 'Predicted Future Milk Production'}, inplace=True)
final = pd.concat([df,forecast_df],axis=1)
final['Predicted Future Milk Production'] = final['Predicted Future Milk Production'].round(2)
final.to_csv('Future Predicted Milk Production.csv')

