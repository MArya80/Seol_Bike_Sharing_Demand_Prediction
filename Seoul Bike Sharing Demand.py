#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/Legion/Desktop/Python/Datasets/SeoulBikeData.csv",encoding= 'unicode_escape')
df.head()


# In[2]:


from sklearn.preprocessing import LabelEncoder
import numpy as np

label_encoders = {}
categorical_columns = list()
for column in df.columns:
    if df.dtypes[column] in [np.int64, np.float64]:
        pass
    else:
        if column != 'Date':
            categorical_columns.append(column)
            Label_encoder = LabelEncoder()
            label_encoders[column] = Label_encoder
            
            numerical_column = Label_encoder.fit_transform(df[column])
            df[column] = numerical_column


# In[3]:


df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')


# <h2>Bike Rents per Month</h2>

# In[4]:


ndf = df.copy()
ndf.set_index('Date', inplace = True)

weekly_counts = ndf['Rented Bike Count'].resample('M').sum()

plt.figure(figsize = (15,5))
plt.bar(weekly_counts.index, weekly_counts.values,
       width = 10)
plt.show()


# The analysis suggests that there is a positive correlation between the season and the number of rented bikes, with higher bike rentals occurring during the spring and summer seasons. This observation may be attributed to various factors, including favorable weather conditions and increased outdoor activities during these seasons.

# <h2>Bike's Rents and Temperature </h2>

# In[5]:


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import warnings

Lr = LinearRegression()

temperatures = ndf['Temperature(°C)'].resample('M').median()
x = [i for i in range(len(weekly_counts))]
x = np.asanyarray(x).reshape(-1,1)

fig,axes = plt.subplots(nrows = 1,
                        ncols = 2,
                        figsize = (15,5))

def plot_reg(x,y,axes,degree = 2):
    poly = PolynomialFeatures(degree = degree)
    X_poly = poly.fit_transform(x)
    
    Lr.fit(X_poly,y)
    
    X_range = np.linspace(0,11,100).reshape(-1,1)
    X_range_poly = poly.fit_transform(X_range)
    Y_range = Lr.predict(X_range_poly)
    
    
    axes.plot(X_range,Y_range,
            color = 'blue')
    
axes[0].scatter(x,temperatures,
           color = 'red')
plot_reg(x,temperatures,axes[0],degree = 3)

warnings.filterwarnings('ignore', category=UserWarning)

date_format = '%Y-%m-%d'
axes[0].set_xticklabels([0] + [date.strftime(date_format) for date in [
    weekly_counts.index[i] for i in range(len(weekly_counts)) if i%2 == 0
]] + [0],
        rotation = 10)

axes[0].set_xlabel('Date')
axes[0].set_ylabel('Temperature(°C)')


axes[1].scatter(temperatures.values.reshape(1,-1),weekly_counts.values,
               color = 'red')
lr = LinearRegression()
poly = PolynomialFeatures(degree = 2)
X_poly = poly.fit_transform(temperatures.values.reshape(-1,1))
lr.fit(X_poly,weekly_counts.values)

X_range = np.linspace(-5,30,100)
X_range_poly = poly.fit_transform(X_range.reshape(-1,1))
y_range = lr.predict(X_range_poly)
plt.plot(X_range,y_range,
        color = 'blue')

axes[1].set_xlabel('Temperature(°C)')
axes[1].set_ylabel('Rented Bike Count')

plt.show()


# As hypothesized, there appears to be a positive correlation between temperature and bike rentals, with warmer temperatures coinciding with increased outdoor activities. The observed trend shown in the graph indicates that temperatures tend to increase as spring and summer approach, which may encourage individuals to engage in more outdoor activities. This finding is consistent with prior research on the relationship between temperature and physical activity, highlighting the importance of weather conditions in influencing human behavior and its potential impact on bike rental demand.

# <h2> Temperature in different dates </h2>

# In[6]:


import seaborn as sns
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

palette = list((sns.color_palette(palette='viridis', n_colors=len(ndf['Seasons'].unique())).as_hex()))

ndf['Seasons'] = label_encoders['Seasons'].inverse_transform(df['Seasons'])
seasons = label_encoders['Seasons'].inverse_transform([0,1,2,3])

fig = go.Figure()
i = 0
for season, color in zip(ndf['Seasons'].unique(),palette):
    info = df.loc[(df['Seasons'] == i)]
    fig.add_trace(go.Scatter(x = info['Date'],
                            y = info['Temperature(°C)'],
                            line_color = color,
                            name = seasons[i]))
    i+=1

fig.show()


# In[7]:


fig,axes = plt.subplots(nrows = 1, ncols = 2, figsize = (15,5))

holidays = ndf.loc[ndf['Holiday'] == 0]
mask = ndf[ndf['Holiday'] == 0].index.unique()
holiday_rented = holidays.resample('D')['Rented Bike Count'].sum()[mask]

# axes 0
axes[0].scatter(holiday_rented.index,holiday_rented.values,
               alpha = 0.8,
               label = 'Holiday')

not_holidays = ndf[ndf['Holiday'] == 1]
mask = ndf[ndf['Holiday'] == 1].index.unique()
not_holidays_rented = not_holidays.resample('D')['Rented Bike Count'].sum()[mask]

axes[0].scatter(not_holidays_rented.index,not_holidays_rented.values,
                alpha = 0.5,
               label = 'Not Holiday')

axes[0].set_xlabel('Date')
axes[0].set_ylabel('Rented Bike Count')
axes[0].legend(loc = 'upper left')

#axes 1
import datetime

def season_holiday(year,month_start,month_end,day,season,axes):
    season_start = datetime.date(year = year, month = month_start, day = day)
    season_end = datetime.date(year = year, month = month_end, day = day)
    
    season_ndf = ndf[(ndf.index.date >= season_start) & (ndf.index.date <= season_end)]
    season_rides = season_ndf.resample('D')['Rented Bike Count'].sum()
    
    axes.scatter(season_rides.index,season_rides.values,
                label = season,
                alpha = 0.5)

season_holiday(year = 2018,month_start = 3,month_end = 6,day = 21,season = 'Spring',axes = axes[1])
season_holiday(year = 2018,month_start = 6,month_end = 9,day = 21,season = 'Summer',axes = axes[1])
season_holiday(year = 2018,month_start = 9,month_end = 12,day = 21,season = 'Autumn',axes = axes[1])
season_holiday(year = 2018,month_start = 1,month_end = 3,day = 21,season = 'Winter',axes = axes[1])

axes[1].legend()
axes[1].tick_params(axis = 'x', rotation = 45)
plt.show()


# The plots above suggest that people tend to participate in more outdoor activities during holidays than on regular days. Additionally, analysis of temperature data across different seasons indicates that spring and summer typically have more moderate weather conditions, which may facilitate increased outdoor activity. Interestingly, the peak of bike rentals corresponds to the peak of summer, suggesting that seasonal factors may play a role in bike rental demand. It is possible that students, who have more free time in the summer months, contribute to the observed increase in bike rentals during this period. Overall, these findings highlight the influence of seasonality and holiday periods on outdoor activity and bike rental demand

# <h2>Daytime and Bike's Rent </h2>

# In[8]:


from sklearn.preprocessing import MinMaxScaler

fig, axes = plt.subplots(figsize = (15,5))

hour_rents = df.groupby(['Hour']).median()['Rented Bike Count']
axes.scatter(hour_rents.index,hour_rents,
                color = 'green',
               alpha = 0.5)

poly = PolynomialFeatures(degree = 3)
X_poly = poly.fit_transform(np.asarray(hour_rents.index).reshape(-1,1))

lr = LinearRegression()
lr.fit(X_poly,hour_rents.values)

X_range = np.linspace(0,23).reshape(-1,1)
X_range_poly = poly.fit_transform(X_range)
Y_range = lr.predict(X_range_poly)

scaler = MinMaxScaler(feature_range = (170,1700))
solar_hour = df.groupby('Hour')['Humidity(%)'].median()
scaler.fit(solar_hour.values.reshape(-1,1))
 
solar_hour_values = scaler.transform(solar_hour.values.reshape(-1,1))
axes.plot(solar_hour.index,solar_hour_values,
            label = 'Humidity(%)')


axes.plot(X_range,Y_range)
axes.set_xlabel('Hour')
axes.set_ylabel('Median Rented Bike Counts')

axes.legend()



plt.show()


# Furthermore, the analysis suggests that individuals tend to spend more time outdoors in the evening and nighttime hours, which corresponds with the times when bike rentals peak. This finding supports the hypothesis that bike rentals are influenced by individuals' free time and availability, with rentals increasing during periods when people have more leisure time.

# In[9]:


import re
def date_to_integer(my_string):
    my_string = re.sub('[^0-9]', '', my_string)
    return int(my_string)

df['Date'] = df['Date'].astype('str')
df['Date'] = df['Date'].apply(date_to_integer)
    
df.head()


# <h2> Matrix Correlation </h2>

# In[10]:


import seaborn as sns
fig,ax = plt.subplots(figsize = (15,10))
sns.heatmap(df.corr(),annot = True)

plt.show()


# <h2> Feature Selection </h2>

# In[11]:


features = []
corr = df.corr()['Rented Bike Count'][1:]
for column in corr.index[1:]:
    if df[column].dtype!='<M8[ns]' and abs(corr[column]) >= 0.2:
        features.append(column)


# <h2> Split the Dataset into Train and Test datas </h2>

# In[12]:


from sklearn.model_selection import train_test_split

X = df[features]
y = np.sqrt(df['Rented Bike Count'])

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2,random_state = 15)


# In[13]:


model_train,model_test = {},dict()


# <h2>Linear Regression </h2>

# In[14]:


lr = LinearRegression()
lr.fit(X_train, y_train)


# <h3> Linear Regression Metric Scores </h3>

# In linear regression, there are several metrics that can be used to evaluate the performance of the model. One of them is Mean Absolute Error and the other one is R-squared.<br><b><i>MAE</i></b> is a measure of the average absolute difference between the predicted and actual values. It is calculated by taking the average of the absolute differences between the predicted and actual values.<br>
# <b><i>R2</i></b> is a measure of how well the model fits the data. It is the proportion of the variance in the dependent variable that is explained by the independent variables. R2 ranges from 0 to 1, with higher values indicating a better fit.

# ![1%20G8xEXSvEUUdpENTr6jVE6w.webp](attachment:1%20G8xEXSvEUUdpENTr6jVE6w.webp)

# In[15]:


from sklearn.metrics import mean_absolute_error

predict_train = lr.predict(X_train)
predict_test = lr.predict(X_test)

mae_train =  mean_absolute_error(y_train,predict_train)
mae_test = mean_absolute_error(y_test,predict_test)

print('Mean Absolute Error on Training set is {:.2f}\nMean Absolute Error on Test set is {:.2f}'.format(mae_train,mae_test))


# The minimal difference between the Mean Absolute Error (MAE) in the training and test datasets suggests that the trained model is effectively generalizing and performing quite well on unseen data.

# In[16]:


from sklearn.metrics import r2_score

r2_train = r2_score(y_train,predict_train)
r2_test = r2_score(y_test, predict_test)

model_train[type(lr).__name__] = r2_train
model_test[type(lr).__name__] = r2_test

print('R2 Error on Training set is {:.2f}\nR2 Error on Test set is {:.2f}'.format(r2_train,r2_test))


# To further evaluate the model, we employed the R-squared method. The obtained score of 0.65, though indicative of some degree of correlation, To some degrees does meet the desired level of model performance. To gain insight into this result and the underlying calculation, we refer to the following two figures

# ![0%208rFYfZJfJZpW2cEV.webp](attachment:0%208rFYfZJfJZpW2cEV.webp)

# ![0%20zgSKANTyadZmfhDJ.gif](attachment:0%20zgSKANTyadZmfhDJ.gif)

# <center> (source: http://www.mathsisfun.com/data/correlation.html) </center>

# The reason for the suboptimal model performance may be attributed to the complexity of predicting the rental bike count at each hour of a given day, which requires a significant amount of data. While the model has performed reasonably well, further analysis is required to identify opportunities for improvement. To this end, we plan to train other models on this dataset and evaluate their performance to identify the best approach for predicting bike rentals.

# <h2>Lasso Regression</h2>

# Lasso Regression is a variation of linear regression that is designed to address some of the issues associated with traditional linear regression models. One of the key challenges with linear regression is overfitting, particularly when there are a large number of independent variables. Lasso Regression seeks to address this challenge by adding a regularization term called L1 regularization, also known as the Lasso Penalty

# This penalty term is the sum of the absolute values of the coefficients multiplied by a regularization parameter lambda. By adding this term, Lasso Regression penalizes large coefficients and can make some of the coefficients equal to zero, effectively performing feature selection. 

# ![C12624_04_59.jpg](attachment:C12624_04_59.jpg)

# In[17]:


from sklearn.linear_model import Lasso

lr = Lasso(alpha = 0.1)
lr.fit(X_train,y_train)


# In[18]:


predict_train = lr.predict(X_train)
predict_test = lr.predict(X_test)

mae_train =  mean_absolute_error(y_train,predict_train)
mae_test = mean_absolute_error(y_test,predict_test)

print('Mean Absolute Error on Training set is {:.2f}\nMean Absolute Error on Test set is {:.2f}'.format(mae_train,mae_test))


# In[19]:


r2_train = r2_score(y_train,predict_train)
r2_test = r2_score(y_test, predict_test)

model_train[type(lr).__name__] = r2_train
model_test[type(lr).__name__] = r2_test

print('R2 Error on Training set is {:.2f}\nR2 Error on Test set is {:.2f}'.format(r2_train,r2_test))


# The similarity of the results obtained from the multiple linear regression model to those of the simple linear regression model indicates that the additional independent variables included in the model did not significantly impact the prediction performance.

# <h2>Decision Tree Based Regression</h2>

# The decision tree-based regression approach employs decision trees to model the relationship between one or more independent variables and a dependent variable. Decision trees are machine learning algorithms that are commonly used for solving both classification and regression problems in Python.
# 

# In[20]:


from sklearn import tree

lr = tree.DecisionTreeRegressor(criterion = 'friedman_mse')
lr.fit(X_train,y_train)


# In[21]:


predict_train = lr.predict(X_train)
predict_test = lr.predict(X_test)

mae_train =  mean_absolute_error(y_train,predict_train)
mae_test = mean_absolute_error(y_test,predict_test)

print('Mean Absolute Error on Training set is {:.2f}\nMean Absolute Error on Test set is {:.2f}'.format(mae_train,mae_test))


# In[22]:


r2_train = r2_score(y_train,predict_train)
r2_test = r2_score(y_test, predict_test)

model_train[type(lr).__name__] = r2_train
model_test[type(lr).__name__] = r2_test

print('R2 Error on Training set is {:.2f}\nR2 Error on Test set is {:.2f}'.format(r2_train,r2_test))


# Due to the nature of the decision tree method, it often performs well on the training set, but its performance on the test set may not be as satisfactory compared to other methods. In our case, we observed a slight drop in performance on the test set compared to the training set. However, it should be noted that the performance of the decision tree-based model on the test set was better than that of other models previously trained

# <h2> Support Vector Regression </h2>

# Support Vector Regression (SVR) is a supervised learning algorithm that belongs to the Support Vector Machine (SVM) family. SVR is specifically designed for regression problems and employs a linear model to find the hyperplane that can optimally separate the data points while minimizing the classification error.

# In[23]:


from sklearn.svm import SVR

lr = SVR()
lr.fit(X_train,y_train)


# In[24]:


predict_train = lr.predict(X_train)
predict_test = lr.predict(X_test)

mae_train =  mean_absolute_error(y_train,predict_train)
mae_test = mean_absolute_error(y_test,predict_test)

print('Mean Absolute Error on Training set is {:.2f}\nMean Absolute Error on Test set is {:.2f}'.format(mae_train,mae_test))


# In[25]:


r2_train = r2_score(y_train,predict_train)
r2_test = r2_score(y_test, predict_test)

model_train[type(lr).__name__] = r2_train
model_test[type(lr).__name__] = r2_test

print('R2 Error on Training set is {:.2f}\nR2 Error on Test set is {:.2f}'.format(r2_train,r2_test))


# <h2>Conclusion</h2>

# The analysis indicates that several factors can influence the number of rented bikes, with temperature and weather conditions being primary factors, followed by hour and season. Temperature has a direct correlation with the season, which in turn affects bike rental demand. Additionally, the hour of the day is found to be associated with the number of rented bikes, with more rentals occurring during the evening and nighttime hours, when people have more free time.

# We utilized machine learning models to predict the number of rented bikes, and found that the decision tree regressor achieved the highest accuracy of approximately 75% on the test set. In addition to the decision tree regressor, we trained other models and the results are shown in the plot below.

# In[26]:


plt.subplots(nrows = 1, ncols = 1,
            figsize = (19,10))

width = 0.25
r1 = np.arange(len(model_test.keys()))
r2 = [x + width for x in r1]

train_bars = plt.bar(r1,
       [round(model_train[key],2) for key in model_test.keys()],            
        width = width,
       label = 'Train Accuracy',
       color = 'green')
test_bars = plt.bar(r2,
       [round(model_test[key],2) for key in model_test.keys()],
        width = width
       ,label = 'Test Accuracy',
       color = 'orange')

plt.xticks([r + width/2 for r in range(4)],model_test.keys())

plt.bar_label(train_bars,padding = 4)
plt.bar_label(test_bars, padding = 4)

 

plt.legend()
plt.show()


# Sources:<br>
# https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand <br>
# https://pieriantraining.com/7-machine-learning-regression-algorithms-python/
