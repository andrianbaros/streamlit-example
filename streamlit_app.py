#library
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm

from factor_analyzer import FactorAnalyzer
from sklearn import metrics
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from streamlit_option_menu import option_menu

#load data
@st.cache
def load_data(url):
    df_hour = pd.read_csv(url)
    return df_hour

# Load data
df_data_hour = load_data('hour.csv')
df_data_day = load_data('day.csv')

#mengakses data
st.write(df_data_hour.info())
st.write(df_data_hour.isnull().sum())

#cleaning data
df_data_hour = df_data_hour.dropna(how='any',axis=0)
st.write("Null values removed successfully.")
st.write(df_data_hour.isnull().sum())
st.write(df_data_hour.duplicated().any())

#explore data
df_data_hour.rename(columns={'instant':'rec_id',
                        'dteday':'datetime',
                        'holiday':'is_holiday',
                        'workingday':'is_workingday',
                        'weathersit':'weather_condition',
                        'hum':'humidity',
                        'mnth':'month',
                        'cnt':'total_count',
                        'hr':'hour',
                        'yr':'year'},inplace=True)


df_data_hour['datetime'] = pd.to_datetime(df_data_hour.datetime)


df_data_hour['season'] = df_data_hour.season.astype('category')
df_data_hour['is_holiday'] = df_data_hour.is_holiday.astype('category')
df_data_hour['weekday'] = df_data_hour.weekday.astype('category')
df_data_hour['weather_condition'] = df_data_hour.weather_condition.astype('category')
df_data_hour['is_workingday'] = df_data_hour.is_workingday.astype('category')
df_data_hour['month'] = df_data_hour.month.astype('category')
df_data_hour['year'] = df_data_hour.year.astype('category')
df_data_hour['hour'] = df_data_hour.hour.astype('category')

sns.set_style('whitegrid')
sns.set_context('talk')
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (30, 10),
          'axes.labelsize': 'x-large',
          'axes.titlesize':'x-large',
          'xtick.labelsize':'x-large',
          'ytick.labelsize':'x-large'}

plt.rcParams.update(params)

fig,ax = plt.subplots()
sns.pointplot(data=df_data_hour[['hour',
                           'total_count',
                           'weekday']],
              x='hour',
              y='total_count',
              hue='weekday',
              ax=ax)
ax.set(title="Distribusi hitungan per jam pada hari kerja")
plt.show()
