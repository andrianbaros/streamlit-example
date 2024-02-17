import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use("Agg")  # Menggunakan mode tanpa GUI untuk Matplotlib

# Data Wrangling

@st.cache
def load_data_hour():
    df_data_hour = pd.read_csv('hour.csv')
    return df_data_hour

def main():
    st.title("Data Analysis with Streamlit")

    # Load data
    df_data_hour = load_data_hour()

    # Display data info
    st.subheader("Data Info")
    st.write(df_data_hour.info())

    # Assessing Data
    st.subheader("Assessing Data")
    st.write(df_data_hour.isnull().sum())

    # Cleaning Data
    st.subheader("Cleaning Data")
    df_data_hour = df_data_hour.dropna(how='any', axis=0)
    st.write("Null values removed successfully.")
    st.write(df_data_hour.isnull().sum())

    # EDA
    st.subheader("Exploratory Data Analysis (EDA)")

    # Visualizations
    st.subheader("Visualizations")

    st.write("Example visualization:")
    fig, ax = plt.subplots()
    sns.pointplot(data=df_data_hour[['hour', 'total_count', 'weekday']],
                  x='hour', y='total_count', hue='weekday', ax=ax)
    ax.set(title="Distribution of bike counts per hour on weekdays")
    st.pyplot(fig)

if __name__ == "__main__":
    main()
