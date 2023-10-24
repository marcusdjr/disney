import streamlit as st
from streamlit_lottie import st_lottie
import json
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import time
import base64
from PIL import Image
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#############################
 #Setting page to wide view#
#############################

st.set_page_config(layout="centered")

#############################
 #Loading lottie images#
#############################
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

lottie1 = load_lottiefile("lottie1.json")
lottieload = load_lottiefile("lottieload.json")


col1, col2, col3 = st.sidebar.columns([1,8,1])
with col1:
    st.write("")
with col2:

    lottie1 = load_lottiefile("lottie1.json")
    st_lottie(lottie1, key="sidebar-lottie", height=300)
with col3:
    st.write("")

#############################
        #SideBar#
#############################
st.sidebar.markdown("## About Disney's BYOD App")
st.sidebar.markdown("This application allows users to bring their own dataset for analysis, visualizations and machine learning. Please feel free to upload your own dataset, and then explore the data in various ways.")
st.sidebar.info("Find more information and the source code for this app on [Github](https://github.com/marcusdjr).", icon="ℹ️")

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)
#############################
    #Uploading Feature#
#############################
def display_upload_section():
    st.subheader("Mickey’s Data Upload Station")
    uploaded_file = st.file_uploader("Pleasse upload a CSV file", type="csv")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.dataframe = df
        
        if not "lottie_displayed" in st.session_state:
            with st.spinner("Loading..."):
                lottie_animation = load_lottiefile("lottieload.json")
                st_lottie(lottie_animation, height=300)
                time.sleep(4)
            st.session_state.lottie_displayed = True
            st.experimental_rerun()
        else:
            if st.button("Next", key="btn_upload"):
                st.session_state.page = "eda"
#############################
      #EDA Feature#
#############################
def display_eda_section():
    st.subheader("Mickey's Data Dive Station")
    

    st.markdown("### Display Data")
    selected_columns = st.multiselect('Select columns to display:', st.session_state.dataframe.columns)
    if selected_columns:
        st.write(st.session_state.dataframe[selected_columns].head())
    
 
    st.markdown("### Missing Values Visualization")
    missing_data = st.session_state.dataframe.isnull().sum()
    if missing_data.sum() > 0:  
        st.bar_chart(missing_data)
    else:
        st.write("No missing values found!")
    
 
    st.markdown("### Numerical Data Statistics")
    selected_numerical = st.multiselect('Select numerical columns for stats:', st.session_state.dataframe.select_dtypes(include=[np.number]).columns)
    if selected_numerical:
        st.write(st.session_state.dataframe[selected_numerical].describe())
    

    st.markdown("### Numerical Data Visualization")
    column_to_plot = st.selectbox('Select column to plot:', st.session_state.dataframe.select_dtypes(include=[np.number]).columns)
    plot_type = st.selectbox('Select plot type:', ['Histogram', 'Box Plot'])
    
    if plot_type == 'Histogram':
        st.set_option('deprecation.showPyplotGlobalUse', False)
        plt.hist(st.session_state.dataframe[column_to_plot], bins=30, color='#4682B4', edgecolor='black')  # added more bins and colors
        plt.title(f'Histogram of {column_to_plot}')
        plt.xlabel(column_to_plot)
        plt.ylabel('Frequency')
        st.pyplot()
    elif plot_type == 'Box Plot':
        plt.boxplot(st.session_state.dataframe[column_to_plot])
        plt.title(f'Box Plot of {column_to_plot}')
        st.pyplot()
    

    st.markdown("### Categorical Data Visualization")
    column_to_visualize = st.selectbox('Select categorical column to visualize:', st.session_state.dataframe.select_dtypes(include=['object']).columns)
    value_counts = st.session_state.dataframe[column_to_visualize].value_counts()
    plt.figure(figsize=(10, 6))
    plt.bar(value_counts.index, value_counts.values, color='#4682B4')
    plt.title(f'Bar Chart of {column_to_visualize}')
    plt.ylabel('Count')
    st.pyplot()


    st.markdown("### Correlation Matrix")
    selected_corr_features = st.multiselect('Select features for correlation matrix:', st.session_state.dataframe.select_dtypes(include=[np.number]).columns)
    if selected_corr_features:
        corr_matrix = st.session_state.dataframe[selected_corr_features].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        st.pyplot()


    st.markdown("### Scatter Plots")
    x_axis = st.selectbox('Select feature for x-axis:', st.session_state.dataframe.select_dtypes(include=[np.number]).columns)
    y_axis = st.selectbox('Select feature for y-axis:', st.session_state.dataframe.select_dtypes(include=[np.number]).columns)
    plt.figure(figsize=(10, 6))
    plt.scatter(st.session_state.dataframe[x_axis], st.session_state.dataframe[y_axis], alpha=0.5, color='#4682B4')
    plt.title(f'Scatter plot: {x_axis} vs {y_axis}')
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    st.pyplot()
    

    if not "lottie_displayed_eda" in st.session_state:
        lottie_animation = load_lottiefile("lottieload2.json")
        st_lottie(lottie_animation, height=200)
        st.session_state.lottie_displayed_eda = True
        time.sleep(4)
        st.experimental_rerun()
    else:
        if st.button("Proceed to ML Training", key="btn_eda_next"):
            st.session_state.page = "ml"
#############################
#Machine Learning Feature#
#############################
def train_model(model_name, X_train, y_train):
    if model_name == "Decision Tree":
        model = DecisionTreeClassifier()
    elif model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=10000) # Added increased max_iter for better convergence
    else:
        raise ValueError(f"Model {model_name} not supported")

    model.fit(X_train, y_train)
    return model


def display_ml_section():
    st.subheader("Train ML Model")

    model_name = st.selectbox("Select Model", ["Decision Tree", "Logistic Regression"])

    if "dataframe" in st.session_state:
        features = st.session_state.dataframe.dropna(axis=1)
        target_col = st.selectbox('Select Target Column:', features.columns)
        features = features.drop(target_col, axis=1)

        X = features.values
        y = st.session_state.dataframe[target_col].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if st.button("Train Model"):
            with st.spinner("Training..."):
                model = train_model(model_name, X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                st.success(f"Model Trained. Test Accuracy: {accuracy:.2f}")

if 'page' not in st.session_state:
    st.session_state.page = "upload"

if st.session_state.page == "upload":
    display_upload_section()
elif st.session_state.page == "eda":
    display_eda_section()
elif st.session_state.page == "ml":
    display_ml_section()


