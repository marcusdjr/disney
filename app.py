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
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
import streamlit as st
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier, LazyRegressor
import pickle
import base64
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
        plt.hist(st.session_state.dataframe[column_to_plot], bins=30, color='#4682B4', edgecolor='black')  
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

def display_ml_section():
    st.subheader("Mickey's Prediction Lab")

    models = ["Choose Model", "LazyPredict - Classification", "LazyPredict - Regression"]
    model_name = st.selectbox("Select Model", models)

    if "dataframe" in st.session_state:
        all_features = st.session_state.dataframe.columns.tolist()
        target_col = st.selectbox('Select Target Column:', all_features)
        available_features = [col for col in all_features if col != target_col]  
        
        if 'selected_features' not in st.session_state:
            st.session_state.selected_features = available_features
        
        valid_defaults = [feat for feat in st.session_state.selected_features if feat in available_features]
        
        st.session_state.selected_features = st.multiselect('Select Features for Training:', available_features, default=valid_defaults)
        
        features = st.session_state.dataframe[st.session_state.selected_features]
        X = features.values
        y = st.session_state.dataframe[target_col].values

        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        except Exception as e:
            st.error(f"Error during data split: {e}")
            return

        if st.button("Train Model"):
            if model_name == "LazyPredict - Classification":
                with st.spinner("Training using LazyPredict Classification..."):
                    try:
                        clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
                        models, predictions = clf.fit(X_train, X_test, y_train, y_test)
                        st.write(models)
                    except Exception as e:
                        st.error(f"Error during LazyPredict classification: {e}")

            elif model_name == "LazyPredict - Regression":
                with st.spinner("Training using LazyPredict Regression..."):
                    try:
                        reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
                        models, predictions = reg.fit(X_train, X_test, y_train, y_test)
                        st.write(models)
                    except Exception as e:
                        st.error(f"Error during LazyPredict regression: {e}")

if 'page' not in st.session_state:
    st.session_state.page = "upload"

if st.session_state.page == "upload":
    display_upload_section()
elif st.session_state.page == "eda":
    display_eda_section()
elif st.session_state.page == "ml":
    display_ml_section()

