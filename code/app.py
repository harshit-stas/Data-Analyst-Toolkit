import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import os

st.set_page_config(page_title="Data Analyst Toolkit", layout="wide")

# Inject custom CSS
css_path = os.path.join(os.path.dirname(__file__), "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("📊 Data Analyst Toolkit")
st.markdown("Upload your dataset (CSV) and start analyzing it with no code required!")

# File uploader
file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.success("✅ File uploaded successfully!")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Sidebar Options
    st.sidebar.header("Select Analysis")
    analysis_type = st.sidebar.radio("What would you like to do?", [
        "Summary Stats", "Correlation Matrix", "Data Visualization", "Linear Regression"])

    # --- Summary Stats ---
    if analysis_type == "Summary Stats":
        st.subheader("📈 Summary Statistics")
        st.write(df.describe(include='all'))

    # --- Correlation Matrix ---
    elif analysis_type == "Correlation Matrix":
        st.subheader("🔗 Correlation Matrix (Pearson)")
        if numeric_cols:
            corr = df[numeric_cols].corr()
            st.dataframe(corr)
            fig, ax = plt.subplots()
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
        else:
            st.warning("No numeric columns found for correlation.")

    # --- Visualization ---
    elif analysis_type == "Data Visualization":
        st.subheader("📊 Data Visualization")
        chart_type = st.selectbox("Select chart type", ["Histogram", "Box Plot", "Bar Chart", "Scatter Plot"])
        
        if chart_type == "Histogram":
            col = st.selectbox("Select numeric column", numeric_cols)
            if col:
                fig, ax = plt.subplots()
                sns.histplot(df[col].dropna(), kde=True, ax=ax)
                st.pyplot(fig)

        elif chart_type == "Box Plot":
            y = st.selectbox("Y-axis (numeric)", numeric_cols)
            x = st.selectbox("X-axis (categorical)", categorical_cols)
            if x and y:
                fig, ax = plt.subplots()
                sns.boxplot(x=df[x], y=df[y], ax=ax)
                st.pyplot(fig)

        elif chart_type == "Bar Chart":
            col = st.selectbox("Select categorical column", categorical_cols)
            if col:
                fig, ax = plt.subplots()
                df[col].value_counts().plot(kind='bar', ax=ax)
                st.pyplot(fig)

        elif chart_type == "Scatter Plot":
            x = st.selectbox("X-axis", numeric_cols, key="scatter_x")
            y = st.selectbox("Y-axis", [col for col in numeric_cols if col != x], key="scatter_y")
            if x and y:
                fig, ax = plt.subplots()
                sns.scatterplot(x=df[x], y=df[y], ax=ax)
                st.pyplot(fig)

    # --- Regression ---
    elif analysis_type == "Linear Regression":
        st.subheader("📉 Linear Regression")
        y = st.selectbox("Dependent variable (Y)", numeric_cols, key="reg_y")
        x = st.selectbox("Independent variable (X)", [col for col in numeric_cols if col != y], key="reg_x")
        if x and y:
            model = smf.ols(f'{y} ~ {x}', data=df).fit()
            st.text(model.summary())
            fig, ax = plt.subplots()
            sns.regplot(x=df[x], y=df[y], ax=ax)
            st.pyplot(fig)
else:
    st.info("Upload a CSV file to begin.")
