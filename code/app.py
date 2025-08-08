# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from fpdf import FPDF
import io

st.set_page_config(page_title="Data Analysis & PDF Report", layout="wide")

st.title("📊 Data Analysis App with PDF Export")

uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

analysis_choice = st.multiselect(
    "Select Analyses to Perform",
    ["Summary Statistics", "Correlation Heatmap", "Regression Analysis", "Normality Check", "Custom Plot"]
)

if uploaded_file:
    # Load data
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Preview of Data")
    st.dataframe(df.head())

    # PDF setup
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    def add_table_to_pdf(dataframe, title):
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, title, ln=True)
        pdf.ln(5)
        pdf.set_font("Arial", size=10)
        col_width = pdf.w / (len(dataframe.columns) + 1)
        # Header
        for col in dataframe.columns:
            pdf.cell(col_width, 8, str(col), border=1)
        pdf.ln()
        # Rows
        for _, row in dataframe.iterrows():
            for item in row:
                pdf.cell(col_width, 8, str(round(item, 3)) if isinstance(item, (float, int)) else str(item), border=1)
            pdf.ln()
        pdf.ln(5)

    def add_image_to_pdf(fig, title):
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, title, ln=True)
        pdf.ln(5)
        img_bytes = io.BytesIO()
        fig.savefig(img_bytes, format="png", bbox_inches="tight")
        img_bytes.seek(0)
        pdf.image(img_bytes, x=10, w=180)
        pdf.ln(10)

    # Perform analyses
    if "Summary Statistics" in analysis_choice:
        st.subheader("Summary Statistics")
        summary = df.describe(include="all").transpose()
        st.dataframe(summary)
        pdf.add_page()
        add_table_to_pdf(summary.reset_index(), "Summary Statistics")

    if "Correlation Heatmap" in analysis_choice:
        st.subheader("Correlation Heatmap")
        corr = df.corr(numeric_only=True)
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
        pdf.add_page()
        add_image_to_pdf(fig, "Correlation Heatmap")

    if "Regression Analysis" in analysis_choice:
        st.subheader("Regression Analysis")
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        x_var = st.selectbox("Select X variable", numeric_cols)
        y_var = st.selectbox("Select Y variable", numeric_cols)
        if x_var and y_var:
            X = sm.add_constant(df[x_var])
            y = df[y_var]
            model = sm.OLS(y, X).fit()
            st.text(model.summary())
            pdf.add_page()
            pdf.set_font("Arial", size=10)
            pdf.multi_cell(0, 8, f"Regression Analysis ({y_var} ~ {x_var})\n\n{model.summary()}")

    if "Normality Check" in analysis_choice:
        st.subheader("Normality Check (Shapiro-Wilk)")
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        col = st.selectbox("Select column for Normality Test", numeric_cols)
        if col:
            stat, p = stats.shapiro(df[col].dropna())
            st.write(f"**W-statistic:** {stat:.3f}, **p-value:** {p:.3f}")
            interpretation = "Data looks Gaussian (fail to reject H0)" if p > 0.05 else "Data does not look Gaussian (reject H0)"
            st.write(f"Interpretation: {interpretation}")
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 8, f"Normality Check for {col}\nW-statistic: {stat:.3f}, p-value: {p:.3f}\n{interpretation}")

    if "Custom Plot" in analysis_choice:
        st.subheader("Custom Scatter Plot")
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        x_var = st.selectbox("X-axis", numeric_cols, key="custom_x")
        y_var = st.selectbox("Y-axis", numeric_cols, key="custom_y")
        if x_var and y_var:
            fig, ax = plt.subplots()
            sns.scatterplot(x=df[x_var], y=df[y_var], ax=ax)
            st.pyplot(fig)
            pdf.add_page()
            add_image_to_pdf(fig, f"Custom Scatter Plot: {x_var} vs {y_var}")

    # Download PDF
    pdf_output = pdf.output(dest='S').encode('latin1')
    st.download_button("📥 Download PDF Report", data=pdf_output, file_name="data_analysis_report.pdf", mime="application/pdf")
