import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import io
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors

# ----------------- Streamlit Page Config -----------------
st.set_page_config(page_title="Smart Data Analyst Toolkit", layout="wide")
sns.set_theme(style="whitegrid")

st.title("📊 Smart Data Analyst Toolkit + PDF Report")
st.markdown("Upload a CSV file to get **instant insights**, **colorful charts**, and a downloadable **PDF report**.")

# ----------------- File Upload -----------------
@st.cache_data
def load_csv(file):
    return pd.read_csv(file)

uploaded_file = st.file_uploader("📂 Upload CSV File", type="csv")

def save_plot(fig):
    """Save Matplotlib figure to BytesIO object."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf

if uploaded_file:
    try:
        df = load_csv(uploaded_file)
    except Exception as e:
        st.error(f"❌ Error reading the file: {e}")
        st.stop()

    st.success("✅ File uploaded successfully.")
    st.subheader("🔍 Data Preview")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    elements.append(Paragraph("Smart Data Analyst Toolkit Report", styles['Title']))
    elements.append(Spacer(1, 12))

    # ----------------- EDA -----------------
    elements.append(Paragraph("Exploratory Data Analysis", styles['Heading2']))
    summary = df.describe(include='all').transpose().fillna("").astype(str).reset_index()
    st.subheader("📌 Summary Statistics")
    st.write(summary)
    table_data = [summary.columns.tolist()] + summary.values.tolist()
    elements.append(Table(table_data, repeatRows=1))

    if df.isnull().sum().sum() > 0:
        msg = "⚠️ Dataset has missing values. Consider imputing them."
        st.warning(msg)
        elements.append(Paragraph(msg, styles['Normal']))
    else:
        msg = "✅ No missing values detected."
        st.info(msg)
        elements.append(Paragraph(msg, styles['Normal']))

    # Numeric plots
    if numeric_cols:
        elements.append(Paragraph("Numeric Distributions", styles['Heading3']))
        for col in numeric_cols:
            fig, ax = plt.subplots()
            sns.histplot(df[col].dropna(), kde=True, color='skyblue', ax=ax)
            ax.set_title(f"Distribution of {col}")
            st.pyplot(fig)
            elements.append(Image(save_plot(fig), width=400, height=300))

    # Categorical plots
    if categorical_cols:
        elements.append(Paragraph("Categorical Distributions", styles['Heading3']))
        for col in categorical_cols:
            fig, ax = plt.subplots()
            sns.countplot(y=col, data=df, palette='Set3', order=df[col].value_counts().index, ax=ax)
            ax.set_title(f"Frequency of {col}")
            st.pyplot(fig)
            elements.append(Image(save_plot(fig), width=400, height=300))

    # ----------------- Correlation -----------------
    if len(numeric_cols) >= 2:
        elements.append(Paragraph("Correlation Analysis", styles['Heading2']))
        corr = df[numeric_cols].corr()
        st.subheader("📉 Correlation Matrix")
        st.write(corr)
        table_data = [corr.columns.tolist()] + corr.round(3).values.tolist()
        elements.append(Table(table_data, repeatRows=1))

        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)
        elements.append(Image(save_plot(fig), width=400, height=300))

    # ----------------- Normality Test -----------------
    if numeric_cols:
        elements.append(Paragraph("Normality Tests", styles['Heading2']))
        for col in numeric_cols:
            stat, p = stats.shapiro(df[col].dropna())
            msg = f"{col}: W-stat={stat:.3f}, p-value={p:.3f}"
            st.write(msg)
            elements.append(Paragraph(msg, styles['Normal']))

    # ----------------- PDF Download -----------------
    doc.build(elements)
    pdf_buffer.seek(0)
    st.download_button(
        label="📥 Download PDF Report",
        data=pdf_buffer,
        file_name="data_analysis_report.pdf",
        mime="application/pdf"
    )

else:
    st.info("👆 Please upload a CSV file to begin analysis.")
