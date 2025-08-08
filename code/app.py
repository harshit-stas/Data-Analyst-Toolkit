import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
import io
from fpdf import FPDF
import tempfile
import os

st.set_page_config(page_title="Universal Data Analysis Tool", layout="wide")
st.title("📊 Universal Data Analysis & PDF Report Generator")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    st.write("### Preview of Data", df.head())

    # Select analysis
    analysis_options = [
        "Summary Statistics",
        "Correlation Heatmap",
        "Regression Analysis",
        "Normality Check",
        "Custom Plot"
    ]
    selected_analysis = st.multiselect("Choose Analysis to Perform", analysis_options)

    # Prepare PDF
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    def add_table_to_pdf(pdf, dataframe, title):
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, title, ln=True)
        pdf.set_font("Arial", '', 10)
        col_width = pdf.w / (len(dataframe.columns) + 1)
        pdf.ln(5)

        # Column names
        for col in dataframe.columns:
            pdf.cell(col_width, 8, str(col), border=1)
        pdf.ln()

        # Rows
        for _, row in dataframe.iterrows():
            for item in row:
                pdf.cell(col_width, 8, str(round(item, 3) if isinstance(item, (int, float)) else str(item)), border=1)
            pdf.ln()

    def add_image_to_pdf(pdf, fig, title):
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, title, ln=True)
        tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        fig.savefig(tmpfile.name, bbox_inches="tight")
        pdf.image(tmpfile.name, w=180)
        plt.close(fig)
        tmpfile.close()
        os.unlink(tmpfile.name)

    # Run analyses
    if "Summary Statistics" in selected_analysis:
        summary_stats = df.describe().T
        st.write("### Summary Statistics", summary_stats)
        add_table_to_pdf(pdf, summary_stats, "Summary Statistics")

    if "Correlation Heatmap" in selected_analysis:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
        add_image_to_pdf(pdf, fig, "Correlation Heatmap")

    if "Regression Analysis" in selected_analysis:
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        if len(num_cols) >= 2:
            x_col = st.selectbox("Select independent variable (X)", num_cols)
            y_col = st.selectbox("Select dependent variable (Y)", num_cols)
            if x_col and y_col:
                X = df[[x_col]].dropna()
                y = df[y_col].dropna()
                model = LinearRegression()
                model.fit(X, y)
                slope = model.coef_[0]
                intercept = model.intercept_
                st.write(f"**Regression Equation:** y = {slope:.3f}x + {intercept:.3f}")
                fig, ax = plt.subplots()
                ax.scatter(X, y, color='blue')
                ax.plot(X, model.predict(X), color='red')
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                st.pyplot(fig)
                add_image_to_pdf(pdf, fig, f"Regression: {y_col} vs {x_col}")

    if "Normality Check" in selected_analysis:
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        col = st.selectbox("Select column for normality test", num_cols)
        stat, p = stats.shapiro(df[col].dropna())
        st.write(f"**Shapiro-Wilk Test:** W = {stat:.3f}, p = {p:.3f}")
        interpretation = "Data looks normally distributed." if p > 0.05 else "Data does not look normally distributed."
        st.write("📌 Interpretation:", interpretation)
        pdf.set_font("Arial", '', 12)
        pdf.multi_cell(0, 8, f"Normality Check ({col}):\nW = {stat:.3f}, p = {p:.3f}\nInterpretation: {interpretation}")

    if "Custom Plot" in selected_analysis:
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        x_axis = st.selectbox("X-axis", num_cols)
        y_axis = st.selectbox("Y-axis", num_cols)
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x=x_axis, y=y_axis, ax=ax)
        st.pyplot(fig)
        add_image_to_pdf(pdf, fig, f"Custom Plot: {y_axis} vs {x_axis}")

    # Export PDF
    if st.button("📄 Generate PDF Report"):
        pdf_output = io.BytesIO()
        pdf.output(pdf_output)
        pdf_output.seek(0)
        st.download_button(
            label="Download PDF Report",
            data=pdf_output,
            file_name="analysis_report.pdf",
            mime="application/pdf"
        )
