# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from fpdf import FPDF
import io

st.set_page_config(page_title="Data Analysis App", layout="wide")

# ------------------ PDF Helper ------------------
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, 'Data Analysis Report', 0, 1, 'C')
        self.ln(5)

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 8, title, 0, 1, 'L')
        self.ln(2)

    def chapter_body(self, body):
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 5, body)
        self.ln()

# ------------------ File Upload ------------------
st.sidebar.title("📂 Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.write("### Preview Data")
    st.dataframe(df.head())

    # ------------------ Analysis Choice ------------------
    analysis_choice = st.sidebar.selectbox(
        "Choose Analysis",
        ["Summary Statistics", "Correlation Matrix", "Regression Analysis", "Normality Check"]
    )

    pdf = PDF()
    pdf.add_page()

    if analysis_choice == "Summary Statistics":
        st.subheader("📊 Summary Statistics")
        summary = df.describe(include="all")
        st.dataframe(summary)

        pdf.chapter_title("Summary Statistics")
        pdf.chapter_body(summary.to_string())

    elif analysis_choice == "Correlation Matrix":
        st.subheader("🔗 Correlation Matrix")
        corr = df.corr(numeric_only=True)
        st.dataframe(corr)

        fig, ax = plt.subplots()
        cax = ax.matshow(corr, cmap="coolwarm")
        fig.colorbar(cax)
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.columns)), corr.columns)
        st.pyplot(fig)

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        pdf.chapter_title("Correlation Matrix")
        pdf.chapter_body(corr.to_string())
        pdf.image(buf, x=10, y=None, w=180)

    elif analysis_choice == "Regression Analysis":
        st.subheader("📈 Regression Analysis")
        num_cols = df.select_dtypes(include=np.number).columns.tolist()

        if len(num_cols) >= 2:
            x_var = st.selectbox("Select X variable", num_cols)
            y_var = st.selectbox("Select Y variable", num_cols)

            X = df[[x_var]].dropna()
            Y = df[y_var].dropna()
            common_idx = X.index.intersection(Y.index)
            X = X.loc[common_idx]
            Y = Y.loc[common_idx]

            model = LinearRegression()
            model.fit(X, Y)
            pred = model.predict(X)

            st.write(f"**R² Score:** {model.score(X, Y):.4f}")
            fig, ax = plt.subplots()
            ax.scatter(X, Y, color="blue", label="Actual")
            ax.plot(X, pred, color="red", label="Predicted")
            ax.set_xlabel(x_var)
            ax.set_ylabel(y_var)
            ax.legend()
            st.pyplot(fig)

            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            pdf.chapter_title(f"Regression Analysis ({x_var} vs {y_var})")
            pdf.chapter_body(f"R² Score: {model.score(X, Y):.4f}")
            pdf.image(buf, x=10, y=None, w=180)
        else:
            st.warning("Need at least two numeric columns for regression.")

    elif analysis_choice == "Normality Check":
        st.subheader("📏 Normality Check (Shapiro-Wilk Test)")
        col = st.selectbox("Select column", df.select_dtypes(include=np.number).columns)
        stat, p = stats.shapiro(df[col].dropna())
        st.write(f"**W-statistic:** {stat:.4f}, **p-value:** {p:.4f}")

        pdf.chapter_title(f"Normality Check: {col}")
        pdf.chapter_body(f"W-statistic: {stat:.4f}, p-value: {p:.4f}")

    # ------------------ Download PDF ------------------
    pdf_output = pdf.output(dest="S").encode("latin-1")
    st.download_button("📥 Download PDF Report", data=pdf_output, file_name="analysis_report.pdf", mime="application/pdf")

else:
    st.info("Upload a CSV or Excel file to begin.")
