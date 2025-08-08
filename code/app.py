# smart_data_analyst_toolkit_with_pdf.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import io
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors

# ----------------- Page config & theme -----------------
st.set_page_config(page_title="Smart Data Analyst Toolkit", layout="wide")
sns.set_theme(style="whitegrid")

st.title("📊 Smart Data Analyst Toolkit — Selective Analysis")
st.markdown("Choose the analysis you want to run, view results, then click **Generate PDF** to export only that analysis.")

# ----------------- Helpers -----------------
@st.cache_data
def load_csv(file):
    return pd.read_csv(file)

def save_plot_to_bytes(fig, dpi=150):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
    buf.seek(0)
    plt.close(fig)
    return buf

def add_image_to_elements(elements, img_bytesio, width=420, height=315):
    img_reader = ImageReader(img_bytesio)
    elements.append(Image(img_reader, width=width, height=height))
    elements.append(Spacer(1, 12))

def df_to_table_data(df):
    # Convert a DataFrame to a list-of-lists with header
    header = df.columns.tolist()
    rows = df.fillna("").astype(str).values.tolist()
    return [header] + rows

def make_pdf(selected_analysis_name, content_blocks):
    """
    content_blocks: list of tuples ('paragraph'|'table'|'image', content)
    - paragraph: string
    - table: pandas.DataFrame
    - image: BytesIO
    """
    pdf_buf = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buf, pagesize=letter, rightMargin=30,leftMargin=30, topMargin=30,bottomMargin=18)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    elements.append(Paragraph("Smart Data Analyst Toolkit Report", styles['Title']))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(f"Analysis: {selected_analysis_name}", styles['Heading2']))
    elements.append(Spacer(1, 12))

    for block_type, content in content_blocks:
        if block_type == 'paragraph':
            elements.append(Paragraph(content, styles['Normal']))
            elements.append(Spacer(1, 8))
        elif block_type == 'table':
            table_data = df_to_table_data(content)
            tbl = Table(table_data, repeatRows=1)
            tbl_style = [
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#d3d3d3")),
                ('GRID', (0,0), (-1,-1), 0.25, colors.black),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ]
            tbl.setStyle(tbl_style)
            elements.append(tbl)
            elements.append(Spacer(1, 12))
        elif block_type == 'image':
            add_image_to_elements(elements, content)
        else:
            # ignore unknown
            pass

    doc.build(elements)
    pdf_buf.seek(0)
    return pdf_buf

# ----------------- File upload -----------------
uploaded_file = st.file_uploader("📂 Upload CSV File", type="csv")
if not uploaded_file:
    st.info("👆 Please upload a CSV file to begin.")
    st.stop()

try:
    df = load_csv(uploaded_file)
except Exception as e:
    st.error(f"❌ Error reading the file: {e}")
    st.stop()

st.success("✅ File uploaded")
st.subheader("🔍 Data preview")
st.dataframe(df.head())

numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

# Sidebar: choose analysis
st.sidebar.header("Choose Your Analysis")
analysis_type = st.sidebar.selectbox("Select Analysis Type", [
    "Exploratory Data Analysis",
    "Compare Means",
    "Association Between Categories",
    "Correlation",
    "Regression",
    "Check Normality",
    "Compare Variances"
])

# Prepare a list to collect content for PDF (only for the selected analysis)
pdf_blocks = []  # each: ('paragraph'|'table'|'image', content)

# ---------- Exploratory Data Analysis ----------
if analysis_type == "Exploratory Data Analysis":
    st.header("📈 Exploratory Data Analysis")
    st.markdown("Summary, missing values, distributions, and smart recommendations.")

    st.subheader("Summary statistics")
    desc = df.describe(include='all').transpose()
    st.dataframe(desc)
    pdf_blocks.append(('paragraph', "Summary statistics (transposed):"))
    pdf_blocks.append(('table', desc.round(4)))

    # Missing values
    missing = df.isnull().sum()
    total_missing = missing.sum()
    if total_missing > 0:
        msg = f"⚠️ Dataset has {int(total_missing)} missing values. Column-wise missing counts shown below."
        st.warning(msg)
        st.write(missing[missing > 0])
        pdf_blocks.append(('paragraph', msg))
        pdf_blocks.append(('table', missing[missing > 0].to_frame(name='missing_count')))
    else:
        msg = "✅ No missing values detected."
        st.info(msg)
        pdf_blocks.append(('paragraph', msg))

    # Visualizations in expanders
    with st.expander("Numeric Distributions"):
        if numeric_cols:
            for col in numeric_cols:
                fig, ax = plt.subplots()
                sns.histplot(df[col].dropna(), kde=True, ax=ax)
                ax.set_title(f"Distribution: {col}")
                st.pyplot(fig)
                imgbuf = save_plot_to_bytes(fig)
                pdf_blocks.append(('image', imgbuf))
        else:
            st.info("No numeric columns to plot.")

    with st.expander("Categorical Distributions"):
        if categorical_cols:
            for col in categorical_cols:
                fig, ax = plt.subplots(figsize=(6, max(2, min(6, df[col].nunique()/2))))
                order = df[col].value_counts().index
                sns.countplot(y=col, data=df, order=order, ax=ax)
                ax.set_title(f"Frequency: {col}")
                st.pyplot(fig)
                imgbuf = save_plot_to_bytes(fig)
                pdf_blocks.append(('image', imgbuf))
        else:
            st.info("No categorical columns to plot.")

    # Recommendations
    st.markdown("### 💡 Recommendations")
    if len(numeric_cols) >= 2:
        st.write("• Explore correlations between numeric variables (Correlation analysis).")
        pdf_blocks.append(('paragraph', "Recommendation: Explore correlations between numeric variables."))
    if len(categorical_cols) >= 2:
        st.write("• Consider Chi-square test for association between categorical variables.")
        pdf_blocks.append(('paragraph', "Recommendation: Consider Chi-square test for categorical associations."))

# ---------- Compare Means ----------
elif analysis_type == "Compare Means":
    st.header("🧪 Compare Means (t-test / ANOVA)")
    if not numeric_cols or not categorical_cols:
        st.error("Need at least one numeric and one categorical column.")
    else:
        target = st.selectbox("Choose numeric variable (target)", numeric_cols)
        group = st.selectbox("Choose categorical grouping variable", categorical_cols)

        if target and group:
            st.subheader("Boxplot by group")
            fig, ax = plt.subplots()
            sns.boxplot(x=group, y=target, data=df, palette="pastel", ax=ax)
            st.pyplot(fig)
            pdf_blocks.append(('image', save_plot_to_bytes(fig)))

            unique_groups = df[group].dropna().unique()
            st.write("Groups found:", list(unique_groups))
            if len(unique_groups) == 2:
                st.info("Running independent t-test (two groups).")
                g1, g2 = unique_groups
                a = df[df[group] == g1][target].dropna()
                b = df[df[group] == g2][target].dropna()
                t_stat, p_val = stats.ttest_ind(a, b, nan_policy='omit', equal_var=False)
                st.write(f"t-statistic = {t_stat:.3f}, p-value = {p_val:.4f}")
                pdf_blocks.append(('paragraph', f"Independent t-test: t = {t_stat:.3f}, p = {p_val:.4f}"))
            elif len(unique_groups) > 2:
                st.info("Running one-way ANOVA (multiple groups).")
                model = smf.ols(f'{target} ~ C({group})', data=df).fit()
                anova_table = sm.stats.anova_lm(model, typ=2)
                st.write(anova_table)
                pdf_blocks.append(('paragraph', "One-way ANOVA results:"))
                pdf_blocks.append(('table', anova_table))
            else:
                st.warning("Need at least 2 groups to compare means.")

# ---------- Association Between Categories (Chi-square) ----------
elif analysis_type == "Association Between Categories":
    st.header("🧩 Association Between Categories (Chi-square)")
    if len(categorical_cols) < 2:
        st.error("Need at least two categorical columns.")
    else:
        cat1 = st.selectbox("First categorical variable", categorical_cols)
        cat2 = st.selectbox("Second categorical variable", [c for c in categorical_cols if c != cat1])
        if cat1 and cat2:
            contingency = pd.crosstab(df[cat1], df[cat2])
            st.subheader("Contingency Table")
            st.write(contingency)
            pdf_blocks.append(('table', contingency))

            chi2, p, dof, ex = stats.chi2_contingency(contingency)
            st.write(f"Chi-square = {chi2:.3f}, p-value = {p:.4f}, dof = {dof}")
            if p < 0.05:
                st.success("Significant association detected (p < 0.05).")
            else:
                st.info("No significant association detected (p >= 0.05).")
            pdf_blocks.append(('paragraph', f"Chi-square = {chi2:.3f}, p = {p:.4f}, dof = {dof}"))

# ---------- Correlation ----------
elif analysis_type == "Correlation":
    st.header("📉 Correlation (Pearson)")
    if len(numeric_cols) < 2:
        st.error("Need at least two numeric columns for correlation.")
    else:
        x = st.selectbox("Variable 1", numeric_cols)
        y = st.selectbox("Variable 2", [c for c in numeric_cols if c != x])
        if x and y:
            # Drop NA pairs
            valid = df[[x, y]].dropna()
            if valid.shape[0] < 3:
                st.warning("Not enough paired observations to compute correlation.")
            else:
                r, p =
