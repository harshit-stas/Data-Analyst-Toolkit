# app.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, Image
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors

# --------- Page config & theme ----------
st.set_page_config(page_title="Smart Data Analyst Toolkit", layout="wide")
sns.set_theme(style="whitegrid")

st.title("📊 Smart Data Analyst Toolkit — Multi-analysis + Single PDF")
st.write("Upload a CSV, choose analyses, run them, and export a single PDF containing only the results you ran.")

# --------- Helpers ----------
@st.cache_data
def load_csv(file):
    return pd.read_csv(file)

def fig_to_bytes(fig, dpi=150):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
    buf.seek(0)
    plt.close(fig)
    return buf

def df_to_table_data(df, max_rows=50):
    df2 = df.copy()
    if df2.shape[0] > max_rows:
        df2 = df2.head(max_rows)
    header = list(df2.columns)
    rows = df2.fillna("").astype(str).values.tolist()
    return [header] + rows

def build_pdf(analysis_title, blocks):
    """
    blocks: list of tuples (type, content)
      type: 'paragraph' | 'table' | 'image'
      content: str | pd.DataFrame | BytesIO
    returns BytesIO PDF
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, rightMargin=30,leftMargin=30, topMargin=30,bottomMargin=18)
    styles = getSampleStyleSheet()
    elems = []
    elems.append(Paragraph("Smart Data Analyst Toolkit Report", styles['Title']))
    elems.append(Spacer(1, 6))
    elems.append(Paragraph(f"Included analyses: {analysis_title}", styles['Heading2']))
    elems.append(Spacer(1, 12))

    for t, content in blocks:
        if t == 'paragraph':
            elems.append(Paragraph(content, styles['Normal']))
            elems.append(Spacer(1, 6))
        elif t == 'table':
            table_data = df_to_table_data(content)
            tbl = Table(table_data, repeatRows=1)
            tbl_style = [
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f0f0f0")),
                ('GRID', (0,0), (-1,-1), 0.25, colors.black),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ]
            tbl.setStyle(tbl_style)
            elems.append(tbl)
            elems.append(Spacer(1, 12))
        elif t == 'image':
            try:
                img = Image(ImageReader(content), width=420, height=315)
                elems.append(img)
                elems.append(Spacer(1, 12))
            except Exception:
                # fallback: tiny paragraph
                elems.append(Paragraph(" (image failed to embed)", styles['Italic']))
                elems.append(Spacer(1, 6))

    doc.build(elems)
    buf.seek(0)
    return buf

# --------- Upload ----------
uploaded = st.file_uploader("📂 Upload CSV file", type="csv")
if not uploaded:
    st.info("Upload a CSV to begin.")
    st.stop()

try:
    df = load_csv(uploaded)
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

st.success("✅ File loaded")
st.subheader("Data preview")
st.dataframe(df.head())

numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

# --------- Choose analyses ----------
analysis_options = [
    "Summary Statistics & Missing Values",
    "Exploratory Visuals (Numeric & Categorical)",
    "Correlation Matrix & Heatmap",
    "Simple Linear Regression",
    "Check Normality (Shapiro-Wilk)",
    "Compare Means (t-test / ANOVA)",
    "Association Between Categories (Chi-square)",
    "Compare Variances (Levene - 2 groups only)"
]
chosen = st.multiselect("Select analyses to include in this run/report", analysis_options, default=["Summary Statistics & Missing Values"])

# Placeholders for UI controls per analysis
st.markdown("---")
st.write("Configure analysis-specific options (appear below).")

# Store user selections for each analysis
# Regression
regression_ui = {}
if "Simple Linear Regression" in chosen:
    st.subheader("Regression options")
    if len(numeric_cols) < 2:
        st.error("Regression requires at least 2 numeric columns.")
    else:
        reg_y = st.selectbox("Dependent variable (Y)", numeric_cols, key="reg_y")
        reg_x = st.selectbox("Independent variable (X)", [c for c in numeric_cols if c != reg_y], key="reg_x")
        regression_ui['y'] = reg_y
        regression_ui['x'] = reg_x

# Correlation
corr_ui = {}
if "Correlation Matrix & Heatmap" in chosen:
    st.subheader("Correlation options")
    if len(numeric_cols) < 2:
        st.error("Need at least 2 numeric columns for correlation.")
    else:
        corr_subset = st.multiselect("Pick numeric columns for correlation (leave blank = all)", numeric_cols, default=numeric_cols[:min(10, len(numeric_cols))])
        corr_ui['cols'] = corr_subset or numeric_cols

# Normality
normality_ui = {}
if "Check Normality (Shapiro-Wilk)" in chosen:
    st.subheader("Normality options")
    if not numeric_cols:
        st.error("No numeric columns available.")
    else:
        normal_cols = st.multiselect("Pick numeric columns to test normality (if none selected, tests all numeric cols)", numeric_cols, default=[])
        normality_ui['cols'] = normal_cols or numeric_cols

# Compare Means
means_ui = {}
if "Compare Means (t-test / ANOVA)" in chosen:
    st.subheader("Compare Means options")
    if not numeric_cols or not categorical_cols:
        st.error("Compare Means requires numeric and categorical columns.")
    else:
        means_target = st.selectbox("Numeric variable (target)", numeric_cols, key="means_target")
        means_group = st.selectbox("Grouping categorical variable", categorical_cols, key="means_group")
        means_ui['target'] = means_target
        means_ui['group'] = means_group

# Chi-square
chi_ui = {}
if "Association Between Categories (Chi-square)" in chosen:
    st.subheader("Chi-square options")
    if len(categorical_cols) < 2:
        st.error("Need at least two categorical columns.")
    else:
        chi_a = st.selectbox("First categorical variable", categorical_cols, key="chi_a")
        chi_b = st.selectbox("Second categorical variable", [c for c in categorical_cols if c != chi_a], key="chi_b")
        chi_ui['a'] = chi_a
        chi_ui['b'] = chi_b

# Levene
levene_ui = {}
if "Compare Variances (Levene - 2 groups only)" in chosen:
    st.subheader("Levene options")
    if not numeric_cols or not categorical_cols:
        st.error("Levene requires numeric and categorical columns.")
    else:
        lev_target = st.selectbox("Numeric variable", numeric_cols, key="lev_target")
        lev_group = st.selectbox("Group by (must have exactly 2 groups)", categorical_cols, key="lev_group")
        levene_ui['target'] = lev_target
        levene_ui['group'] = lev_group

# Exploratory Visuals configuration
explore_ui = {}
if "Exploratory Visuals (Numeric & Categorical)" in chosen:
    st.subheader("Exploratory Visuals options")
    explore_num = st.multiselect("Numeric columns to plot (histograms). Leave blank = all numeric", numeric_cols, default=[])
    explore_cat = st.multiselect("Categorical columns to plot (countplots). Leave blank = all categorical", categorical_cols, default=[])
    explore_ui['num'] = explore_num or numeric_cols
    explore_ui['cat'] = explore_cat or categorical_cols

# Run analyses button
run_btn = st.button("▶️ Run Selected Analyses and Show Results")

# Prepare container to store blocks for PDF (persist across runs)
if 'pdf_blocks' not in st.session_state:
    st.session_state.pdf_blocks = []

if run_btn:
    st.session_state.pdf_blocks = []  # reset for fresh run
    st.success("Running analyses...")

    # --- Summary & missing values ---
    if "Summary Statistics & Missing Values" in chosen:
        st.header("📌 Summary Statistics & Missing Values")
        desc = df.describe(include='all').transpose()
        st.dataframe(desc)
        st.session_state.pdf_blocks.append(('paragraph', "Summary statistics (transposed):"))
        st.session_state.pdf_blocks.append(('table', desc))

        missing = df.isnull().sum()
        total_missing = int(missing.sum())
        if total_missing > 0:
            st.warning(f"⚠️ Dataset has {total_missing} missing values (column-wise shown).")
            st.write(missing[missing > 0])
            st.session_state.pdf_blocks.append(('paragraph', f"Dataset has {total_missing} missing values."))
            st.session_state.pdf_blocks.append(('table', missing[missing > 0].to_frame(name='missing_count')))
        else:
            st.info("✅ No missing values detected.")
            st.session_state.pdf_blocks.append(('paragraph', "No missing values detected."))

    # --- Exploratory Visuals ---
    if "Exploratory Visuals (Numeric & Categorical)" in chosen:
        st.header("📊 Exploratory Visuals")
        # Numeric histograms
        if explore_ui['num']:
            st.subheader("Numeric Distributions")
            for col in explore_ui['num']:
                if col not in df.columns:
                    continue
                fig, ax = plt.subplots()
                data = df[col].dropna()
                if len(data) == 0:
                    st.write(f"{col}: No data")
                    continue
                ax.hist(data, bins='auto', alpha=0.9)
                sns.kdeplot(data, ax=ax, fill=False)
                ax.set_title(f"Distribution — {col}")
                st.pyplot(fig)
                st.session_state.pdf_blocks.append(('image', fig_to_bytes(fig)))
        else:
            st.info("No numeric columns selected for plots.")

        # Categorical counts
        if explore_ui['cat']:
            st.subheader("Categorical Frequencies")
            for col in explore_ui['cat']:
                if col not in df.columns:
                    continue
                fig, ax = plt.subplots(figsize=(6, max(2, min(6, df[col].nunique()/2))))
                order = df[col].value_counts().index
                ax.barh(range(len(order)), df[col].value_counts().values, align='center')
                ax.set_yticks(range(len(order)))
                ax.set_yticklabels([str(x) for x in order])
                ax.set_title(f"Frequency — {col}")
                st.pyplot(fig)
                st.session_state.pdf_blocks.append(('image', fig_to_bytes(fig)))
        else:
            st.info("No categorical columns selected for plots.")

    # --- Correlation ---
    if "Correlation Matrix & Heatmap" in chosen:
        st.header("🔗 Correlation Matrix")
        cols = corr_ui.get('cols', numeric_cols)
        if len(cols) < 2:
            st.error("Need at least 2 numeric columns for correlation.")
        else:
            corr = df[cols].corr()
            st.dataframe(corr.round(3))
            st.session_state.pdf_blocks.append(('paragraph', "Correlation matrix (Pearson):"))
            st.session_state.pdf_blocks.append(('table', corr.round(4)))

            fig, ax = plt.subplots(figsize=(min(10, len(cols)*0.6), min(8, len(cols)*0.6)))
            sns.heatmap(corr, annot=True, fmt=".2f", ax=ax, cmap="coolwarm")
            ax.set_title("Correlation heatmap")
            st.pyplot(fig)
            st.session_state.pdf_blocks.append(('image', fig_to_bytes(fig)))

    # --- Regression ---
    if "Simple Linear Regression" in chosen and regression_ui:
        st.header("📈 Simple Linear Regression")
        y = regression_ui['y']
        x = regression_ui['x']
        if y not in df.columns or x not in df.columns:
            st.error("Regression columns missing.")
        else:
            data = df[[x, y]].dropna()
            if data.shape[0] < 3:
                st.warning("Too few observations for a meaningful regression.")
            else:
                model = smf.ols(f"{y} ~ {x}", data=data).fit()
                st.subheader("Model summary")
                st.text(model.summary().as_text())
                st.session_state.pdf_blocks.append(('paragraph', f"Regression: {y} ~ {x}"))
                # coefficients table
                coeffs = model.params.reset_index()
                coeffs.columns = ['term', 'coefficient']
                st.table(coeffs)
                st.session_state.pdf_blocks.append(('table', coeffs))

                fig, ax = plt.subplots()
                sns.regplot(x=x, y=y, data=data, ax=ax)
                ax.set_title(f"{y} vs {x} (with regression line)")
                st.pyplot(fig)
                st.session_state.pdf_blocks.append(('image', fig_to_bytes(fig)))

    # --- Normality ---
    if "Check Normality (Shapiro-Wilk)" in chosen:
        st.header("🔎 Normality (Shapiro-Wilk)")
        cols = normality_ui.get('cols', numeric_cols)
        for col in cols:
            data = df[col].dropna()
            if len(data) < 3:
                st.write(f"{col}: Too few observations for Shapiro-Wilk (need >= 3).")
                st.session_state.pdf_blocks.append(('paragraph', f"{col}: Too few observations for Shapiro-Wilk (need >= 3)."))
                continue
            stat, p = stats.shapiro(data)
            st.write(f"{col} — W = {stat:.4f}, p = {p:.4f}")
            st.session_state.pdf_blocks.append(('paragraph', f"{col} — Shapiro-Wilk W = {stat:.4f}, p = {p:.4f}"))
            # show histogram
            fig, ax = plt.subplots()
            ax.hist(data, bins='auto', alpha=0.9)
            sns.kdeplot(data, ax=ax, fill=False)
            ax.set_title(f"Distribution — {col}")
            st.pyplot(fig)
            st.session_state.pdf_blocks.append(('image', fig_to_bytes(fig)))

    # --- Compare Means (t-test / ANOVA) ---
    if "Compare Means (t-test / ANOVA)" in chosen and means_ui:
        st.header("🧪 Compare Means")
        target = means_ui['target']
        group = means_ui['group']
        if target not in df.columns or group not in df.columns:
            st.error("Selected columns missing.")
        else:
            st.write("Groups found:", df[group].dropna().unique().tolist())
            groups = df[group].dropna().unique()
            fig, ax = plt.subplots()
            sns.boxplot(x=group, y=target, data=df, ax=ax)
            ax.set_title(f"{target} by {group}")
            st.pyplot(fig)
            st.session_state.pdf_blocks.append(('image', fig_to_bytes(fig)))

            if len(groups) == 2:
                g1, g2 = groups
                a = df[df[group] == g1][target].dropna()
                b = df[df[group] == g2][target].dropna()
                if a.size < 2 or b.size < 2:
                    st.warning("Not enough observations in one or both groups.")
                    st.session_state.pdf_blocks.append(('paragraph', "Not enough observations for t-test."))
                else:
                    t_stat, p_val = stats.ttest_ind(a, b, nan_policy='omit', equal_var=False)
                    st.write(f"Independent t-test: t = {t_stat:.4f}, p = {p_val:.4f}")
                    st.session_state.pdf_blocks.append(('paragraph', f"Independent t-test: t = {t_stat:.4f}, p = {p_val:.4f}"))
            elif len(groups) > 2:
                model = smf.ols(f'{target} ~ C({group})', data=df).fit()
                anova_table = sm.stats.anova_lm(model, typ=2)
                st.write(anova_table)
                st.session_state.pdf_blocks.append(('paragraph', "One-way ANOVA results:"))
                st.session_state.pdf_blocks.append(('table', anova_table))
            else:
                st.warning("Need >=2 groups to compare means.")
                st.session_state.pdf_blocks.append(('paragraph', "Need >=2 groups to compare means."))

    # --- Chi-square ---
    if "Association Between Categories (Chi-square)" in chosen and chi_ui:
        st.header("🧩 Chi-square Test")
        a = chi_ui['a']
        b = chi_ui['b']
        if a not in df.columns or b not in df.columns:
            st.error("Selected columns missing.")
        else:
            contingency = pd.crosstab(df[a], df[b])
            st.subheader("Contingency table")
            st.write(contingency)
            st.session_state.pdf_blocks.append(('table', contingency))
            try:
                chi2, p, dof, ex = stats.chi2_contingency(contingency)
                st.write(f"Chi-square = {chi2:.4f}, p = {p:.4f}, dof = {dof}")
                st.session_state.pdf_blocks.append(('paragraph', f"Chi-square = {chi2:.4f}, p = {p:.4f}, dof = {dof}"))
            except Exception as e:
                st.error(f"Chi-square failed: {e}")
                st.session_state.pdf_blocks.append(('paragraph', f"Chi-square failed: {e}"))

    # --- Levene (2 groups only) ---
    if "Compare Variances (Levene - 2 groups only)" in chosen and levene_ui:
        st.header("📏 Levene's Test (2 groups)")
        target = levene_ui['target']
        grp = levene_ui['group']
        if target not in df.columns or grp not in df.columns:
            st.error("Selected columns missing.")
        else:
            groups = df[grp].dropna().unique()
            if len(groups) != 2:
                st.warning("Levene implementation requires exactly 2 groups.")
                st.session_state.pdf_blocks.append(('paragraph', "Levene requires exactly 2 groups."))
            else:
                g1, g2 = groups
                a = df[df[grp] == g1][target].dropna()
                b = df[df[grp] == g2][target].dropna()
                if a.size < 2 or b.size < 2:
                    st.warning("Not enough observations in one or both groups.")
                    st.session_state.pdf_blocks.append(('paragraph', "Not enough observations for Levene test."))
                else:
                    stat, p = stats.levene(a, b)
                    st.write(f"Levene stat = {stat:.4f}, p = {p:.4f}")
                    st.session_state.pdf_blocks.append(('paragraph', f"Levene stat = {stat:.4f}, p = {p:.4f}"))
                    fig, ax = plt.subplots()
                    sns.boxplot(x=grp, y=target, data=df, ax=ax)
                    ax.set_title(f"Boxplot {target} by {grp}")
                    st.pyplot(fig)
                    st.session_state.pdf_blocks.append(('image', fig_to_bytes(fig)))

    st.success("Finished running selected analyses. Click Generate PDF to export results.")

# --------- Generate PDF ----------
st.markdown("---")
col1, col2 = st.columns([1, 3])
with col1:
    if st.button("📄 Generate PDF (single file with all run analyses)"):
        if not st.session_state.get('pdf_blocks'):
            st.warning("No analysis output found. Please run selected analyses first.")
        else:
            analysis_title = ", ".join(chosen)
            pdf = build_pdf(analysis_title, st.session_state.pdf_blocks)
            st.download_button("📥 Download PDF", data=pdf, file_name="analysis_report.pdf", mime="application/pdf")
with col2:
    st.write("Tip: Use 'Run Selected Analyses' to produce outputs, then 'Generate PDF' to export those outputs in one file.")

# end
