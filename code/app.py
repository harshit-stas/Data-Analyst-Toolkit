# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, Image
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
import io
import datetime

# ---- Page config & theme ----
st.set_page_config(page_title="Advanced Data Analyst Toolkit", layout="wide")
sns.set_theme(style="whitegrid")

# ---- Helper functions ----
@st.cache_data
def load_data(file):
    # allow csv and excel streams - streamlit gives a BytesIO-like object
    name = getattr(file, "name", "")
    if name.endswith(".csv") or name.lower().endswith(".csv"):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

def fig_to_bytes(fig, dpi=150):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
    buf.seek(0)
    plt.close(fig)
    return buf

def df_to_table_data_with_index(df, max_rows=60):
    df2 = df.copy()
    if df2.shape[0] > max_rows:
        df2 = df2.head(max_rows)
    df2 = df2.reset_index()
    header = df2.columns.tolist()
    rows = df2.fillna("").astype(str).values.tolist()
    return [header] + rows

def safe_corr_with_p(df, cols):
    """
    Returns DataFrame of r and p-values for pairs in cols (symmetric).
    """
    n = len(cols)
    rmat = pd.DataFrame(index=cols, columns=cols, dtype=float)
    pmat = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if i <= j:
                valid = df[[a,b]].dropna()
                if valid.shape[0] < 3:
                    r, p = np.nan, np.nan
                else:
                    try:
                        r, p = stats.pearsonr(valid[a], valid[b])
                    except Exception:
                        r, p = np.nan, np.nan
                rmat.at[a,b] = r
                pmat.at[a,b] = p
                rmat.at[b,a] = r
                pmat.at[b,a] = p
    return rmat, pmat

def add_paragraph(blocks, text):
    blocks.append(('paragraph', text))

def add_table(blocks, df_table):
    blocks.append(('table', df_table))

def add_figure(blocks, fig, title=None):
    blocks.append(('image', fig_to_bytes(fig)))
    if title:
        blocks.append(('paragraph', title))

def build_pdf_report(title_text, author_text, analysis_title_list, blocks):
    """
    blocks: list of ('paragraph'|'table'|'image', content)
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, rightMargin=30,leftMargin=30, topMargin=30,bottomMargin=18)
    styles = getSampleStyleSheet()
    corporate_hdr = ParagraphStyle('CorpHeader', parent=styles['Title'], fontSize=18, alignment=1)
    small = styles['Normal']
    elements = []

    # Cover page
    elements.append(Paragraph(title_text, corporate_hdr))
    elements.append(Spacer(1,8))
    elements.append(Paragraph(f"Author: {author_text}", small))
    elements.append(Paragraph(f"Date: {datetime.date.today().isoformat()}", small))
    elements.append(Spacer(1,12))
    elements.append(Paragraph("Analyses included:", styles['Heading3']))
    elements.append(Paragraph(", ".join(analysis_title_list), small))
    elements.append(Spacer(1,18))

    # For each block, render appropriately
    for block_type, content in blocks:
        if block_type == 'paragraph':
            elements.append(Paragraph(content, small))
            elements.append(Spacer(1,6))
        elif block_type == 'table':
            table_data = df_to_table_data_with_index(content)
            tbl = Table(table_data, repeatRows=1)
            tbl_style = [
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2E4053")),
                ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                ('GRID', (0,0), (-1,-1), 0.25, colors.HexColor("#b0b0b0")),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('BACKGROUND', (0,1), (-1,-1), colors.whitesmoke)
            ]
            tbl.setStyle(tbl_style)
            elements.append(tbl)
            elements.append(Spacer(1,12))
        elif block_type == 'image':
            try:
                img = Image(ImageReader(content), width=420, height=315)
                elements.append(img)
                elements.append(Spacer(1,12))
            except Exception as e:
                elements.append(Paragraph(f"(Image embed failed: {e})", small))
                elements.append(Spacer(1,6))

    doc.build(elements)
    buf.seek(0)
    return buf

# ---- UI ----
st.title("Advanced Data Analyst Toolkit — Master Level")
st.markdown("Upload a dataset, pick analyses (multiple), run them, and generate a corporate-formatted PDF report. All tests include null/alternative hypotheses and p-values.")

uploaded = st.file_uploader("Upload CSV / Excel file", type=["csv","xlsx"])
if not uploaded:
    st.info("Please upload a CSV or Excel file to proceed.")
    st.stop()

try:
    df = load_data(uploaded)
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

st.subheader("Dataset preview")
st.dataframe(df.head())

numeric_cols = df.select_dtypes(include='number').columns.tolist()
categorical_cols = df.select_dtypes(include=['object','category','bool']).columns.tolist()

# analyses options
analysis_options = [
    "Summary Statistics & Missing Values",
    "Exploratory Visuals (Numeric & Categorical)",
    "Correlation Matrix (r & p)",
    "Simple Linear Regression",
    "Multiple Linear Regression",
    "Partial Regression (controls)",
    "Regression Diagnostics",
    "Check Normality (Shapiro-Wilk)",
    "Compare Means (t-test / ANOVA)",
    "Compare Variances (Levene)",
    "Association Between Categories (Chi-square)"
]
chosen = st.multiselect("Select analyses", analysis_options, default=["Summary Statistics & Missing Values","Correlation Matrix (r & p)"])
st.write("Configure analysis options below.")

# UI specifics
# Multiple regression
mlr_ui = {}
if "Multiple Linear Regression" in chosen or "Partial Regression (controls)" in chosen:
    st.subheader("Multiple / Partial Regression options")
    if len(numeric_cols) < 2:
        st.error("Need at least 2 numeric columns for regression.")
    else:
        dep_var = st.selectbox("Dependent variable (Y)", numeric_cols, key="dep_var")
        indep_vars = st.multiselect("Independent variables (X1, X2, ...)", [c for c in numeric_cols if c!=dep_var], default=[c for c in numeric_cols if c!=dep_var][:2])
        mlr_ui['y'] = dep_var
        mlr_ui['X'] = indep_vars

# Partial regression: choose focal predictor and controls
partial_ui = {}
if "Partial Regression (controls)" in chosen:
    st.subheader("Partial Regression options")
    if not mlr_ui.get('X'):
        st.info("Select multiple regression predictors first (above).")
    else:
        focal = st.selectbox("Focal predictor for partial effect", mlr_ui['X'], key="focal")
        controls = [v for v in mlr_ui['X'] if v != focal]
        partial_ui['focal'] = focal
        partial_ui['controls'] = controls

# Simple linear regression
slr_ui = {}
if "Simple Linear Regression" in chosen:
    st.subheader("Simple Regression options")
    if len(numeric_cols) >= 2:
        slr_x = st.selectbox("X (independent)", numeric_cols, key="slr_x")
        slr_y = st.selectbox("Y (dependent)", [c for c in numeric_cols if c!=slr_x], key="slr_y")
        slr_ui['x'] = slr_x; slr_ui['y'] = slr_y

# Correlation options
corr_ui = {}
if "Correlation Matrix (r & p)" in chosen:
    st.subheader("Correlation options")
    if len(numeric_cols) < 2:
        st.error("Need ≥2 numeric columns.")
    else:
        corr_subset = st.multiselect("Choose numeric columns for correlation (blank = all)", numeric_cols, default=numeric_cols[:min(10, len(numeric_cols))])
        corr_ui['cols'] = corr_subset or numeric_cols

# Normality options
normal_ui = {}
if "Check Normality (Shapiro-Wilk)" in chosen:
    st.subheader("Normality options")
    if numeric_cols:
        normal_cols = st.multiselect("Columns to test for normality (blank = all numeric)", numeric_cols, default=[])
        normal_ui['cols'] = normal_cols or numeric_cols

# Compare means options
means_ui = {}
if "Compare Means (t-test / ANOVA)" in chosen:
    st.subheader("Compare Means options")
    if numeric_cols and categorical_cols:
        means_target = st.selectbox("Numeric target", numeric_cols, key="means_target")
        means_group = st.selectbox("Grouping categorical variable", categorical_cols, key="means_group")
        means_ui['target'] = means_target; means_ui['group'] = means_group

# Levene options
levene_ui = {}
if "Compare Variances (Levene)" in chosen:
    st.subheader("Levene options")
    if numeric_cols and categorical_cols:
        lev_target = st.selectbox("Numeric variable", numeric_cols, key="lev_target")
        lev_group = st.selectbox("Group variable (2 or more groups allowed)", categorical_cols, key="lev_group")
        levene_ui['target'] = lev_target; levene_ui['group'] = lev_group

# Chi-square
chi_ui = {}
if "Association Between Categories (Chi-square)" in chosen:
    st.subheader("Chi-square options")
    if len(categorical_cols) >= 2:
        chi_a = st.selectbox("Categorical A", categorical_cols, key="chi_a")
        chi_b = st.selectbox("Categorical B", [c for c in categorical_cols if c!=chi_a], key="chi_b")
        chi_ui['a'] = chi_a; chi_ui['b'] = chi_b

# Exploratory visuals selection
explore_ui = {}
if "Exploratory Visuals (Numeric & Categorical)" in chosen:
    st.subheader("Exploratory visuals options")
    explore_num = st.multiselect("Numeric columns for histograms (blank=all)", numeric_cols, default=[])
    explore_cat = st.multiselect("Categorical columns for counts (blank=all)", categorical_cols, default=[])
    explore_ui['num'] = explore_num or numeric_cols
    explore_ui['cat'] = explore_cat or categorical_cols

# Prepare PDF blocks container
if 'pdf_blocks' not in st.session_state:
    st.session_state.pdf_blocks = []

# Run analyses
if st.button("▶ Run selected analyses"):
    st.session_state.pdf_blocks = []
    st.success("Running analyses...")
    # 1) Summary
    if "Summary Statistics & Missing Values" in chosen:
        st.header("Summary Statistics & Missing Values")
        desc = df.describe(include='all').transpose()
        st.dataframe(desc)
        add_paragraph(st.session_state.pdf_blocks, "<b>Summary Statistics (variables + descriptive statistics)</b>")
        add_table(st.session_state.pdf_blocks, desc)

        missing = df.isnull().sum()
        total_missing = int(missing.sum())
        add_paragraph(st.session_state.pdf_blocks, f"Missing values: total = {total_missing}")
        if total_missing > 0:
            add_table(st.session_state.pdf_blocks, missing[missing>0].to_frame('missing_count'))

    # 2) Exploratory visuals
    if "Exploratory Visuals (Numeric & Categorical)" in chosen:
        st.header("Exploratory Visuals")
        # Numeric histograms with annotation heuristics
        for col in explore_ui.get('num', []):
            data = df[col].dropna()
            if data.empty:
                st.write(f"{col}: No data")
                add_paragraph(st.session_state.pdf_blocks, f"{col}: No data")
                continue
            fig, ax = plt.subplots()
            counts, bins, patches = ax.hist(data, bins='auto', alpha=0.9)
            sns.kdeplot(data, ax=ax, fill=False)
            ax.set_title(f"Distribution — {col}")
            # annotate only if bins <=12 and spacing is sufficient
            annotate = len(patches) <= 12
            if annotate:
                # simplistic spacing test
                xs = [p.get_x()+p.get_width()/2 for p in patches]
                if len(xs) >= 2 and (min(np.diff(sorted(xs))) < (bins[-1]-bins[0]) * 0.02):
                    annotate = False
            if annotate:
                for rect, count in zip(patches, counts):
                    height = rect.get_height()
                    ax.annotate(int(count), xy=(rect.get_x()+rect.get_width()/2, height), xytext=(0,3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
            st.pyplot(fig)
            add_figure(st.session_state.pdf_blocks, fig, f"Histogram: {col}")

        # Categorical counts (annotate when categories small)
        for col in explore_ui.get('cat', []):
            vc = df[col].value_counts()
            labels = vc.index.astype(str).tolist()
            values = vc.values.tolist()
            fig, ax = plt.subplots(figsize=(6, max(2, min(8, len(labels)*0.5))))
            y_pos = np.arange(len(labels))
            ax.barh(y_pos, values, align='center')
            ax.set_yticks(y_pos); ax.set_yticklabels(labels)
            ax.set_title(f"Frequency — {col}")
            annotate = len(labels) <= 12
            if annotate:
                max_val = max(values) if values else 0
                for i,v in enumerate(values):
                    ax.text(v + max_val*0.01, i, str(v), va='center', fontsize=8)
            st.pyplot(fig)
            add_figure(st.session_state.pdf_blocks, fig, f"Counts: {col}")

    # 3) Correlation r & p
    if "Correlation Matrix (r & p)" in chosen:
        st.header("Correlation Matrix (r & p)")
        cols = corr_ui.get('cols', numeric_cols)
        if len(cols) < 2:
            st.error("Need ≥2 numeric columns")
            add_paragraph(st.session_state.pdf_blocks, "Correlation skipped: insufficient numeric variables.")
        else:
            rmat, pmat = safe_corr_with_p(df, cols)
            st.subheader("Pearson r")
            st.dataframe(rmat.round(4))
            st.subheader("p-values")
            st.dataframe(pmat.round(4))
            add_paragraph(st.session_state.pdf_blocks, "Correlation matrix (Pearson r)")
            add_table(st.session_state.pdf_blocks, rmat.round(4))
            add_paragraph(st.session_state.pdf_blocks, "Corresponding p-values")
            add_table(st.session_state.pdf_blocks, pmat.round(4))
            # heatmap of r only
            fig, ax = plt.subplots(figsize=(min(10,len(cols)*0.6), min(8,len(cols)*0.6)))
            sns.heatmap(rmat.astype(float), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            ax.set_title("Correlation (r)")
            st.pyplot(fig)
            add_figure(st.session_state.pdf_blocks, fig, "Correlation heatmap (r)")

    # 4) Simple linear regression
    if "Simple Linear Regression" in chosen and slr_ui:
        st.header("Simple Linear Regression")
        x = slr_ui['x']; y = slr_ui['y']
        add_paragraph(st.session_state.pdf_blocks, f"<b>Simple regression:</b> {y} ~ {x}")
        data = df[[x,y]].dropna()
        if data.shape[0] < 3:
            st.warning("Too few observations")
            add_paragraph(st.session_state.pdf_blocks, "Too few observations for regression.")
        else:
            model = smf.ols(f"{y} ~ {x}", data=data).fit()
            st.text(model.summary().as_text())
            add_paragraph(st.session_state.pdf_blocks, "Model summary:")
            # coefficients table
            coeffs = model.summary2().tables[1]
            add_table(st.session_state.pdf_blocks, coeffs)
            # Plot
            fig, ax = plt.subplots()
            sns.regplot(x=x, y=y, data=data, ax=ax)
            ax.set_title(f"{y} vs {x} with regression")
            st.pyplot(fig)
            add_figure(st.session_state.pdf_blocks, fig, f"Regression plot: {y} ~ {x}")

    # 5) Multiple linear regression
    if "Multiple Linear Regression" in chosen and mlr_ui.get('X'):
        st.header("Multiple Linear Regression")
        y = mlr_ui['y']; X_list = mlr_ui['X']
        add_paragraph(st.session_state.pdf_blocks, f"<b>Multiple regression:</b> {y} ~ {' + '.join(X_list)}")
        data = df[[y]+X_list].dropna()
        if data.shape[0] < (len(X_list)+3):
            st.warning("Too few observations for stable multiple regression.")
            add_paragraph(st.session_state.pdf_blocks, "Too few observations for stable multiple regression.")
        else:
            X = sm.add_constant(data[X_list])
            model = sm.OLS(data[y], X).fit()
            st.text(model.summary().as_text())
            add_paragraph(st.session_state.pdf_blocks, "Model summary (coefficients & tests):")
            coef_df = model.params.reset_index()
            coef_df.columns = ['term', 'coefficient']
            # extend with std err, t, p
            coef_stats = model.summary2().tables[1]
            add_table(st.session_state.pdf_blocks, coef_stats)
            # diagnostics plots: residuals vs fitted
            fig, ax = plt.subplots()
            fitted = model.fittedvalues; resid = model.resid
            ax.scatter(fitted, resid, alpha=0.6)
            ax.axhline(0, color='red', linestyle='--')
            ax.set_xlabel("Fitted values"); ax.set_ylabel("Residuals")
            ax.set_title("Residuals vs Fitted")
            st.pyplot(fig)
            add_figure(st.session_state.pdf_blocks, fig, "Residuals vs Fitted")

    # 6) Partial regression (effect of focal controlling for others)
    if "Partial Regression (controls)" in chosen and partial_ui:
        st.header("Partial Regression (partial effect / partial correlation)")
        focal = partial_ui['focal']; controls = partial_ui['controls']; y = mlr_ui['y']
        add_paragraph(st.session_state.pdf_blocks, f"<b>Partial regression:</b> effect of {focal} on {y} controlling for {', '.join(controls)}")
        # compute residuals: regress focal ~ controls ; regress y ~ controls ; correlate residuals
        sub = df[[y, focal] + controls].dropna()
        if sub.shape[0] < 4:
            st.warning("Too few observations for partial regression.")
            add_paragraph(st.session_state.pdf_blocks, "Too few observations for partial regression.")
        else:
            # regress focal on controls
            Xc = sm.add_constant(sub[controls])
            mod1 = sm.OLS(sub[focal], Xc).fit()
            res_focal = mod1.resid
            mod2 = sm.OLS(sub[y], Xc).fit()
            res_y = mod2.resid
            r, p = stats.pearsonr(res_focal, res_y)
            st.write(f"Partial correlation (controlled r) between {focal} and {y}: r={r:.4f}, p={p:.4f}")
            add_paragraph(st.session_state.pdf_blocks, f"Partial correlation (r) between {focal} and {y} controlling for {', '.join(controls)}: r = {r:.4f}, p = {p:.4f}")
            # partial regression plot (residuals scatter)
            fig, ax = plt.subplots()
            ax.scatter(res_focal, res_y, alpha=0.6)
            m, b = np.polyfit(res_focal, res_y, 1)
            xs = np.linspace(min(res_focal), max(res_focal), 50)
            ax.plot(xs, m*xs + b, color='red')
            ax.set_xlabel(f"Residuals of {focal} (wrt controls)")
            ax.set_ylabel(f"Residuals of {y} (wrt controls)")
            ax.set_title("Partial regression plot")
            st.pyplot(fig)
            add_figure(st.session_state.pdf_blocks, fig, "Partial regression residuals plot")

    # 7) Regression diagnostics (if multiple regression run)
    if "Regression Diagnostics" in chosen and mlr_ui.get('X'):
        st.header("Regression Diagnostics")
        # reuse multiple regression model if available
        data = df[[mlr_ui['y']] + mlr_ui['X']].dropna()
        if data.shape[0] < (len(mlr_ui['X'])+3):
            st.warning("Not enough data for diagnostics.")
            add_paragraph(st.session_state.pdf_blocks, "Not enough data for regression diagnostics.")
        else:
            X = sm.add_constant(data[mlr_ui['X']])
            model = sm.OLS(data[mlr_ui['y']], X).fit()
            # VIF
            try:
                from statsmodels.stats.outliers_influence import variance_inflation_factor
                vif_data = []
                for i in range(1, X.shape[1]):
                    vif = variance_inflation_factor(X.values, i)
                    vif_data.append({'variable': X.columns[i], 'VIF': round(vif,3)})
                vif_df = pd.DataFrame(vif_data)
                st.subheader("Variance Inflation Factors (VIF)")
                st.dataframe(vif_df)
                add_paragraph(st.session_state.pdf_blocks, "Variance Inflation Factors (VIF)")
                add_table(st.session_state.pdf_blocks, vif_df)
            except Exception as e:
                st.warning(f"Could not compute VIF: {e}")
                add_paragraph(st.session_state.pdf_blocks, f"VIF computation failed: {e}")

            # Normality of residuals
            resid = model.resid
            if len(resid) >= 3 and len(resid) <= 5000:
                w, p_sh = stats.shapiro(resid)
                st.write(f"Shapiro-Wilk on residuals: W={w:.4f}, p={p_sh:.4f}")
                add_paragraph(st.session_state.pdf_blocks, f"Shapiro-Wilk on residuals: W={w:.4f}, p={p_sh:.4f}")
            else:
                st.info("Residual normality test skipped (needs 3<=n<=5000).")
                add_paragraph(st.session_state.pdf_blocks, "Residual normality test skipped (sample size constraint).")

    # 8) Check normality
    if "Check Normality (Shapiro-Wilk)" in chosen:
        st.header("Normality Tests (Shapiro-Wilk)")
        for col in normal_ui.get('cols', []):
            data = df[col].dropna()
            if len(data) < 3:
                st.write(f"{col}: too few observations")
                add_paragraph(st.session_state.pdf_blocks, f"{col}: too few observations for Shapiro-Wilk.")
                continue
            w, p = stats.shapiro(data)
            st.write(f"{col}: W={w:.4f}, p={p:.4f}")
            add_paragraph(st.session_state.pdf_blocks, f"Shapiro-Wilk for {col}: W = {w:.4f}, p = {p:.4f}")
            # plot
            fig, ax = plt.subplots()
            ax.hist(data, bins='auto', alpha=0.9)
            sns.kdeplot(data, ax=ax, fill=False)
            ax.set_title(f"Distribution: {col}")
            st.pyplot(fig)
            add_figure(st.session_state.pdf_blocks, fig, f"Histogram: {col}")

    # 9) Compare means t-test / ANOVA
    if "Compare Means (t-test / ANOVA)" in chosen and means_ui:
        st.header("Compare Means")
        target = means_ui['target']; group = means_ui['group']
        add_paragraph(st.session_state.pdf_blocks, f"<b>Hypothesis:</b> H0: group means equal; H1: not all means equal.")
        ct = df[[target, group]].dropna()
        groups = ct[group].unique()
        st.write("Groups:", groups)
        add_paragraph(st.session_state.pdf_blocks, f"Groups: {', '.join(map(str,groups))}")
        if len(groups) == 2:
            a = ct[ct[group]==groups[0]][target]; b = ct[ct[group]==groups[1]][target]
            if len(a) < 2 or len(b) < 2:
                st.warning("Not enough data for t-test")
                add_paragraph(st.session_state.pdf_blocks, "Not enough data for t-test.")
            else:
                st.write("H0: µ1 = µ2 ; H1: µ1 != µ2")
                t_stat, p = stats.ttest_ind(a,b, nan_policy='omit', equal_var=False)
                st.write(f"t = {t_stat:.4f}, p = {p:.4f}")
                add_paragraph(st.session_state.pdf_blocks, f"Independent t-test: t = {t_stat:.4f}, p = {p:.4f}")
        elif len(groups) > 2:
            formula = f"{target} ~ C({group})"
            model = smf.ols(formula, data=ct).fit()
            anova_table = anova_lm(model, typ=2)
            st.dataframe(anova_table)
            add_paragraph(st.session_state.pdf_blocks, "ANOVA (one-way) — H0: all group means equal; H1: not all equal")
            add_table(st.session_state.pdf_blocks, anova_table)
            # Tukey post-hoc if significant
            if anova_table['PR(>F)'].iloc[0] < 0.05:
                try:
                    res = pairwise_tukeyhsd(ct[target], ct[group])
                    res_df = pd.DataFrame(data=res._results_table.data[1:], columns=res._results_table.data[0])
                    st.subheader("Tukey HSD post-hoc")
                    st.dataframe(res_df)
                    add_paragraph(st.session_state.pdf_blocks, "Tukey HSD post-hoc comparisons")
                    add_table(st.session_state.pdf_blocks, res_df)
                except Exception as e:
                    st.warning(f"Tukey HSD failed: {e}")
                    add_paragraph(st.session_state.pdf_blocks, f"Tukey HSD failed: {e}")

    # 10) Levene (compare variances)
    if "Compare Variances (Levene)" in chosen and levene_ui:
        st.header("Levene's Test for equal variances")
        target = levene_ui['target']; grp = levene_ui['group']
        add_paragraph(st.session_state.pdf_blocks, f"H0: group variances equal; H1: not equal")
        sub = df[[target, grp]].dropna()
        lev_groups = sub[grp].unique()
        arrays = [sub[sub[grp]==g][target].dropna() for g in lev_groups]
        if any(len(a)<2 for a in arrays):
            st.warning("Not enough observations in at least one group for Levene")
            add_paragraph(st.session_state.pdf_blocks, "Not enough observations for Levene.")
        else:
            stat, p = stats.levene(*arrays)
            st.write(f"Levene stat = {stat:.4f}, p = {p:.4f}")
            add_paragraph(st.session_state.pdf_blocks, f"Levene statistic = {stat:.4f}, p = {p:.4f}")

    # 11) Chi-square for categorical association
    if "Association Between Categories (Chi-square)" in chosen and chi_ui:
        st.header("Chi-square Test of Independence")
        a = chi_ui['a']; b = chi_ui['b']
        add_paragraph(st.session_state.pdf_blocks, f"H0: variables {a} and {b} are independent; H1: not independent")
        ct = pd.crosstab(df[a], df[b])
        st.subheader("Contingency table")
        st.dataframe(ct)
        add_table(st.session_state.pdf_blocks, ct)
        try:
            chi2, p, dof, ex = stats.chi2_contingency(ct)
            st.write(f"Chi2 = {chi2:.4f}, p = {p:.4f}, dof = {dof}")
            add_paragraph(st.session_state.pdf_blocks, f"Chi-square = {chi2:.4f}, p = {p:.4f}, dof = {dof}")
        except Exception as e:
            st.error(f"Chi-square failed: {e}")
            add_paragraph(st.session_state.pdf_blocks, f"Chi-square failed: {e}")

    st.success("Analyses finished — generate PDF below.")
    # end run button

# ---- Generate PDF button ----
st.markdown("---")
if st.button("📄 Generate corporate PDF report (analysis-wise)"):
    if not st.session_state.get('pdf_blocks'):
        st.warning("No results to put into PDF. Run analyses first.")
    else:
        title_text = "Data Analysis Report"
        author_text = st.text_input("Report author (will appear on cover)", value="Data Analyst")
        # build
        pdf_buf = build_pdf_report(title_text, author_text or "Data Analyst", chosen, st.session_state.pdf_blocks)
        st.download_button("📥 Download PDF", data=pdf_buf, file_name="corporate_analysis_report.pdf", mime="application/pdf")
