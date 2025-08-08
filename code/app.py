import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# ----------------- Streamlit Page Config -----------------
st.set_page_config(page_title="Smart Data Analyst Toolkit", layout="wide")
sns.set_theme(style="whitegrid")

st.title("📊 Smart Data Analyst Toolkit")
st.markdown("Upload a CSV file and get **instant insights**, **colorful charts**, and **smart statistical recommendations**.")

# ----------------- File Upload & Cache -----------------
@st.cache_data
def load_csv(file):
    return pd.read_csv(file)

uploaded_file = st.file_uploader("📂 Upload CSV File", type="csv")

if uploaded_file:
    try:
        df = load_csv(uploaded_file)
    except Exception as e:
        st.error(f"❌ Error reading the file: {e}")
        st.stop()

    st.success("✅ File uploaded successfully.")
    st.subheader("🔍 Preview of Your Data")
    st.dataframe(df.head())

    # Identify column types
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    # Handle no numeric or categorical data edge case
    if not numeric_cols:
        st.warning("⚠️ No numeric columns detected. Some analyses will be unavailable.")
    if not categorical_cols:
        st.warning("⚠️ No categorical columns detected. Some analyses will be unavailable.")

    # Sidebar
    st.sidebar.header("Choose Your Analysis")
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type",
        [
            "Exploratory Data Analysis",
            "Compare Means" if numeric_cols and categorical_cols else None,
            "Association Between Categories" if len(categorical_cols) >= 2 else None,
            "Correlation" if len(numeric_cols) >= 2 else None,
            "Regression" if len(numeric_cols) >= 2 else None,
            "Check Normality" if numeric_cols else None,
            "Compare Variances" if numeric_cols and categorical_cols else None
        ]
    )

    # ----------------- EDA -----------------
    if analysis_type == "Exploratory Data Analysis":
        st.subheader("📈 Exploratory Data Analysis")
        st.write("Summary statistics and smart suggestions.")

        st.markdown("### 📌 Summary")
        st.write(df.describe(include='all').transpose())

        st.markdown("### 🧠 Smart Insights")
        if df.isnull().sum().sum() > 0:
            st.warning("⚠️ Your dataset has missing values. Consider cleaning or imputing them.")
        else:
            st.info("✅ No missing values detected.")

        if len(df) < 50:
            st.info("ℹ️ Dataset is small — be careful with statistical assumptions.")

        st.markdown("### 🎨 Visualizations")
        with st.expander("Numeric Distributions"):
            for col in numeric_cols:
                fig, ax = plt.subplots()
                sns.histplot(df[col].dropna(), kde=True, palette="husl", ax=ax)
                ax.set_title(f"Distribution of {col}")
                st.pyplot(fig)

        with st.expander("Categorical Distributions"):
            for col in categorical_cols:
                fig, ax = plt.subplots()
                sns.countplot(y=col, data=df, palette="Set3", order=df[col].value_counts().index, ax=ax)
                ax.set_title(f"Frequency of {col}")
                st.pyplot(fig)

        st.markdown("### 💡 Recommendations")
        if len(numeric_cols) >= 2:
            st.write("🔹 Explore correlations between numeric variables.")
        if len(categorical_cols) >= 2:
            st.write("🔹 Use Chi-square test to check relationships between categories.")

    # ----------------- Compare Means -----------------
    elif analysis_type == "Compare Means":
        st.subheader("🧪 Compare Means")
        target = st.sidebar.selectbox("Choose numeric variable", numeric_cols)
        group = st.sidebar.selectbox("Group by", categorical_cols)

        if target and group:
            unique_groups = df[group].dropna().unique()
            if len(unique_groups) == 2:
                st.info("Recommended test: Independent t-test")
                group1, group2 = unique_groups
                t_stat, p_val = stats.ttest_ind(
                    df[df[group] == group1][target],
                    df[df[group] == group2][target],
                    nan_policy='omit'
                )
                st.write(f"**t-statistic**: {t_stat:.3f}, **p-value**: {p_val:.3f}")
            elif len(unique_groups) > 2:
                st.info("Recommended test: One-way ANOVA")
                model = smf.ols(f'{target} ~ C({group})', data=df).fit()
                anova_table = sm.stats.anova_lm(model, typ=2)
                st.write(anova_table)

            fig, ax = plt.subplots()
            sns.boxplot(x=group, y=target, data=df, palette="pastel", ax=ax)
            st.pyplot(fig)

    # ----------------- Association Between Categories -----------------
    elif analysis_type == "Association Between Categories":
        st.subheader("🧩 Chi-square Test")
        cat1 = st.sidebar.selectbox("First categorical variable", categorical_cols)
        cat2 = st.sidebar.selectbox("Second categorical variable", [c for c in categorical_cols if c != cat1])

        if cat1 and cat2:
            contingency = pd.crosstab(df[cat1], df[cat2])
            st.write("Contingency Table")
            st.write(contingency)
            chi2, p, dof, _ = stats.chi2_contingency(contingency)
            st.write(f"**Chi-square statistic**: {chi2:.3f}, **p-value**: {p:.3f}")
            if p < 0.05:
                st.success("✅ Significant association detected.")
            else:
                st.info("ℹ️ No significant association detected.")

    # ----------------- Correlation -----------------
    elif analysis_type == "Correlation":
        st.subheader("📉 Correlation")
        x = st.sidebar.selectbox("Variable 1", numeric_cols)
        y = st.sidebar.selectbox("Variable 2", [col for col in numeric_cols if col != x])

        if x and y:
            r, p = stats.pearsonr(df[x], df[y])
            st.write(f"**Pearson correlation coefficient**: {r:.3f}, **p-value**: {p:.3f}")
            if abs(r) > 0.7:
                st.success("Strong correlation detected — possible predictive relationship.")
            fig, ax = plt.subplots()
            sns.scatterplot(x=x, y=y, data=df, hue=df[categorical_cols[0]] if categorical_cols else None, palette="coolwarm", ax=ax)
            st.pyplot(fig)

    # ----------------- Regression -----------------
    elif analysis_type == "Regression":
        st.subheader("📈 Linear Regression")
        y = st.sidebar.selectbox("Dependent Variable", numeric_cols)
        x = st.sidebar.selectbox("Independent Variable", [col for col in numeric_cols if col != y])

        if x and y:
            model = smf.ols(f'{y} ~ {x}', data=df).fit()
            st.write(model.summary())
            fig, ax = plt.subplots()
            sns.regplot(x=x, y=y, data=df, color="tomato", ax=ax)
            st.pyplot(fig)

    # ----------------- Normality Check -----------------
    elif analysis_type == "Check Normality":
        st.subheader("📊 Normality Check (Shapiro-Wilk Test)")
        col = st.sidebar.selectbox("Numeric Column", numeric_cols)
        if col:
            stat, p = stats.shapiro(df[col].dropna())
            st.write(f"**W-statistic**: {stat:.3f}, **p-value**: {p:.3f}")
            if p < 0.05:
                st.warning("Data is not normally distributed — consider non-parametric tests.")
            else:
                st.success("Data is normally distributed.")
            fig, ax = plt.subplots()
            sns.histplot(df[col].dropna(), kde=True, palette="husl", ax=ax)
            st.pyplot(fig)

    # ----------------- Compare Variances -----------------
    elif analysis_type == "Compare Variances":
        st.subheader("📏 Compare Variances (Levene's Test)")
        target = st.sidebar.selectbox("Numeric Variable", numeric_cols)
        group = st.sidebar.selectbox("Group by", categorical_cols)

        if target and group:
            unique_groups = df[group].dropna().unique()
            if len(unique_groups) == 2:
                g1 = df[df[group] == unique_groups[0]][target].dropna()
                g2 = df[df[group] == unique_groups[1]][target].dropna()
                stat, p = stats.levene(g1, g2)
                st.write(f"**Levene's statistic**: {stat:.3f}, **p-value**: {p:.3f}")
                if p < 0.05:
                    st.warning("Variances are significantly different.")
                else:
                    st.success("Variances are similar.")
                fig, ax = plt.subplots()
                sns.boxplot(x=group, y=target, data=df, palette="pastel", ax=ax)
                st.pyplot(fig)

else:
    st.info("👆 Please upload a CSV file to begin analysis.")
