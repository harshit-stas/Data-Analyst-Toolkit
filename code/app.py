import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

st.set_page_config(page_title="Smart Data Analyst Toolkit", layout="wide")

st.title("📊 Smart Data Analyst Toolkit")
st.markdown("Upload a CSV file and get instant insights, colorful charts, smart recommendations, and statistical analysis.")

uploaded_file = st.file_uploader("📂 Upload CSV File", type="csv")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading the file: {e}")
        st.stop()

    st.success("✅ File uploaded and read successfully.")
    st.subheader("🔍 Preview of Your Data")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

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

    # --- EDA & Business Intelligence ---
    if analysis_type == "Exploratory Data Analysis":
        st.subheader("📈 Exploratory Data Analysis")
        st.write("Summary statistics and smart suggestions")

        st.markdown("### 📌 Summary")
        st.write(df.describe(include='all').transpose())

        st.markdown("### 🧠 Smart Insights")
        if df.isnull().sum().sum() > 0:
            st.warning("⚠️ Your dataset has missing values. Consider cleaning or imputing them.")
        else:
            st.info("✅ No missing values detected.")

        st.markdown("### 🎨 Visualizations")

        for col in numeric_cols:
            fig, ax = plt.subplots()
            sns.histplot(df[col].dropna(), kde=True, color='skyblue', ax=ax)
            ax.set_title(f"Distribution of {col}")
            st.pyplot(fig)

        for col in categorical_cols:
            fig, ax = plt.subplots()
            sns.countplot(y=col, data=df, palette='Set3', order=df[col].value_counts().index, ax=ax)
            ax.set_title(f"Frequency of {col}")
            st.pyplot(fig)

        st.markdown("### 💡 Recommendations")
        if len(numeric_cols) >= 2:
            st.write("You can explore relationships between numeric variables using correlation or regression.")
        if len(categorical_cols) >= 2:
            st.write("Consider Chi-square test for association between categorical variables.")

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

    elif analysis_type == "Correlation":
        st.subheader("📉 Correlation")
        x = st.sidebar.selectbox("Variable 1", numeric_cols)
        y = st.sidebar.selectbox("Variable 2", [col for col in numeric_cols if col != x])

        if x and y:
            r, p = stats.pearsonr(df[x], df[y])
            st.write(f"**Pearson correlation coefficient**: {r:.3f}, **p-value**: {p:.3f}")
            fig, ax = plt.subplots()
            sns.scatterplot(x=x, y=y, data=df, hue=df[categorical_cols[0]] if categorical_cols else None, ax=ax)
            st.pyplot(fig)

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

    elif analysis_type == "Check Normality":
        st.subheader("📊 Normality Check (Shapiro-Wilk Test)")
        col = st.sidebar.selectbox("Numeric Column", numeric_cols)
        if col:
            stat, p = stats.shapiro(df[col].dropna())
            st.write(f"**W-statistic**: {stat:.3f}, **p-value**: {p:.3f}")
            fig, ax = plt.subplots()
            sns.histplot(df[col].dropna(), kde=True, ax=ax)
            st.pyplot(fig)

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
                fig, ax = plt.subplots()
                sns.boxplot(x=group, y=target, data=df, ax=ax)
                st.pyplot(fig)

else:
    st.info("👆 Please upload a CSV file to begin analysis.")
