import streamlit as st
import pandas as pd
import json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from groq import Groq
import re
import io
from matplotlib.backends.backend_pdf import PdfPages
from dotenv import load_dotenv
import os
load_dotenv()

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="AI-Assisted Data Wrangler & Visualizer",
    layout="wide"
)

# -----------------------------
# SESSION STATE INITIALIZATION
# -----------------------------
def init_session_state():
    defaults = {
        "original_df": None,
        "working_df": None,
        "uploaded_filename": None,
        "transformation_log": [],
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# -----------------------------
# RESET FUNCTION
# -----------------------------
def reset_session():
    st.session_state["original_df"] = None
    st.session_state["working_df"] = None
    st.session_state["uploaded_filename"] = None
    st.session_state["transformation_log"] = []


# -----------------------------
# TRANSFORMATION LOG
# -----------------------------
def add_log_step(operation_name, parameters=None, affected_columns=None):
    if parameters is None:
        parameters = {}
    if affected_columns is None:
        affected_columns = []

    st.session_state["transformation_log"].append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "operation": operation_name,
        "parameters": parameters,
        "affected_columns": affected_columns,
    })


# -----------------------------
# FILE LOADING
# -----------------------------
@st.cache_data
def load_file(file_name, file_bytes):
    """
    Load CSV, Excel, or JSON from uploaded file content.
    file_bytes is raw bytes from uploader.
    """

    if file_name.endswith(".csv"):
        return pd.read_csv(file_bytes)

    elif file_name.endswith(".xlsx"):
        return pd.read_excel(file_bytes)

    elif file_name.endswith(".json"):
        # Try normal JSON loading first
        try:
            return pd.read_json(file_bytes)
        except ValueError:
            # Fallback if json is not in a direct tabular format
            data = json.load(file_bytes)
            return pd.json_normalize(data)

    else:
        raise ValueError("Unsupported file format. Please upload CSV, XLSX, or JSON.")


# -----------------------------
# DATA SUMMARY HELPERS
# -----------------------------
def get_missing_summary(df):
    missing_count = df.isnull().sum()
    missing_percent = (df.isnull().sum() / len(df) * 100).round(2)

    summary = pd.DataFrame({
        "column": df.columns,
        "missing_count": missing_count.values,
        "missing_percent": missing_percent.values
    })
    return summary.sort_values(by="missing_count", ascending=False)


def get_dtype_summary(df):
    return pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str).values
    })


# -----------------------------
# APP START
# -----------------------------
init_session_state()

st.title("AI-Assisted Data Wrangler & Visualizer")

# Sidebar navigation
page = st.sidebar.radio(
    "Navigation",
    [
        "Upload & Overview",
        "Cleaning & Preparation",
        "Visualization Builder",
        "Export & Report"
    ]
)

st.sidebar.markdown("---")

if st.sidebar.button("Reset session"):
    reset_session()
    st.success("Session reset successfully.")



#Helper functions


def drop_rows_with_missing(df, selected_columns):
    before_shape = df.shape
    new_df = df.dropna(subset=selected_columns).copy()
    after_shape = new_df.shape

    impact = {
        "rows_before": before_shape[0],
        "rows_after": after_shape[0],
        "rows_removed": before_shape[0] - after_shape[0],
        "columns_before": before_shape[1],
        "columns_after": after_shape[1],
    }
    return new_df, impact


def drop_columns_above_threshold(df, threshold_percent):
    before_shape = df.shape
    missing_percent = (df.isnull().sum() / len(df) * 100)

    cols_to_drop = missing_percent[missing_percent > threshold_percent].index.tolist()
    new_df = df.drop(columns=cols_to_drop).copy()
    after_shape = new_df.shape

    impact = {
        "rows_before": before_shape[0],
        "rows_after": after_shape[0],
        "rows_removed": before_shape[0] - after_shape[0],
        "columns_before": before_shape[1],
        "columns_after": after_shape[1],
        "columns_removed": cols_to_drop,
    }
    return new_df, impact


def fill_missing_values(df, column, strategy, constant_value=None):
    new_df = df.copy()
    before_missing = int(new_df[column].isnull().sum())

    if before_missing == 0:
        return new_df, {
            "column": column,
            "strategy": strategy,
            "missing_before": 0,
            "missing_after": 0,
            "filled_count": 0
        }

    if strategy == "constant":
        new_df[column] = new_df[column].fillna(constant_value)

    elif strategy == "mean":
        if pd.api.types.is_numeric_dtype(new_df[column]):
            new_df[column] = new_df[column].fillna(new_df[column].mean())
        else:
            raise ValueError("Mean can only be applied to numeric columns.")

    elif strategy == "median":
        if pd.api.types.is_numeric_dtype(new_df[column]):
            new_df[column] = new_df[column].fillna(new_df[column].median())
        else:
            raise ValueError("Median can only be applied to numeric columns.")

    elif strategy == "mode":
        mode_series = new_df[column].mode(dropna=True)
        if not mode_series.empty:
            new_df[column] = new_df[column].fillna(mode_series.iloc[0])
        else:
            raise ValueError("Mode could not be computed for this column.")

    elif strategy == "most_frequent":
        mode_series = new_df[column].mode(dropna=True)
        if not mode_series.empty:
            new_df[column] = new_df[column].fillna(mode_series.iloc[0])
        else:
            raise ValueError("Most frequent value could not be computed.")

    elif strategy == "forward_fill":
        new_df[column] = new_df[column].ffill()

    elif strategy == "backward_fill":
        new_df[column] = new_df[column].bfill()

    else:
        raise ValueError("Unsupported fill strategy.")

    after_missing = int(new_df[column].isnull().sum())

    impact = {
        "column": column,
        "strategy": strategy,
        "missing_before": before_missing,
        "missing_after": after_missing,
        "filled_count": before_missing - after_missing
    }
    return new_df, impact

def convert_to_numeric_with_report(df, column):
    new_df = df.copy()

    original_series = new_df[column]
    non_null_before = original_series.notna().sum()
    missing_before = original_series.isna().sum()

    cleaned = (
        original_series.astype(str)
        .str.strip()
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("€", "", regex=False)
        .str.replace("£", "", regex=False)
        .str.replace("%", "", regex=False)
    )

    cleaned = cleaned.replace(["", "nan", "NaN", "None", "null", "NULL"], pd.NA)

    converted = pd.to_numeric(cleaned, errors="coerce")

    # values that were non-null before but became null after conversion
    failed_mask = original_series.notna() & converted.isna()
    failed_examples = original_series[failed_mask].astype(str).drop_duplicates().head(10).tolist()

    new_df[column] = converted

    missing_after = new_df[column].isna().sum()
    parsed_successfully = non_null_before - failed_mask.sum()

    report = {
        "column": column,
        "before_dtype": str(original_series.dtype),
        "after_dtype": str(new_df[column].dtype),
        "non_null_before": int(non_null_before),
        "missing_before": int(missing_before),
        "missing_after": int(missing_after),
        "failed_count": int(failed_mask.sum()),
        "parsed_successfully": int(parsed_successfully),
        "failed_examples": failed_examples,
    }

    return new_df, report


def convert_to_datetime_with_report(df, column, datetime_format=None, dayfirst=False):
    new_df = df.copy()

    original_series = new_df[column]
    non_null_before = original_series.notna().sum()
    missing_before = original_series.isna().sum()

    cleaned = original_series.astype(str).str.strip()
    cleaned = cleaned.replace(["", "nan", "NaN", "None", "null", "NULL"], pd.NA)

    converted = pd.to_datetime(
        cleaned,
        format=datetime_format if datetime_format else "mixed",
        errors="coerce",
        dayfirst=dayfirst
    )

    failed_mask = original_series.notna() & converted.isna()
    failed_examples = original_series[failed_mask].astype(str).drop_duplicates().head(10).tolist()

    new_df[column] = converted

    missing_after = new_df[column].isna().sum()
    parsed_successfully = non_null_before - failed_mask.sum()

    report = {
        "column": column,
        "before_dtype": str(original_series.dtype),
        "after_dtype": str(new_df[column].dtype),
        "non_null_before": int(non_null_before),
        "missing_before": int(missing_before),
        "missing_after": int(missing_after),
        "failed_count": int(failed_mask.sum()),
        "parsed_successfully": int(parsed_successfully),
        "failed_examples": failed_examples,
    }

    return new_df, report

def convert_to_category_with_report(df, column):
    new_df = df.copy()
    before_dtype = str(new_df[column].dtype)
    unique_before = new_df[column].nunique(dropna=True)

    new_df[column] = new_df[column].astype("category")

    report = {
        "column": column,
        "before_dtype": before_dtype,
        "after_dtype": str(new_df[column].dtype),
        "unique_before": int(unique_before),
        "unique_after": int(new_df[column].nunique(dropna=True)),
    }
    return new_df, report


def convert_to_string_with_report(df, column):
    new_df = df.copy()
    before_dtype = str(new_df[column].dtype)

    new_df[column] = new_df[column].astype("string")

    report = {
        "column": column,
        "before_dtype": before_dtype,
        "after_dtype": str(new_df[column].dtype),
    }
    return new_df, report


def get_duplicate_summary(df, subset_columns=None):
    if subset_columns:
        duplicate_mask = df.duplicated(subset=subset_columns, keep=False)
    else:
        duplicate_mask = df.duplicated(keep=False)

    duplicate_df = df[duplicate_mask].copy()

    return {
        "duplicate_count": int(duplicate_mask.sum()),
        "duplicate_rows": duplicate_df
    }


def remove_duplicates(df, subset_columns=None, keep_option="first"):
    before_shape = df.shape

    new_df = df.drop_duplicates(subset=subset_columns, keep=keep_option).copy()

    after_shape = new_df.shape

    impact = {
        "rows_before": before_shape[0],
        "rows_after": after_shape[0],
        "rows_removed": before_shape[0] - after_shape[0],
        "subset_columns": subset_columns if subset_columns else "full row",
        "keep_option": keep_option
    }

    return new_df, impact


def remove_duplicates_keep_by_order(df, subset_columns, order_column, keep_highest=True):
    """
    Keeps one row per duplicate group based on ordering column.
    Example:
    - keep_highest=True keeps the row with max(order_column)
    - keep_highest=False keeps the row with min(order_column)
    """
    before_shape = df.shape

    ascending = not keep_highest
    sorted_df = df.sort_values(by=order_column, ascending=ascending).copy()
    new_df = sorted_df.drop_duplicates(subset=subset_columns, keep="first").copy()

    after_shape = new_df.shape

    impact = {
        "rows_before": before_shape[0],
        "rows_after": after_shape[0],
        "rows_removed": before_shape[0] - after_shape[0],
        "subset_columns": subset_columns,
        "order_column": order_column,
        "kept": "highest" if keep_highest else "lowest"
    }

    return new_df, impact

def clean_text_column(
    df,
    column,
    trim_whitespace=False,
    remove_all_spaces=False,
    remove_quotes=False,
    case_option="no_change",
    find_text="",
    replace_text=""
):
    new_df = df.copy()
    original_series = new_df[column].copy()

    # Keep nulls as nulls
    mask_not_null = new_df[column].notna()
    cleaned = new_df.loc[mask_not_null, column].astype(str)

    if trim_whitespace:
        cleaned = cleaned.str.strip()

    if remove_all_spaces:
        cleaned = cleaned.str.replace(" ", "", regex=False)

    if remove_quotes:
        cleaned = cleaned.str.replace('"', "", regex=False)
        cleaned = cleaned.str.replace("'", "", regex=False)

    if case_option == "lower":
        cleaned = cleaned.str.lower()
    elif case_option == "upper":
        cleaned = cleaned.str.upper()
    elif case_option == "title":
        cleaned = cleaned.str.title()

    if find_text != "":
        cleaned = cleaned.str.replace(find_text, replace_text, regex=False)

    new_df.loc[mask_not_null, column] = cleaned

    changed_mask = (
        original_series.astype("string") != new_df[column].astype("string")
    ) & ~(original_series.isna() & new_df[column].isna())

    changed_count = int(changed_mask.sum())

    report = {
        "column": column,
        "before_dtype": str(original_series.dtype),
        "after_dtype": str(new_df[column].dtype),
        "changed_count": changed_count,
        "trim_whitespace": trim_whitespace,
        "remove_all_spaces": remove_all_spaces,
        "remove_quotes": remove_quotes,
        "case_option": case_option,
        "find_text": find_text,
        "replace_text": replace_text,
    }

    return new_df, report

def group_rare_categories(df, column, min_frequency=5, other_label="Other"):
    new_df = df.copy()
    original_series = new_df[column].copy()

    value_counts = original_series.value_counts(dropna=False)

    rare_categories = value_counts[value_counts < min_frequency].index.tolist()

    mask_rare = new_df[column].isin(rare_categories)
    replaced_count = int(mask_rare.sum())

    new_df.loc[mask_rare, column] = other_label

    report = {
        "column": column,
        "min_frequency": int(min_frequency),
        "other_label": other_label,
        "rare_categories": [str(x) for x in rare_categories[:20]],
        "rare_category_count": int(len(rare_categories)),
        "replaced_count": int(replaced_count)
    }

    return new_df, report


def one_hot_encode_columns(df, columns, drop_first=False, dummy_na=False):
    new_df = df.copy()
    before_columns = list(new_df.columns)

    new_df = pd.get_dummies(
        new_df,
        columns=columns,
        drop_first=drop_first,
        dummy_na=dummy_na
    )

    after_columns = list(new_df.columns)
    created_columns = [col for col in after_columns if col not in before_columns]

    report = {
        "encoded_columns": columns,
        "drop_first": drop_first,
        "dummy_na": dummy_na,
        "created_column_count": int(len(created_columns)),
        "created_columns_preview": created_columns[:30]
    }

    return new_df, report


def detect_outliers_iqr(df, column):
    series = df[column].dropna()

    if series.empty:
        return {
            "column": column,
            "q1": None,
            "q3": None,
            "iqr": None,
            "lower_bound": None,
            "upper_bound": None,
            "outlier_count": 0,
            "outlier_mask": pd.Series([False] * len(df), index=df.index)
        }

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)

    return {
        "column": column,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "outlier_count": int(outlier_mask.sum()),
        "outlier_mask": outlier_mask
    }

def winsorize_column(df, column, lower_quantile=0.05, upper_quantile=0.95):
    new_df = df.copy()

    lower_cap = new_df[column].quantile(lower_quantile)
    upper_cap = new_df[column].quantile(upper_quantile)

    before_series = new_df[column].copy()
    new_df[column] = new_df[column].clip(lower=lower_cap, upper=upper_cap)

    changed_count = int((before_series != new_df[column]).sum())

    report = {
        "column": column,
        "lower_quantile": lower_quantile,
        "upper_quantile": upper_quantile,
        "lower_cap": lower_cap,
        "upper_cap": upper_cap,
        "changed_count": changed_count
    }

    return new_df, report


def remove_outlier_rows_iqr(df, column):
    info = detect_outliers_iqr(df, column)
    outlier_mask = info["outlier_mask"]

    before_shape = df.shape
    new_df = df.loc[~outlier_mask].copy()
    after_shape = new_df.shape

    report = {
        "column": column,
        "rows_before": before_shape[0],
        "rows_after": after_shape[0],
        "rows_removed": before_shape[0] - after_shape[0],
        "lower_bound": info["lower_bound"],
        "upper_bound": info["upper_bound"]
    }

    return new_df, report


def min_max_scale_columns(df, columns):
    new_df = df.copy()
    report_rows = []

    for col in columns:
        col_min = new_df[col].min()
        col_max = new_df[col].max()

        before_mean = new_df[col].mean()
        before_std = new_df[col].std()

        if pd.isna(col_min) or pd.isna(col_max) or col_min == col_max:
            # keep unchanged if constant or invalid
            after_mean = new_df[col].mean()
            after_std = new_df[col].std()
        else:
            new_df[col] = (new_df[col] - col_min) / (col_max - col_min)
            after_mean = new_df[col].mean()
            after_std = new_df[col].std()

        report_rows.append({
            "column": col,
            "before_min": col_min,
            "before_max": col_max,
            "before_mean": before_mean,
            "before_std": before_std,
            "after_min": new_df[col].min(),
            "after_max": new_df[col].max(),
            "after_mean": after_mean,
            "after_std": after_std
        })

    return new_df, pd.DataFrame(report_rows)




def zscore_scale_columns(df, columns):
    new_df = df.copy()
    report_rows = []

    for col in columns:
        col_mean = new_df[col].mean()
        col_std = new_df[col].std()

        before_min = new_df[col].min()
        before_max = new_df[col].max()

        if pd.isna(col_std) or col_std == 0:
            after_mean = new_df[col].mean()
            after_std = new_df[col].std()
        else:
            new_df[col] = (new_df[col] - col_mean) / col_std
            after_mean = new_df[col].mean()
            after_std = new_df[col].std()

        report_rows.append({
            "column": col,
            "before_min": before_min,
            "before_max": before_max,
            "before_mean": col_mean,
            "before_std": col_std,
            "after_min": new_df[col].min(),
            "after_max": new_df[col].max(),
            "after_mean": after_mean,
            "after_std": after_std
        })

    return new_df, pd.DataFrame(report_rows)


def rename_column(df, old_name, new_name):
    new_df = df.copy()
    new_df = new_df.rename(columns={old_name: new_name})
    return new_df

def drop_selected_columns(df, columns):
    new_df = df.copy()
    new_df = new_df.drop(columns=columns)
    return new_df

def create_formula_column(df, new_column_name, formula_type, col_a=None, col_b=None):
    new_df = df.copy()

    if formula_type == "colA / colB":
        new_df[new_column_name] = new_df[col_a] / new_df[col_b]

    elif formula_type == "log(colA)":
        new_df[new_column_name] = np.log(new_df[col_a])

    elif formula_type == "colA - mean(colA)":
        new_df[new_column_name] = new_df[col_a] - new_df[col_a].mean()

    else:
        raise ValueError("Unsupported formula type.")

    return new_df

def bin_numeric_column(df, column, new_column_name, method="equal_width", bins=4, labels=None):
    new_df = df.copy()

    if labels is not None and len(labels) != bins:
        raise ValueError("Number of labels must match number of bins.")

    if method == "equal_width":
        new_df[new_column_name] = pd.cut(
            new_df[column],
            bins=bins,
            labels=labels
        )

    elif method == "quantile":
        new_df[new_column_name] = pd.qcut(
            new_df[column],
            q=bins,
            labels=labels,
            duplicates="drop"
        )

    else:
        raise ValueError("Unsupported binning method.")

    return new_df


def get_numeric_columns(df):
    return df.select_dtypes(include="number").columns.tolist()


def get_categorical_like_columns(df):
    return df.select_dtypes(include=["object", "category", "string", "bool"]).columns.tolist()


def apply_visual_filters(df, category_filter_col=None, category_filter_values=None,
                         numeric_filter_col=None, numeric_min=None, numeric_max=None):
    filtered_df = df.copy()

    if category_filter_col and category_filter_values:
        filtered_df = filtered_df[filtered_df[category_filter_col].isin(category_filter_values)]

    if numeric_filter_col and numeric_filter_col in filtered_df.columns:
        if numeric_min is not None:
            filtered_df = filtered_df[filtered_df[numeric_filter_col] >= numeric_min]
        if numeric_max is not None:
            filtered_df = filtered_df[filtered_df[numeric_filter_col] <= numeric_max]

    return filtered_df


def build_bar_chart_data(df, x_col, y_col=None, group_col=None, aggregation="count", top_n=None):
    plot_df = df.copy()

    if aggregation == "count":
        if group_col:
            grouped = plot_df.groupby([x_col, group_col], dropna=False).size().reset_index(name="value")
        else:
            grouped = plot_df.groupby(x_col, dropna=False).size().reset_index(name="value")
    else:
        if y_col is None:
            raise ValueError("y column is required for this aggregation.")

        if group_col:
            grouped = (
                plot_df.groupby([x_col, group_col], dropna=False)[y_col]
                .agg(aggregation)
                .reset_index(name="value")
            )
        else:
            grouped = (
                plot_df.groupby(x_col, dropna=False)[y_col]
                .agg(aggregation)
                .reset_index(name="value")
            )

    if top_n is not None and top_n > 0:
        if group_col:
            totals = grouped.groupby(x_col)["value"].sum().reset_index()
            top_categories = totals.nlargest(top_n, "value")[x_col]
            grouped = grouped[grouped[x_col].isin(top_categories)]
        else:
            grouped = grouped.nlargest(top_n, "value")

    return grouped


def ai_suggest_visualization(prompt, df):
    columns = df.columns.tolist()
    dtypes = df.dtypes.astype(str).to_dict()

    system_prompt = f"""
You are a data visualization assistant.

Dataset columns:
{columns}

Column data types:
{dtypes}

User request: "{prompt}"

Your job:
Suggest the BEST visualization approach.

Return STRICTLY in this format:

Chart Type: <one of: histogram, box plot, scatter plot, line chart, bar chart, heatmap>

X Column: <column or None>
Y Column: <column or None>

Optional Group Column: <column or None>

Preprocessing Steps:
- <step 1>
- <step 2>

Reason:
<short explanation>

Rules:
- If numeric distribution → histogram or box plot
- If relationship → scatter
- If time → line chart
- If categories → bar chart
- If multiple numeric → heatmap
- Suggest binning if needed
- Suggest aggregation if needed
- Be concise
"""

    try:

        text = os.getenv("API_KEY")
        client = Groq(api_key=text)

        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-versatile"
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error: {str(e)}"

def extract_sheet_id(sheet_url: str) -> str:
    match = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", sheet_url)
    if not match:
        raise ValueError("Invalid Google Sheets URL.")
    return match.group(1)

def load_google_sheet_public(sheet_url: str, gid: str | None = None) -> pd.DataFrame:
    sheet_id = extract_sheet_id(sheet_url)
    gid = gid if gid else "0"
    csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
    return pd.read_csv(csv_url)


def dataframe_to_excel_bytes(df):
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="cleaned_data")

    output.seek(0)
    return output.getvalue()
def transformation_log_to_json(log_data):
    return json.dumps(log_data, indent=4, default=str)
def build_recipe_json(log_data):
    recipe = {
        "recipe_name": "data_wrangling_recipe",
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "steps": log_data
    }
    return json.dumps(recipe, indent=4, default=str)
def build_python_replay_script(log_data):
    lines = [
        "import pandas as pd",
        "",
        "# Load your dataset first",
        "df = pd.read_csv('your_input_file.csv')",
        "",
        "# Replay transformation steps"
    ]

    for i, step in enumerate(log_data, start=1):
        operation = step.get("operation", "Unknown operation")
        parameters = step.get("parameters", {})
        lines.append(f"# Step {i}: {operation}")
        lines.append(f"# Parameters: {parameters}")
        lines.append("")

    lines.append("# Save result")
    lines.append("df.to_csv('replayed_output.csv', index=False)")

    return "\n".join(lines)

def build_dashboard_pdf_bytes(df, log_data, chart_figure=None, chart_title="Dashboard Chart"):
    pdf_buffer = io.BytesIO()

    with PdfPages(pdf_buffer) as pdf:
        # Page 1: summary
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis("off")

        summary_lines = [
            "AI-Assisted Data Wrangler & Visualizer",
            "",
            f"Generated at: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"Rows: {df.shape[0]}",
            f"Columns: {df.shape[1]}",
            "",
            f"Number of transformation steps: {len(log_data)}",
            "",
            "Recent transformation steps:"
        ]

        for i, step in enumerate(log_data[-10:], start=1):
            summary_lines.append(
                f"{i}. {step.get('operation', 'Unknown')} | {step.get('timestamp', '')}"
            )

        ax.text(
            0.01,
            0.99,
            "\n".join(summary_lines),
            va="top",
            ha="left",
            fontsize=10,
            wrap=True
        )

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 2: chart if available
        if chart_figure is not None:
            pdf.savefig(chart_figure, bbox_inches="tight")

    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()

def build_report_pdf_bytes(df, log_data):
    pdf_buffer = io.BytesIO()

    with PdfPages(pdf_buffer) as pdf:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis("off")

        summary_lines = [
            "AI-Assisted Data Wrangler & Visualizer",
            "",
            f"Generated at: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"Rows: {df.shape[0]}",
            f"Columns: {df.shape[1]}",
            "",
            f"Number of transformation steps: {len(log_data)}",
            "",
            "Recent transformation steps:"
        ]

        for i, step in enumerate(log_data, start=1):
            summary_lines.append(
                f"{i}. {step.get('operation', 'Unknown')} | {step.get('timestamp', '')}"
            )

        ax.text(
            0.01,
            0.99,
            "\n".join(summary_lines),
            va="top",
            ha="left",
            fontsize=10,
            wrap=True
        )

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()
# -----------------------------
# PAGE A — UPLOAD & OVERVIEW
# -----------------------------
if page == "Upload & Overview":
    st.header("Upload & Overview")

    uploaded_file = st.file_uploader(
        "Upload a dataset",
        type=["csv", "xlsx", "json"]
    )

    if uploaded_file is not None:
        try:
            df = load_file(uploaded_file.name, uploaded_file)

            st.session_state["original_df"] = df.copy()
            st.session_state["working_df"] = df.copy()
            st.session_state["uploaded_filename"] = uploaded_file.name

            if not st.session_state["transformation_log"]:
                add_log_step(
                    operation_name="File upload",
                    parameters={"filename": uploaded_file.name},
                    affected_columns=list(df.columns)
                )

            st.success(f"Loaded file: {uploaded_file.name}")

        except Exception as e:
            st.error(f"Error loading file: {e}")

    if st.session_state["working_df"] is not None:
        df = st.session_state["working_df"]

        # Top metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Duplicates", int(df.duplicated().sum()))

        st.subheader("Preview")
        st.dataframe(df.head(20), width="stretch")

        st.subheader("Column Data Types")
        st.dataframe(get_dtype_summary(df), width="stretch")

        st.subheader("Summary Statistics")
        st.write("Numeric summary:")
        numeric_df = df.select_dtypes(include="number")

        if not numeric_df.empty:
            st.dataframe(numeric_df.describe().T, width="stretch")
        else:
            st.info("No numeric columns found.")

        st.write("Categorical summary:")
        try:
            st.dataframe(df.describe(include=["object", "category"]).T, width="stretch")
        except ValueError:
            st.info("No categorical columns found.")

        st.subheader("Missing Values Summary")
        st.dataframe(get_missing_summary(df), width="stretch")

        st.subheader("Column Names")
        st.write(list(df.columns))
    data_source = st.radio(
        "Choose data source",
        options=["Upload File", "Google Sheets"],
        horizontal=True
    )

    if data_source == "Google Sheets":
        sheet_url = st.text_input("Paste Google Sheets URL")
        gid = st.text_input("Worksheet gid (optional, default 0)")

        if st.button("Load Google Sheet"):
            try:
                df = load_google_sheet_public(sheet_url, gid if gid.strip() else None)

                st.session_state["original_df"] = df.copy()
                st.session_state["working_df"] = df.copy()
                st.session_state["uploaded_filename"] = "Google Sheet"

                add_log_step(
                    operation_name="Google Sheets import",
                    parameters={
                        "sheet_url": sheet_url,
                        "gid": gid if gid.strip() else "0"
                    },
                    affected_columns=list(df.columns)
                )

                st.success("Google Sheet loaded successfully.")
            except Exception as e:
                st.error(f"Error loading Google Sheet: {e}")
    else:
        st.info("Please upload a CSV, XLSX, or JSON file to begin.")


# -----------------------------
# PAGE B — CLEANING & PREPARATION
# -----------------------------
elif page == "Cleaning & Preparation":
    st.header("Cleaning & Preparation Studio")

    if st.session_state["working_df"] is None:
        st.warning("Please upload a dataset first.")
    else:
        df = st.session_state["working_df"]

        st.subheader("Current Dataset Preview")
        st.dataframe(df.head(10), width="stretch")

        st.subheader("Current Dataset Shape")
        col1, col2 = st.columns(2)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])

        st.markdown("---")

        cleaning_section = st.radio(
            "Choose cleaning section",
            options=[
                "Missing Values + Data Types",
                "Duplicates",
                "Categorical / String Cleaning",
                "Numeric Cleaning",
                "Scaling",
                "Column Operations"
            ],
            horizontal=True
        )
        # =========================================================
        # SUBPAGE 1 — MISSING VALUES + DATA TYPES
        # =========================================================
        if cleaning_section == "Missing Values + Data Types":
            st.subheader("Data Types & Parsing")

            st.dataframe(get_dtype_summary(df), width="stretch")

            with st.expander("Convert a column to another type"):
                selected_column = st.selectbox(
                    "Select column",
                    options=df.columns.tolist(),
                    key="cast_selected_column"
                )

                target_type = st.selectbox(
                    "Cast selected column to",
                    options=["numeric", "datetime", "category", "string"],
                    key="cast_target_type"
                )

                datetime_format = ""
                dayfirst = False

                if target_type == "datetime":
                    datetime_format = st.text_input(
                        "Optional datetime format (example: %d/%m/%Y or %Y-%m-%d). Leave blank for flexible parsing.",
                        key="cast_datetime_format"
                    )
                    dayfirst = st.checkbox(
                        "Treat day as first in ambiguous dates",
                        value=False,
                        key="cast_dayfirst"
                    )

                if st.button("Preview conversion result", key="btn_preview_cast"):
                    try:
                        if target_type == "numeric":
                            preview_df, report = convert_to_numeric_with_report(df, selected_column)

                        elif target_type == "datetime":
                            preview_df, report = convert_to_datetime_with_report(
                                df,
                                selected_column,
                                datetime_format=datetime_format,
                                dayfirst=dayfirst
                            )

                        elif target_type == "category":
                            preview_df, report = convert_to_category_with_report(df, selected_column)

                        else:
                            preview_df, report = convert_to_string_with_report(df, selected_column)

                        st.session_state["cast_preview_df"] = preview_df
                        st.session_state["cast_preview_report"] = report
                        st.session_state["cast_preview_column"] = selected_column
                        st.session_state["cast_preview_target_type"] = target_type

                    except Exception as e:
                        st.error(f"Preview failed: {e}")

                if "cast_preview_report" in st.session_state:
                    report = st.session_state["cast_preview_report"]
                    preview_df = st.session_state["cast_preview_df"]

                    st.write("Conversion report:")
                    st.write("Column:", report["column"])
                    st.write("Before dtype:", report["before_dtype"])
                    st.write("After dtype:", report["after_dtype"])

                    if "parsed_successfully" in report:
                        st.write("Successfully parsed:", report["parsed_successfully"])
                        st.write("Failed parses:", report["failed_count"])
                        st.write("Missing before:", report["missing_before"])
                        st.write("Missing after:", report["missing_after"])

                        if report["failed_examples"]:
                            st.write("Examples of values that could not be parsed:")
                            st.write(report["failed_examples"])

                    if "unique_before" in report:
                        st.write("Unique values before:", report["unique_before"])
                        st.write("Unique values after:", report["unique_after"])

                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("Before preview:")
                        st.dataframe(df[[selected_column]].head(10), width="stretch")

                    with col2:
                        st.write("After preview:")
                        st.dataframe(preview_df[[selected_column]].head(10), width="stretch")

                    if st.button("Apply conversion", key="btn_apply_cast"):
                        st.session_state["working_df"] = preview_df

                        add_log_step(
                            operation_name="Convert column type",
                            parameters={
                                "column": report["column"],
                                "target_type": st.session_state["cast_preview_target_type"],
                                "failed_count": report.get("failed_count", 0),
                                "datetime_format": datetime_format if target_type == "datetime" else None,
                                "dayfirst": dayfirst if target_type == "datetime" else None,
                            },
                            affected_columns=[report["column"]]
                        )

                        st.success("Conversion applied successfully.")

            # -----------------------------------
            # MISSING VALUES SECTION
            # -----------------------------------
            st.markdown("---")
            st.subheader("Missing Values Handling")

            missing_summary = get_missing_summary(df)
            st.write("Missing value summary by column:")
            st.dataframe(missing_summary, width="stretch")

            with st.expander("Drop rows with missing values in selected columns"):
                selectable_columns = df.columns.tolist()

                selected_columns_for_drop = st.multiselect(
                    "Select columns to check for missing values",
                    options=selectable_columns,
                    key="drop_missing_cols"
                )

                if st.button("Apply: Drop rows", key="btn_drop_rows_missing"):
                    if not selected_columns_for_drop:
                        st.warning("Please select at least one column.")
                    else:
                        before_df = df.copy()
                        new_df, impact = drop_rows_with_missing(df, selected_columns_for_drop)

                        st.session_state["working_df"] = new_df
                        add_log_step(
                            operation_name="Drop rows with missing values",
                            parameters={"selected_columns": selected_columns_for_drop},
                            affected_columns=selected_columns_for_drop
                        )

                        st.success("Rows dropped successfully.")
                        st.write("Before shape:", before_df.shape)
                        st.write("After shape:", new_df.shape)
                        st.write("Rows removed:", impact["rows_removed"])

            with st.expander("Drop columns with missing % above threshold"):
                threshold = st.slider(
                    "Missing value threshold (%)",
                    min_value=0,
                    max_value=100,
                    value=50,
                    step=1,
                    key="missing_threshold_slider"
                )

                if st.button("Apply: Drop columns above threshold", key="btn_drop_cols_threshold"):
                    before_df = df.copy()
                    new_df, impact = drop_columns_above_threshold(df, threshold)

                    st.session_state["working_df"] = new_df
                    add_log_step(
                        operation_name="Drop columns above missing threshold",
                        parameters={"threshold_percent": threshold},
                        affected_columns=impact["columns_removed"]
                    )

                    st.success("Column threshold operation completed.")
                    st.write("Before shape:", before_df.shape)
                    st.write("After shape:", new_df.shape)
                    st.write("Columns removed:", impact["columns_removed"])

            with st.expander("Fill missing values in a selected column"):
                column_with_missing = st.selectbox(
                    "Select column",
                    options=df.columns.tolist(),
                    key="fill_missing_column"
                )

                strategy_options = [
                    "constant",
                    "mean",
                    "median",
                    "mode",
                    "most_frequent",
                    "forward_fill",
                    "backward_fill"
                ]

                selected_strategy = st.selectbox(
                    "Select fill strategy",
                    options=strategy_options,
                    key="fill_missing_strategy"
                )

                constant_value = None
                if selected_strategy == "constant":
                    constant_value = st.text_input(
                        "Enter constant replacement value",
                        key="constant_fill_value"
                    )

                if st.button("Apply: Fill missing values", key="btn_fill_missing"):
                    try:
                        before_df = df.copy()

                        new_df, impact = fill_missing_values(
                            df=df,
                            column=column_with_missing,
                            strategy=selected_strategy,
                            constant_value=constant_value
                        )

                        st.session_state["working_df"] = new_df
                        add_log_step(
                            operation_name="Fill missing values",
                            parameters={
                                "column": column_with_missing,
                                "strategy": selected_strategy,
                                "constant_value": constant_value
                            },
                            affected_columns=[column_with_missing]
                        )

                        st.success("Missing value operation applied successfully.")
                        st.write("Column:", impact["column"])
                        st.write("Strategy:", impact["strategy"])
                        st.write("Missing before:", impact["missing_before"])
                        st.write("Missing after:", impact["missing_after"])
                        st.write("Values filled:", impact["filled_count"])

                        st.write("Before preview:")
                        st.dataframe(before_df[[column_with_missing]].head(10), width="stretch")

                        st.write("After preview:")
                        st.dataframe(new_df[[column_with_missing]].head(10), width="stretch")

                    except Exception as e:
                        st.error(f"Error: {e}")

        # =========================================================
        # SUBPAGE 2 — DUPLICATES
        # =========================================================
        elif cleaning_section == "Duplicates":
            st.subheader("Duplicates Handling")

            duplicate_mode = st.radio(
                "Duplicate detection mode",
                options=[
                    "Full-row duplicates",
                    "Duplicates by selected columns"
                ],
                horizontal=True
            )

            if duplicate_mode == "Full-row duplicates":
                summary = get_duplicate_summary(df)

                st.write("Duplicate rows found:", summary["duplicate_count"])

                if not summary["duplicate_rows"].empty:
                    st.dataframe(summary["duplicate_rows"], width="stretch")
                else:
                    st.info("No full-row duplicates found.")

                st.markdown("### Remove full-row duplicates")
                keep_option = st.selectbox(
                    "When duplicates are found, keep:",
                    options=["first", "last"],
                    key="dup_keep_fullrow"
                )

                if st.button("Apply: Remove full-row duplicates", key="btn_remove_fullrow_duplicates"):
                    before_df = df.copy()
                    new_df, impact = remove_duplicates(df, subset_columns=None, keep_option=keep_option)

                    st.session_state["working_df"] = new_df
                    add_log_step(
                        operation_name="Remove full-row duplicates",
                        parameters={"keep_option": keep_option},
                        affected_columns=[]
                    )

                    st.success("Full-row duplicates removed.")
                    st.write("Before shape:", before_df.shape)
                    st.write("After shape:", new_df.shape)
                    st.write("Rows removed:", impact["rows_removed"])

            else:
                subset_columns = st.multiselect(
                    "Select columns that define duplicate groups",
                    options=df.columns.tolist(),
                    key="dup_subset_columns"
                )

                if subset_columns:
                    summary = get_duplicate_summary(df, subset_columns=subset_columns)

                    st.write("Duplicate rows found:", summary["duplicate_count"])

                    if not summary["duplicate_rows"].empty:
                        st.dataframe(summary["duplicate_rows"], width="stretch")
                    else:
                        st.info("No duplicates found for the selected columns.")

                    st.markdown("### Remove duplicates by selected columns")

                    removal_mode = st.radio(
                        "Choose duplicate removal method",
                        options=[
                            "Keep first / last",
                            "Keep by order column"
                        ],
                        horizontal=True,
                        key="dup_removal_mode"
                    )

                    if removal_mode == "Keep first / last":
                        keep_option = st.selectbox(
                            "When duplicates are found, keep:",
                            options=["first", "last"],
                            key="dup_keep_subset"
                        )

                        if st.button("Apply: Remove subset duplicates", key="btn_remove_subset_duplicates"):
                            before_df = df.copy()
                            new_df, impact = remove_duplicates(
                                df,
                                subset_columns=subset_columns,
                                keep_option=keep_option
                            )

                            st.session_state["working_df"] = new_df
                            add_log_step(
                                operation_name="Remove duplicates by selected columns",
                                parameters={
                                    "subset_columns": subset_columns,
                                    "keep_option": keep_option
                                },
                                affected_columns=subset_columns
                            )

                            st.success("Subset-based duplicates removed.")
                            st.write("Before shape:", before_df.shape)
                            st.write("After shape:", new_df.shape)
                            st.write("Rows removed:", impact["rows_removed"])

                    else:
                        order_column = st.selectbox(
                            "Select order column",
                            options=df.columns.tolist(),
                            key="dup_order_column"
                        )

                        keep_highest = st.radio(
                            "Which row should be kept in each duplicate group?",
                            options=["Keep highest value", "Keep lowest value"],
                            key="dup_keep_order_choice"
                        )

                        if st.button("Apply: Remove duplicates by order column", key="btn_remove_duplicates_order"):
                            try:
                                before_df = df.copy()

                                new_df, impact = remove_duplicates_keep_by_order(
                                    df=df,
                                    subset_columns=subset_columns,
                                    order_column=order_column,
                                    keep_highest=(keep_highest == "Keep highest value")
                                )

                                st.session_state["working_df"] = new_df
                                add_log_step(
                                    operation_name="Remove duplicates by order column",
                                    parameters={
                                        "subset_columns": subset_columns,
                                        "order_column": order_column,
                                        "keep": "highest" if keep_highest == "Keep highest value" else "lowest"
                                    },
                                    affected_columns=subset_columns + [order_column]
                                )

                                st.success("Duplicates removed using order column.")
                                st.write("Before shape:", before_df.shape)
                                st.write("After shape:", new_df.shape)
                                st.write("Rows removed:", impact["rows_removed"])

                            except Exception as e:
                                st.error(f"Error: {e}")
                else:
                    st.info("Select one or more columns to detect duplicates by subset.")



        # =========================================================
        # SUBPAGE 3 — CATEGORICAL / STRING CLEANING
        # =========================================================
        elif cleaning_section == "Categorical / String Cleaning":
            st.subheader("Categorical / String Cleaning")

            string_like_columns = df.columns.tolist()

            selected_text_column = st.selectbox(
                "Select column to clean",
                options=string_like_columns,
                key="text_clean_column"
            )

            st.write("Current preview:")
            st.dataframe(df[[selected_text_column]].head(10), width="stretch")

            with st.expander("Clean selected text column", expanded=True):
                trim_whitespace = st.checkbox(
                    "Trim leading and trailing whitespace",
                    value=True,
                    key="text_trim_whitespace"
                )

                remove_all_spaces = st.checkbox(
                    "Remove all spaces inside the text",
                    value=False,
                    key="text_remove_all_spaces"
                )

                remove_quotes = st.checkbox(
                    "Remove single and double quotes",
                    value=False,
                    key="text_remove_quotes"
                )

                case_option = st.selectbox(
                    "Case standardization",
                    options=["no_change", "lower", "upper", "title"],
                    key="text_case_option"
                )

                st.markdown("### Replace text / symbol")
                find_text = st.text_input(
                    "Find",
                    value="",
                    key="text_find_text"
                )

                replace_text = st.text_input(
                    "Replace with",
                    value="",
                    key="text_replace_text"
                )

                if st.button("Preview text cleaning", key="btn_preview_text_clean"):
                    try:
                        preview_df, report = clean_text_column(
                            df=df,
                            column=selected_text_column,
                            trim_whitespace=trim_whitespace,
                            remove_all_spaces=remove_all_spaces,
                            remove_quotes=remove_quotes,
                            case_option=case_option,
                            find_text=find_text,
                            replace_text=replace_text
                        )

                        st.session_state["text_clean_preview_df"] = preview_df
                        st.session_state["text_clean_preview_report"] = report
                        st.session_state["text_clean_preview_column"] = selected_text_column

                    except Exception as e:
                        st.error(f"Preview failed: {e}")

                if "text_clean_preview_report" in st.session_state:
                    report = st.session_state["text_clean_preview_report"]
                    preview_df = st.session_state["text_clean_preview_df"]
                    preview_column = st.session_state["text_clean_preview_column"]

                    if preview_column == selected_text_column:
                        st.write("Cleaning report:")
                        st.write("Column:", report["column"])
                        st.write("Changed values:", report["changed_count"])
                        st.write("Trim whitespace:", report["trim_whitespace"])
                        st.write("Remove all spaces:", report["remove_all_spaces"])
                        st.write("Remove quotes:", report["remove_quotes"])
                        st.write("Case option:", report["case_option"])
                        st.write("Find text:", report["find_text"])
                        st.write("Replace text:", report["replace_text"])

                        col1, col2 = st.columns(2)

                        with col1:
                            st.write("Before preview:")
                            st.dataframe(df[[selected_text_column]].head(10), width="stretch")

                        with col2:
                            st.write("After preview:")
                            st.dataframe(preview_df[[selected_text_column]].head(10), width="stretch")

                        if st.button("Apply text cleaning", key="btn_apply_text_clean"):
                            st.session_state["working_df"] = preview_df

                            add_log_step(
                                operation_name="Categorical / string cleaning",
                                parameters={
                                    "column": report["column"],
                                    "trim_whitespace": report["trim_whitespace"],
                                    "remove_all_spaces": report["remove_all_spaces"],
                                    "remove_quotes": report["remove_quotes"],
                                    "case_option": report["case_option"],
                                    "find_text": report["find_text"],
                                    "replace_text": report["replace_text"],
                                    "changed_count": report["changed_count"],
                                },
                                affected_columns=[report["column"]]
                            )

                            st.success("Text cleaning applied successfully.")
            st.markdown("---")
            with st.expander("Rare category grouping"):
                rare_group_column = st.selectbox(
                    "Select categorical column",
                    options=df.columns.tolist(),
                    key="rare_group_column"
                )

                min_frequency = st.number_input(
                    "Minimum frequency required to keep a category",
                    min_value=1,
                    value=5,
                    step=1,
                    key="rare_group_min_frequency"
                )

                other_label = st.text_input(
                    "Replacement label for rare categories",
                    value="Other",
                    key="rare_group_other_label"
                )

                if st.button("Preview rare category grouping", key="btn_preview_rare_grouping"):
                    try:
                        preview_df, report = group_rare_categories(
                            df=df,
                            column=rare_group_column,
                            min_frequency=min_frequency,
                            other_label=other_label
                        )

                        st.session_state["rare_group_preview_df"] = preview_df
                        st.session_state["rare_group_preview_report"] = report
                        st.session_state["rare_group_preview_column"] = rare_group_column

                    except Exception as e:
                        st.error(f"Preview failed: {e}")

                if "rare_group_preview_report" in st.session_state:
                    report = st.session_state["rare_group_preview_report"]
                    preview_df = st.session_state["rare_group_preview_df"]
                    preview_column = st.session_state["rare_group_preview_column"]

                    if preview_column == rare_group_column:
                        st.write("Rare category grouping report:")
                        st.write("Column:", report["column"])
                        st.write("Minimum frequency:", report["min_frequency"])
                        st.write("Replacement label:", report["other_label"])
                        st.write("Number of rare categories grouped:", report["rare_category_count"])
                        st.write("Rows replaced:", report["replaced_count"])

                        if report["rare_categories"]:
                            st.write("Examples of grouped categories:")
                            st.write(report["rare_categories"])

                        before_counts = (
                            df[rare_group_column]
                            .value_counts(dropna=False)
                            .reset_index(name="frequency")
                            .rename(columns={"index": "category_value"})
                        )

                        after_counts = (
                            preview_df[rare_group_column]
                            .value_counts(dropna=False)
                            .reset_index(name="frequency")
                            .rename(columns={"index": "category_value"})
                        )

                        col1, col2 = st.columns(2)

                        with col1:
                            st.write("Before value counts:")
                            st.dataframe(before_counts, width="stretch")

                        with col2:
                            st.write("After value counts:")
                            st.dataframe(after_counts, width="stretch")

                        if st.button("Apply rare category grouping", key="btn_apply_rare_grouping"):
                            st.session_state["working_df"] = preview_df

                            add_log_step(
                                operation_name="Rare category grouping",
                                parameters={
                                    "column": report["column"],
                                    "min_frequency": report["min_frequency"],
                                    "other_label": report["other_label"],
                                    "rare_category_count": report["rare_category_count"],
                                    "replaced_count": report["replaced_count"]
                                },
                                affected_columns=[report["column"]]
                            )

                            st.success("Rare category grouping applied successfully.")

            st.markdown("---")
            with st.expander("One-hot encoding"):
                one_hot_columns = st.multiselect(
                    "Select columns to one-hot encode",
                    options=df.columns.tolist(),
                    key="one_hot_columns"
                )

                drop_first = st.checkbox(
                    "Drop first dummy column",
                    value=False,
                    key="one_hot_drop_first"
                )

                dummy_na = st.checkbox(
                    "Create a separate column for missing values",
                    value=False,
                    key="one_hot_dummy_na"
                )

                if st.button("Preview one-hot encoding", key="btn_preview_one_hot"):
                    if not one_hot_columns:
                        st.warning("Please select at least one column.")
                    else:
                        try:
                            preview_df, report = one_hot_encode_columns(
                                df=df,
                                columns=one_hot_columns,
                                drop_first=drop_first,
                                dummy_na=dummy_na
                            )

                            st.session_state["one_hot_preview_df"] = preview_df
                            st.session_state["one_hot_preview_report"] = report

                        except Exception as e:
                            st.error(f"Preview failed: {e}")

                if "one_hot_preview_report" in st.session_state:
                    report = st.session_state["one_hot_preview_report"]
                    preview_df = st.session_state["one_hot_preview_df"]

                    st.write("One-hot encoding report:")
                    st.write("Encoded columns:", report["encoded_columns"])
                    st.write("Drop first:", report["drop_first"])
                    st.write("Dummy NA:", report["dummy_na"])
                    st.write("Number of created columns:", report["created_column_count"])

                    if report["created_columns_preview"]:
                        st.write("Created columns preview:")
                        st.write(report["created_columns_preview"])

                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("Before shape:")
                        st.write(df.shape)

                    with col2:
                        st.write("After shape:")
                        st.write(preview_df.shape)

                    st.write("Preview of encoded dataframe:")
                    st.dataframe(preview_df.head(10), width="stretch")

                    if st.button("Apply one-hot encoding", key="btn_apply_one_hot"):
                        st.session_state["working_df"] = preview_df

                        add_log_step(
                            operation_name="One-hot encoding",
                            parameters={
                                "encoded_columns": report["encoded_columns"],
                                "drop_first": report["drop_first"],
                                "dummy_na": report["dummy_na"],
                                "created_column_count": report["created_column_count"]
                            },
                            affected_columns=report["encoded_columns"]
                        )

                        st.success("One-hot encoding applied successfully.")

                # =========================================================
        # SUBPAGE 4 — NUMERIC CLEANING
        # =========================================================
        elif cleaning_section == "Numeric Cleaning":
            st.subheader("Numeric Cleaning")

            numeric_columns = get_numeric_columns(df)

            if not numeric_columns:
                st.info("No numeric columns found. Convert columns to numeric first.")
            else:
                selected_numeric_column = st.selectbox(
                    "Select numeric column",
                    options=numeric_columns,
                    key="numeric_clean_column"
                )

                outlier_info = detect_outliers_iqr(df, selected_numeric_column)

                st.write("Outlier detection summary (IQR method):")
                summary_df = pd.DataFrame([{
                    "column": outlier_info["column"],
                    "q1": outlier_info["q1"],
                    "q3": outlier_info["q3"],
                    "iqr": outlier_info["iqr"],
                    "lower_bound": outlier_info["lower_bound"],
                    "upper_bound": outlier_info["upper_bound"],
                    "outlier_count": outlier_info["outlier_count"]
                }])
                st.dataframe(summary_df, width="stretch")

                numeric_action = st.radio(
                    "Choose action",
                    options=[
                        "Do nothing",
                        "Cap / winsorize at quantiles",
                        "Remove outlier rows"
                    ],
                    horizontal=True,
                    key="numeric_clean_action"
                )

                if numeric_action == "Cap / winsorize at quantiles":
                    lower_q = st.slider(
                        "Lower quantile",
                        min_value=0.0,
                        max_value=0.49,
                        value=0.05,
                        step=0.01,
                        key="winsor_lower_q"
                    )

                    upper_q = st.slider(
                        "Upper quantile",
                        min_value=0.51,
                        max_value=1.0,
                        value=0.95,
                        step=0.01,
                        key="winsor_upper_q"
                    )

                    if st.button("Preview winsorization", key="btn_preview_winsor"):
                        try:
                            preview_df, report = winsorize_column(
                                df,
                                selected_numeric_column,
                                lower_quantile=lower_q,
                                upper_quantile=upper_q
                            )
                            st.session_state["winsor_preview_df"] = preview_df
                            st.session_state["winsor_preview_report"] = report
                            st.session_state["winsor_preview_column"] = selected_numeric_column
                        except Exception as e:
                            st.error(f"Preview failed: {e}")

                    if "winsor_preview_report" in st.session_state:
                        report = st.session_state["winsor_preview_report"]
                        preview_df = st.session_state["winsor_preview_df"]
                        preview_column = st.session_state["winsor_preview_column"]

                        if preview_column == selected_numeric_column:
                            st.write("Winsorization report:")
                            st.write("Column:", report["column"])
                            st.write("Lower cap:", report["lower_cap"])
                            st.write("Upper cap:", report["upper_cap"])
                            st.write("Values capped:", report["changed_count"])

                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("Before preview:")
                                st.dataframe(df[[selected_numeric_column]].head(10), width="stretch")
                            with col2:
                                st.write("After preview:")
                                st.dataframe(preview_df[[selected_numeric_column]].head(10), width="stretch")

                            if st.button("Apply winsorization", key="btn_apply_winsor"):
                                st.session_state["working_df"] = preview_df

                                add_log_step(
                                    operation_name="Winsorize numeric column",
                                    parameters={
                                        "column": report["column"],
                                        "lower_quantile": report["lower_quantile"],
                                        "upper_quantile": report["upper_quantile"],
                                        "changed_count": report["changed_count"]
                                    },
                                    affected_columns=[report["column"]]
                                )

                                st.success("Winsorization applied successfully.")

                elif numeric_action == "Remove outlier rows":
                    if st.button("Preview outlier row removal", key="btn_preview_remove_outliers"):
                        try:
                            preview_df, report = remove_outlier_rows_iqr(df, selected_numeric_column)
                            st.session_state["remove_outlier_preview_df"] = preview_df
                            st.session_state["remove_outlier_preview_report"] = report
                            st.session_state["remove_outlier_preview_column"] = selected_numeric_column
                        except Exception as e:
                            st.error(f"Preview failed: {e}")

                    if "remove_outlier_preview_report" in st.session_state:
                        report = st.session_state["remove_outlier_preview_report"]
                        preview_df = st.session_state["remove_outlier_preview_df"]
                        preview_column = st.session_state["remove_outlier_preview_column"]

                        if preview_column == selected_numeric_column:
                            st.write("Outlier row removal report:")
                            st.write("Column:", report["column"])
                            st.write("Lower bound:", report["lower_bound"])
                            st.write("Upper bound:", report["upper_bound"])
                            st.write("Rows removed:", report["rows_removed"])
                            st.write("Before shape:", (report["rows_before"], df.shape[1]))
                            st.write("After shape:", preview_df.shape)

                            if st.button("Apply outlier row removal", key="btn_apply_remove_outliers"):
                                st.session_state["working_df"] = preview_df

                                add_log_step(
                                    operation_name="Remove outlier rows",
                                    parameters={
                                        "column": report["column"],
                                        "rows_removed": report["rows_removed"],
                                        "lower_bound": report["lower_bound"],
                                        "upper_bound": report["upper_bound"]
                                    },
                                    affected_columns=[report["column"]]
                                )

                                st.success("Outlier rows removed successfully.")

                else:
                    st.info("No cleaning action selected.")
        # =========================================================
        # SUBPAGE 5 — SCALING
        # =========================================================
        elif cleaning_section == "Scaling":
            st.subheader("Normalization / Scaling")

            numeric_columns = get_numeric_columns(df)

            if not numeric_columns:
                st.info("No numeric columns found. Convert columns to numeric first.")
            else:
                selected_scale_columns = st.multiselect(
                    "Select numeric columns to scale",
                    options=numeric_columns,
                    key="scale_columns"
                )

                scaling_method = st.radio(
                    "Scaling method",
                    options=["Min-max scaling", "Z-score standardization"],
                    horizontal=True,
                    key="scaling_method"
                )

                if st.button("Preview scaling", key="btn_preview_scaling"):
                    if not selected_scale_columns:
                        st.warning("Please select at least one numeric column.")
                    else:
                        try:
                            if scaling_method == "Min-max scaling":
                                preview_df, report_df = min_max_scale_columns(df, selected_scale_columns)
                            else:
                                preview_df, report_df = zscore_scale_columns(df, selected_scale_columns)

                            st.session_state["scaling_preview_df"] = preview_df
                            st.session_state["scaling_preview_report_df"] = report_df
                            st.session_state["scaling_preview_columns"] = selected_scale_columns
                            st.session_state["scaling_preview_method"] = scaling_method

                        except Exception as e:
                            st.error(f"Preview failed: {e}")

                if "scaling_preview_report_df" in st.session_state:
                    report_df = st.session_state["scaling_preview_report_df"]
                    preview_df = st.session_state["scaling_preview_df"]

                    st.write("Before / after stats:")
                    st.dataframe(report_df, width="stretch")

                    st.write("Preview of scaled dataframe:")
                    st.dataframe(preview_df[selected_scale_columns].head(10), width="stretch")

                    if st.button("Apply scaling", key="btn_apply_scaling"):
                        st.session_state["working_df"] = preview_df

                        add_log_step(
                            operation_name="Scaling",
                            parameters={
                                "columns": st.session_state["scaling_preview_columns"],
                                "method": st.session_state["scaling_preview_method"]
                            },
                            affected_columns=st.session_state["scaling_preview_columns"]
                        )

                        st.success("Scaling applied successfully.")
        # =========================================================
        # SUBPAGE 6 — COLUMN OPERATIONS
        # =========================================================
        elif cleaning_section == "Column Operations":
            st.subheader("Column Operations")

            operation_mode = st.radio(
                "Choose operation",
                options=[
                    "Rename columns",
                    "Drop columns",
                    "Create new column with formula",
                    "Bin numeric column"
                ],
                horizontal=True,
                key="column_ops_mode"
            )

            if operation_mode == "Rename columns":
                old_name = st.selectbox(
                    "Select column to rename",
                    options=df.columns.tolist(),
                    key="rename_old_name"
                )

                new_name = st.text_input(
                    "Enter new column name",
                    key="rename_new_name"
                )

                if st.button("Apply rename", key="btn_apply_rename"):
                    if not new_name.strip():
                        st.warning("Please enter a new column name.")
                    elif new_name in df.columns:
                        st.warning("That column name already exists.")
                    else:
                        new_df = rename_column(df, old_name, new_name)
                        st.session_state["working_df"] = new_df

                        add_log_step(
                            operation_name="Rename column",
                            parameters={"old_name": old_name, "new_name": new_name},
                            affected_columns=[old_name]
                        )

                        st.success("Column renamed successfully.")

            elif operation_mode == "Drop columns":
                columns_to_drop = st.multiselect(
                    "Select columns to drop",
                    options=df.columns.tolist(),
                    key="drop_columns_selected"
                )

                if st.button("Apply drop columns", key="btn_apply_drop_columns"):
                    if not columns_to_drop:
                        st.warning("Please select at least one column.")
                    else:
                        new_df = drop_selected_columns(df, columns_to_drop)
                        st.session_state["working_df"] = new_df

                        add_log_step(
                            operation_name="Drop columns",
                            parameters={"columns_dropped": columns_to_drop},
                            affected_columns=columns_to_drop
                        )

                        st.success("Columns dropped successfully.")

            elif operation_mode == "Create new column with formula":
                numeric_columns = get_numeric_columns(df)

                if not numeric_columns:
                    st.info("No numeric columns found. Convert columns to numeric first.")
                else:
                    new_column_name = st.text_input(
                        "New column name",
                        key="formula_new_column_name"
                    )

                    formula_type = st.selectbox(
                        "Formula type",
                        options=["colA / colB", "log(colA)", "colA - mean(colA)"],
                        key="formula_type"
                    )

                    col_a = st.selectbox(
                        "Select colA",
                        options=numeric_columns,
                        key="formula_col_a"
                    )

                    col_b = None
                    if formula_type == "colA / colB":
                        col_b = st.selectbox(
                            "Select colB",
                            options=numeric_columns,
                            key="formula_col_b"
                        )

                    if st.button("Apply create formula column", key="btn_apply_formula_column"):
                        try:
                            if not new_column_name.strip():
                                st.warning("Please enter a new column name.")
                            elif new_column_name in df.columns:
                                st.warning("That column name already exists.")
                            else:
                                new_df = create_formula_column(
                                    df=df,
                                    new_column_name=new_column_name,
                                    formula_type=formula_type,
                                    col_a=col_a,
                                    col_b=col_b
                                )

                                st.session_state["working_df"] = new_df

                                add_log_step(
                                    operation_name="Create formula column",
                                    parameters={
                                        "new_column_name": new_column_name,
                                        "formula_type": formula_type,
                                        "col_a": col_a,
                                        "col_b": col_b
                                    },
                                    affected_columns=[col_a] + ([col_b] if col_b else [])
                                )

                                st.success("New formula column created successfully.")
                                st.dataframe(new_df[[new_column_name]].head(10), width="stretch")

                        except Exception as e:
                            st.error(f"Error: {e}")

            else:
                numeric_columns = get_numeric_columns(df)

                if not numeric_columns:
                    st.info("No numeric columns found. Convert columns to numeric first.")
                else:
                    source_column = st.selectbox(
                    "Select numeric column to bin",
                    options=numeric_columns,
                    key="bin_source_column"
                )

                new_column_name = st.text_input(
                    "New binned column name",
                    key="bin_new_column_name"
                )

                bin_method = st.radio(
                    "Binning method",
                    options=["equal_width", "quantile"],
                    horizontal=True,
                    key="bin_method"
                )

                bins = st.number_input(
                    "Number of bins",
                    min_value=2,
                    max_value=10,
                    value=3,
                    step=1,
                    key="bin_count"
                )

                label_mode = st.radio(
                    "Label type",
                    options=["Auto labels", "Custom labels"],
                    horizontal=True,
                    key="bin_label_mode"
                )

                labels = None

                if label_mode == "Auto labels":
                    default_labels = ["Low", "Medium", "High", "Very High", "Extreme"]
                    labels = default_labels[:bins]
                    st.write("Generated labels:", labels)

                else:
                    label_input = st.text_input(
                        "Enter labels separated by comma (e.g. Low,Medium,High)",
                        key="bin_custom_labels"
                    )

                    if label_input:
                        labels = [x.strip() for x in label_input.split(",")]

                        if len(labels) != bins:
                            st.warning(f"You must provide exactly {bins} labels.")
                            labels = None

                if st.button("Apply binning", key="btn_apply_binning"):
                    try:
                        if not new_column_name.strip():
                            st.warning("Please enter a new column name.")

                        elif new_column_name in df.columns:
                            st.warning("That column name already exists.")

                        elif labels is None:
                            st.warning("Labels are invalid or missing.")

                        else:
                            new_df = bin_numeric_column(
                                df=df,
                                column=source_column,
                                new_column_name=new_column_name,
                                method=bin_method,
                                bins=bins,
                                labels=labels
                            )

                            st.session_state["working_df"] = new_df

                            add_log_step(
                                operation_name="Bin numeric column",
                                parameters={
                                    "source_column": source_column,
                                    "new_column_name": new_column_name,
                                    "method": bin_method,
                                    "bins": bins,
                                    "labels": labels
                                },
                                affected_columns=[source_column]
                            )

                            st.success("Binned column created successfully.")
                            st.dataframe(new_df[[source_column, new_column_name]].head(10), width="stretch")

                    except Exception as e:
                        st.error(f"Error: {e}")

# -----------------------------
# PAGE C — VISUALIZATION BUILDER
# -----------------------------
# -----------------------------
# PAGE C — VISUALIZATION BUILDER
# -----------------------------
elif page == "Visualization Builder":
    st.header("Visualization Builder")

    if st.session_state["working_df"] is None:
        st.warning("Please upload a dataset first.")
    else:

        df = st.session_state["working_df"]

        numeric_columns = get_numeric_columns(df)
        categorical_columns = get_categorical_like_columns(df)
        all_columns = df.columns.tolist()

        st.markdown("---")
        st.subheader("AI Chart Assistant")

        user_prompt = st.text_input(
            "Describe what you want to visualize (e.g., 'show salary distribution', 'compare sales by region')",
            key="ai_viz_prompt"
        )

        if st.button("Get AI suggestion", key="btn_ai_viz"):
            if not user_prompt.strip():
                st.warning("Please enter a prompt.")
            else:
                result = ai_suggest_visualization(user_prompt, df)
                st.write(result)


        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), width="stretch")

        st.markdown("---")
        st.subheader("Filters")

        # Category filter
        category_filter_col = st.selectbox(
            "Filter by category column (optional)",
            options=["None"] + categorical_columns,
            key="viz_category_filter_col"
        )

        category_filter_values = []
        if category_filter_col != "None":
            available_values = df[category_filter_col].dropna().astype(str).unique().tolist()
            available_values = sorted(available_values)
            category_filter_values = st.multiselect(
                "Select category values",
                options=available_values,
                key="viz_category_filter_values"
            )

        # Numeric range filter
        numeric_filter_col = st.selectbox(
            "Filter by numeric column (optional)",
            options=["None"] + numeric_columns,
            key="viz_numeric_filter_col"
        )

        numeric_min = None
        numeric_max = None
        if numeric_filter_col != "None":
            col_min = float(df[numeric_filter_col].min())
            col_max = float(df[numeric_filter_col].max())

            numeric_min, numeric_max = st.slider(
                "Select numeric range",
                min_value=col_min,
                max_value=col_max,
                value=(col_min, col_max),
                key="viz_numeric_range"
            )

        filtered_df = apply_visual_filters(
            df=df,
            category_filter_col=None if category_filter_col == "None" else category_filter_col,
            category_filter_values=category_filter_values,
            numeric_filter_col=None if numeric_filter_col == "None" else numeric_filter_col,
            numeric_min=numeric_min,
            numeric_max=numeric_max
        )

        st.write("Filtered dataset shape:", filtered_df.shape)

        st.markdown("---")
        st.subheader("Choose Your Chart")

        plot_type = st.selectbox(
            "Plot type",
            options=[
                "Histogram",
                "Box Plot",
                "Scatter Plot",
                "Line Chart",
                "Bar Chart",
                "Correlation Heatmap"
            ],
            key="viz_plot_type"
        )

        aggregation = None
        x_col = None
        y_col = None
        group_col = None
        top_n = None

        if plot_type == "Histogram":
            if not numeric_columns:
                st.info("No numeric columns available for histogram.")
            else:
                x_col = st.selectbox(
                    "Select numeric column",
                    options=numeric_columns,
                    key="hist_x_col"
                )

                bins = st.slider(
                    "Number of bins",
                    min_value=5,
                    max_value=100,
                    value=20,
                    key="hist_bins"
                )

                fig, ax = plt.subplots()
                ax.hist(filtered_df[x_col].dropna(), bins=bins)
                ax.set_title(f"Histogram of {x_col}")
                ax.set_xlabel(x_col)
                ax.set_ylabel("Frequency")
                st.pyplot(fig)
                st.session_state["latest_chart_fig"] = fig
                st.session_state["latest_chart_title"] = plot_type

        elif plot_type == "Box Plot":
            if not numeric_columns:
                st.info("No numeric columns available for box plot.")
            else:
                y_col = st.selectbox(
                    "Select numeric column",
                    options=numeric_columns,
                    key="box_y_col"
                )

                fig, ax = plt.subplots()
                ax.boxplot(filtered_df[y_col].dropna())
                ax.set_title(f"Box Plot of {y_col}")
                ax.set_ylabel(y_col)
                st.pyplot(fig)
                st.session_state["latest_chart_fig"] = fig
                st.session_state["latest_chart_title"] = plot_type

        elif plot_type == "Scatter Plot":
            if len(numeric_columns) < 2:
                st.info("At least two numeric columns are required for scatter plot.")
            else:
                x_col = st.selectbox(
                    "Select X column",
                    options=numeric_columns,
                    key="scatter_x_col"
                )

                y_col = st.selectbox(
                    "Select Y column",
                    options=numeric_columns,
                    key="scatter_y_col"
                )

                group_col = st.selectbox(
                    "Optional color/group column",
                    options=["None"] + categorical_columns,
                    key="scatter_group_col"
                )

                fig, ax = plt.subplots()

                if group_col != "None":
                    for group_value, group_data in filtered_df.groupby(group_col):
                        ax.scatter(group_data[x_col], group_data[y_col], label=str(group_value))
                    ax.legend()
                else:
                    ax.scatter(filtered_df[x_col], filtered_df[y_col])

                ax.set_title(f"Scatter Plot: {x_col} vs {y_col}")
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                st.pyplot(fig)
                st.session_state["latest_chart_fig"] = fig
                st.session_state["latest_chart_title"] = plot_type

        elif plot_type == "Line Chart":
            if not all_columns or not numeric_columns:
                st.info("You need at least one x column and one numeric y column for line chart.")
            else:
                x_col = st.selectbox(
                    "Select X column (preferably datetime)",
                    options=all_columns,
                    key="line_x_col"
                )

                y_col = st.selectbox(
                    "Select Y column",
                    options=numeric_columns,
                    key="line_y_col"
                )

                group_col = st.selectbox(
                    "Optional group column",
                    options=["None"] + categorical_columns,
                    key="line_group_col"
                )

                aggregation = st.selectbox(
                    "Aggregation",
                    options=["sum", "mean", "count", "median"],
                    key="line_aggregation"
                )

                plot_df = filtered_df.copy()

                if aggregation == "count":
                    if group_col != "None":
                        grouped = plot_df.groupby([x_col, group_col], dropna=False).size().reset_index(name="value")
                    else:
                        grouped = plot_df.groupby(x_col, dropna=False).size().reset_index(name="value")
                else:
                    if group_col != "None":
                        grouped = (
                            plot_df.groupby([x_col, group_col], dropna=False)[y_col]
                            .agg(aggregation)
                            .reset_index(name="value")
                        )
                    else:
                        grouped = (
                            plot_df.groupby(x_col, dropna=False)[y_col]
                            .agg(aggregation)
                            .reset_index(name="value")
                        )

                fig, ax = plt.subplots()

                if group_col != "None":
                    for group_value, group_data in grouped.groupby(group_col):
                        sorted_data = group_data.sort_values(by=x_col)
                        ax.plot(sorted_data[x_col], sorted_data["value"], label=str(group_value))
                    ax.legend()
                else:
                    grouped = grouped.sort_values(by=x_col)
                    ax.plot(grouped[x_col], grouped["value"])

                ax.set_title(f"Line Chart of {y_col} by {x_col}")
                ax.set_xlabel(x_col)
                ax.set_ylabel("Value")
                plt.xticks(rotation=45)
                st.pyplot(fig)

                st.write("Aggregated data preview:")
                st.dataframe(grouped.head(20), width="stretch")
                st.session_state["latest_chart_fig"] = fig
                st.session_state["latest_chart_title"] = plot_type

        elif plot_type == "Bar Chart":
            if not all_columns:
                st.info("No columns available.")
            else:
                x_col = st.selectbox(
                    "Select X column",
                    options=all_columns,
                    key="bar_x_col"
                )

                possible_y_options = ["None"] + numeric_columns
                y_col_selected = st.selectbox(
                    "Select Y column (optional for count aggregation)",
                    options=possible_y_options,
                    key="bar_y_col"
                )
                y_col = None if y_col_selected == "None" else y_col_selected

                group_col = st.selectbox(
                    "Optional color/group column",
                    options=["None"] + categorical_columns,
                    key="bar_group_col"
                )

                aggregation = st.selectbox(
                    "Aggregation",
                    options=["count", "sum", "mean", "median"],
                    key="bar_aggregation"
                )

                top_n = st.number_input(
                    "Top N categories to show",
                    min_value=1,
                    max_value=100,
                    value=10,
                    step=1,
                    key="bar_top_n"
                )

                try:
                    grouped = build_bar_chart_data(
                        df=filtered_df,
                        x_col=x_col,
                        y_col=y_col,
                        group_col=None if group_col == "None" else group_col,
                        aggregation=aggregation,
                        top_n=top_n
                    )

                    fig, ax = plt.subplots()

                    if group_col != "None":
                        pivot_df = grouped.pivot(index=x_col, columns=group_col, values="value").fillna(0)
                        pivot_df.plot(kind="bar", ax=ax)
                    else:
                        ax.bar(grouped[x_col].astype(str), grouped["value"])

                    ax.set_title(f"Bar Chart of {x_col}")
                    ax.set_xlabel(x_col)
                    ax.set_ylabel("Value")
                    plt.xticks(rotation=45)
                    st.pyplot(fig)

                    st.write("Aggregated data preview:")
                    st.dataframe(grouped.head(20), width="stretch")
                    st.session_state["latest_chart_fig"] = fig
                    st.session_state["latest_chart_title"] = plot_type
                except Exception as e:
                    st.error(f"Error: {e}")
            
        elif plot_type == "Correlation Heatmap":
            if len(numeric_columns) < 2:
                st.info("At least two numeric columns are required for correlation heatmap.")
            else:
                selected_corr_columns = st.multiselect(
                    "Select numeric columns for correlation",
                    options=numeric_columns,
                    default=numeric_columns[:min(5, len(numeric_columns))],
                    key="corr_columns"
                )

                if len(selected_corr_columns) < 2:
                    st.warning("Please select at least two numeric columns.")
                else:
                    corr_df = filtered_df[selected_corr_columns].corr()

                    fig, ax = plt.subplots()
                    cax = ax.imshow(corr_df, aspect="auto")
                    ax.set_xticks(range(len(selected_corr_columns)))
                    ax.set_yticks(range(len(selected_corr_columns)))
                    ax.set_xticklabels(selected_corr_columns, rotation=45, ha="right")
                    ax.set_yticklabels(selected_corr_columns)
                    ax.set_title("Correlation Heatmap")
                    fig.colorbar(cax)
                    st.pyplot(fig)

                    st.write("Correlation matrix:")
                    st.dataframe(corr_df, width="stretch")
                    st.session_state["latest_chart_fig"] = fig
                    st.session_state["latest_chart_title"] = plot_type
# -----------------------------
# PAGE D — EXPORT & REPORT
# -----------------------------
# -----------------------------
# PAGE D — EXPORT & REPORT
# -----------------------------
elif page == "Export & Report":
    st.header("Export & Report")

    if st.session_state["working_df"] is None:
        st.warning("Please upload a dataset first.")
    else:
        df = st.session_state["working_df"]
        log_data = st.session_state["transformation_log"]

        st.subheader("Current Cleaned Dataset Preview")
        st.dataframe(df.head(10), width="stretch")

        col1, col2 = st.columns(2)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])

        st.markdown("---")
        st.subheader("Transformation Report")

        if log_data:
            log_df = pd.DataFrame(log_data)
            st.dataframe(log_df, width="stretch")
        else:
            st.info("No transformation steps recorded yet.")

        st.markdown("---")
        st.subheader("Download Cleaned Dataset")

        # CSV export
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Cleaned Dataset as CSV",
            data=csv_data,
            file_name="cleaned_dataset.csv",
            mime="text/csv"
        )

        # Excel export
        excel_data = dataframe_to_excel_bytes(df)
        st.download_button(
            label="Download Cleaned Dataset as Excel",
            data=excel_data,
            file_name="cleaned_dataset.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.markdown("---")
        st.subheader("Download Transformation Report")

        transformation_report_json = transformation_log_to_json(log_data)
        st.download_button(
            label="Download Transformation Report (JSON)",
            data=transformation_report_json,
            file_name="transformation_report.json",
            mime="application/json"
        )

        st.markdown("---")
        st.subheader("Download Recipe / Replay Output")

        recipe_json = build_recipe_json(log_data)
        st.download_button(
            label="Download JSON Recipe",
            data=recipe_json,
            file_name="transformation_recipe.json",
            mime="application/json"
        )

        python_script = build_python_replay_script(log_data)
        st.download_button(
            label="Download Python Replay Script",
            data=python_script,
            file_name="replay_pipeline.py",
            mime="text/x-python"
        )
        st.markdown("---")
        st.subheader("Download Dashboard Report")

        latest_chart_fig = st.session_state.get("latest_chart_fig", None)
        latest_chart_title = st.session_state.get("latest_chart_title", "Dashboard Chart")

        pdf_data = build_dashboard_pdf_bytes(
            df=df,
            log_data=log_data,
            chart_figure=latest_chart_fig,
            chart_title=latest_chart_title
        )

        st.download_button(
            label="Download Dashboard Report as PDF",
            data=pdf_data,
            file_name="dashboard_report.pdf",
            mime="application/pdf"
        )