"""
U.S. Provisional Natality Exploration Dashboard (2025)
Integrated Single-File Streamlit Application
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI

# -----------------------------------------------------------------------------
# 1. CONSTANTS & LOOKUPS
# -----------------------------------------------------------------------------

MONTH_ORDER = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

STATE_TO_ABBR = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI",
    "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
    "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME",
    "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
    "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE",
    "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM",
    "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH",
    "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI",
    "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX",
    "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}

SEX_COLORS = {
    "Female": "#2b5c8f",
    "Male": "#d95f02",
}

# --- AI Data Assistant (chatbot) constants ---
# Secret names accepted for the API key (checked in this order)
API_KEY_SECRET_NAMES = ["GROQ_API_KEY", "GROK_API_KEY", "XAI_API_KEY", "LLM_API_KEY"]

# Provider is detected from the API key prefix
LLM_PROVIDERS = {
    "gsk_": {
        "name": "Groq",
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "openai/gpt-oss-120b",
        "fallback_models": [
            "openai/gpt-oss-20b",
            "qwen/qwen3.8-27b",
            "llama-3.3-70b-versatile",
        ],
    },
    "xai-": {
        "name": "xAI (Grok)",
        "base_url": "https://api.x.ai/v1",
        "default_model": "grok-3-mini",
        "fallback_models": [],
    },
}
DEFAULT_PROVIDER_PREFIX = "gsk_"

SUGGESTED_QUESTIONS = [
    "Which 3 states had the most births?",
    "Which month had the fewest births, and why might that be?",
    "What is the male-to-female ratio in the current selection?",
]

CHAT_HISTORY_LENGTH = 6  # number of most recent messages sent to the model
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 2000    # gpt-oss is a reasoning model and spends tokens "thinking"

MODEL_UNAVAILABLE_MARKERS = [
    "not found", "decommissioned", "deprecated", "does not exist",
    "not available", "model_not_found", "model_decommissioned",
]

# -----------------------------------------------------------------------------
# 2. PAGE CONFIGURATION
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="U.S. Provisional Natality Dashboard (2025)",
    page_icon="📊",
    layout="wide",
)

# -----------------------------------------------------------------------------
# 3. DATA LOADING & VALIDATION
# -----------------------------------------------------------------------------

def get_data_path() -> Path:
    """Resolve file path across root and data/ directories."""
    current_dir = Path(__file__).resolve().parent
    candidate_paths = [
        current_dir / "Provisional_Natality_2025_CDC.csv",
        current_dir / "data" / "Provisional_Natality_2025_CDC.csv",
        Path("Provisional_Natality_2025_CDC.csv"),
        Path("data/Provisional_Natality_2025_CDC.csv"),
    ]
    for path in candidate_paths:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Provisional_Natality_2025_CDC.csv not found in current folder or data/ folder."
    )


def validate_raw_data(df: pd.DataFrame) -> None:
    """Validate dataframe structure and data integrity."""
    required_cols = {
        "state_of_residence", "month", "month_code",
        "year_code", "sex_of_infant", "births"
    }
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if df.empty:
        raise ValueError("The dataset is empty.")

    if not pd.api.types.is_numeric_dtype(df["births"]):
        raise TypeError("Column 'births' must be numeric.")

    if (df["births"] < 0).any():
        raise ValueError("Column 'births' contains negative values.")


@st.cache_data(show_spinner="Loading CDC Natality Data...")
def load_and_preprocess_data() -> pd.DataFrame:
    """Load, clean, order categorical variables, and map state abbreviations."""
    file_path = get_data_path()
    df = pd.read_csv(file_path)
    validate_raw_data(df)

    # State postal code mapping
    df["state_abbr"] = df["state_of_residence"].map(STATE_TO_ABBR)

    # Clean and order chronological months
    df["month"] = df["month"].astype(str).str.strip()
    df["month"] = pd.Categorical(df["month"], categories=MONTH_ORDER, ordered=True)

    # Clean strings and enforce integer counts
    df["sex_of_infant"] = df["sex_of_infant"].astype(str).str.strip()
    df["births"] = df["births"].astype(int)

    return df

# -----------------------------------------------------------------------------
# 4. KPI METRIC COMPUTATIONS
# -----------------------------------------------------------------------------

def compute_kpis(filtered_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate summary figures from the active filtered slice."""
    if filtered_df.empty:
        return {
            "total_births": 0,
            "selected_geographies": 0,
            "avg_monthly_births": 0.0,
            "top_geography_name": "N/A",
            "top_geography_count": 0,
            "peak_month_name": "N/A",
            "peak_month_count": 0,
        }

    total_births = int(filtered_df["births"].sum())
    num_geos = int(filtered_df["state_of_residence"].nunique())
    num_months = max(1, int(filtered_df["month"].nunique()))
    avg_monthly_births = total_births / num_months

    # Top state by count
    geo_totals = (
        filtered_df.groupby("state_of_residence")["births"]
        .sum()
        .sort_values(ascending=False)
    )
    top_geo = geo_totals.index[0]
    top_geo_val = int(geo_totals.iloc[0])

    # Top month by count (respects categorical ordering)
    month_totals = (
        filtered_df.groupby("month", observed=False)["births"]
        .sum()
        .sort_values(ascending=False)
    )
    peak_month = str(month_totals.index[0])
    peak_month_val = int(month_totals.iloc[0])

    return {
        "total_births": total_births,
        "selected_geographies": num_geos,
        "avg_monthly_births": avg_monthly_births,
        "top_geography_name": top_geo,
        "top_geography_count": top_geo_val,
        "peak_month_name": peak_month,
        "peak_month_count": peak_month_val,
    }

# -----------------------------------------------------------------------------
# 5. VISUALIZATION GENERATORS
# -----------------------------------------------------------------------------

def plot_top_bottom_geographies(filtered_df: pd.DataFrame, top_n: int = 5) -> go.Figure:
    """Horizontal bar chart comparing highest and lowest volume states."""
    geo_agg = (
        filtered_df.groupby("state_of_residence")["births"]
        .sum()
        .reset_index()
        .sort_values("births", ascending=True)
    )

    if len(geo_agg) <= top_n * 2:
        chart_data = geo_agg.copy()
        chart_data["Group"] = "Selected Entities"
    else:
        bottoms = geo_agg.head(top_n).copy()
        bottoms["Group"] = f"Bottom {top_n}"
        tops = geo_agg.tail(top_n).copy()
        tops["Group"] = f"Top {top_n}"
        chart_data = pd.concat([bottoms, tops])

    fig = px.bar(
        chart_data,
        x="births",
        y="state_of_residence",
        color="Group",
        orientation="h",
        labels={"births": "Total Births", "state_of_residence": "State / Geography"},
        title=f"Highest and Lowest Birth Volumes (Top & Bottom {top_n})",
        color_discrete_map={
            f"Top {top_n}": "#2b5c8f",
            f"Bottom {top_n}": "#d95f02",
            "Selected Entities": "#2b5c8f",
        },
    )
    fig.update_layout(
        xaxis=dict(rangemode="tozero", tickformat=","),
        yaxis=dict(categoryorder="total ascending"),
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=30),
        legend_title_text="",
    )
    fig.update_traces(hovertemplate="<b>%{y}</b><br>Births: %{x:,.0f}<extra></extra>")
    return fig


def plot_macro_trendline(filtered_df: pd.DataFrame) -> go.Figure:
    """Aggregate monthly time-series line chart."""
    trend = (
        filtered_df.groupby("month", observed=False)["births"]
        .sum()
        .reset_index()
    )
    fig = px.line(
        trend,
        x="month",
        y="births",
        markers=True,
        labels={"month": "Month", "births": "Total Births"},
        title="Aggregate Monthly Birth Trend",
    )
    fig.update_traces(
        line=dict(color="#1f77b4", width=3),
        marker=dict(size=8),
        hovertemplate="Month: <b>%{x}</b><br>Births: %{y:,.0f}<extra></extra>",
    )
    fig.update_layout(
        yaxis=dict(rangemode="tozero", tickformat=","),
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=30),
    )
    return fig


def plot_choropleth_map(filtered_df: pd.DataFrame) -> go.Figure:
    """Interactive US Choropleth map with state abbreviations."""
    state_totals = (
        filtered_df.dropna(subset=["state_abbr"])
        .groupby(["state_of_residence", "state_abbr"])["births"]
        .sum()
        .reset_index()
    )
    fig = px.choropleth(
        state_totals,
        locations="state_abbr",
        locationmode="USA-states",
        color="births",
        scope="usa",
        color_continuous_scale="Blues",
        labels={"births": "Total Births"},
        hover_name="state_of_residence",
        title="Geographic Distribution of Provisional Births",
    )
    fig.update_traces(
        hovertemplate="<b>%{hovertext}</b> (%{location})<br>Births: %{z:,.0f}<extra></extra>"
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=40, b=0),
        coloraxis_colorbar=dict(title="Births", tickformat=","),
    )
    return fig


def plot_state_rankings(filtered_df: pd.DataFrame) -> go.Figure:
    """Full ranked horizontal bar chart of selected states."""
    geo_totals = (
        filtered_df.groupby("state_of_residence")["births"]
        .sum()
        .reset_index()
        .sort_values("births", ascending=True)
    )
    fig = px.bar(
        geo_totals,
        x="births",
        y="state_of_residence",
        orientation="h",
        labels={"births": "Total Births", "state_of_residence": "State / Geography"},
        title="Total Births by State (Ranked)",
    )
    fig.update_traces(
        marker_color="#2b5c8f",
        hovertemplate="<b>%{y}</b><br>Births: %{x:,.0f}<extra></extra>",
    )
    height = max(450, len(geo_totals) * 18)
    fig.update_layout(
        height=height,
        xaxis=dict(rangemode="tozero", tickformat=","),
        yaxis=dict(categoryorder="total ascending"),
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=30),
    )
    return fig


def plot_monthly_sex_comparison(filtered_df: pd.DataFrame) -> go.Figure:
    """Side-by-side grouped bar chart comparing monthly births by infant sex."""
    trend_sex = (
        filtered_df.groupby(["month", "sex_of_infant"], observed=False)["births"]
        .sum()
        .reset_index()
    )
    fig = px.bar(
        trend_sex,
        x="month",
        y="births",
        color="sex_of_infant",
        barmode="group",
        labels={"month": "Month", "births": "Births", "sex_of_infant": "Infant Sex"},
        color_discrete_map=SEX_COLORS,
        title="Monthly Birth Comparison by Infant Sex",
    )
    fig.update_traces(
        hovertemplate="Month: <b>%{x}</b><br>Sex: %{fullData.name}<br>Births: %{y:,.0f}<extra></extra>"
    )
    fig.update_layout(
        yaxis=dict(rangemode="tozero", tickformat=","),
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_state_month_heatmap(filtered_df: pd.DataFrame) -> go.Figure:
    """Seasonality cross-tabulation heatmap (State vs. Month)."""
    pivot = filtered_df.pivot_table(
        index="state_of_residence",
        columns="month",
        values="births",
        aggfunc="sum",
        fill_value=0,
        observed=False,
    )
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=True).index]

    fig = px.imshow(
        pivot,
        labels=dict(x="Month", y="State / Geography", color="Births"),
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        aspect="auto",
        color_continuous_scale="YlGnBu",
        title="Seasonality Heatmap: State vs. Month",
    )
    fig.update_traces(
        hovertemplate="State: <b>%{y}</b><br>Month: <b>%{x}</b><br>Births: %{z:,.0f}<extra></extra>"
    )
    height = max(500, len(pivot) * 16)
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=50, b=30),
        coloraxis_colorbar=dict(title="Births", tickformat=","),
    )
    return fig

# -----------------------------------------------------------------------------
# 6. AI DATA ASSISTANT (CHATBOT)
# -----------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a friendly data assistant for CDC/NCHS provisional 2025 U.S. natality (birth) data.

Rules:
- Answer ONLY from the DATA SUMMARY provided below. It reflects the user's current sidebar filters.
- Values are raw birth COUNTS, not birth or fertility rates. When comparing states, remind the user that population size drives the counts.
- The data is provisional and may be revised as late records are processed.
- If a question cannot be answered from the data (e.g., race, mother's age, other years, birth weight), say so clearly and suggest what additional data would be needed.
- Use thousands separators for numbers, double-check your arithmetic, and be concise.
"""


class AllModelsUnavailableError(Exception):
    """Raised when every configured model is retired, unknown or unavailable."""


def get_api_key() -> Optional[str]:
    """Return the first API key found in Streamlit secrets, or None."""
    for name in API_KEY_SECRET_NAMES:
        try:
            value = st.secrets.get(name)
        except Exception:
            # No secrets.toml / secrets not configured
            return None
        if value:
            return str(value).strip()
    return None


def get_secret(name: str) -> Optional[str]:
    """Safely read an optional secret value."""
    try:
        value = st.secrets.get(name)
    except Exception:
        return None
    return str(value).strip() if value else None


def get_provider_config(api_key: str) -> Dict[str, Any]:
    """Detect the LLM provider from the API key prefix."""
    for prefix, config in LLM_PROVIDERS.items():
        if api_key.startswith(prefix):
            return config
    return LLM_PROVIDERS[DEFAULT_PROVIDER_PREFIX]


def get_candidate_models(provider: Dict[str, Any]) -> List[str]:
    """Ordered, de-duplicated list of models to try."""
    candidates = [
        st.session_state.get("active_model"),
        get_secret("LLM_MODEL"),
        provider["default_model"],
        *provider["fallback_models"],
    ]
    ordered: List[str] = []
    for model in candidates:
        if model and model not in ordered:
            ordered.append(model)
    return ordered


def is_model_unavailable_error(err: Exception) -> bool:
    """True only if the error means the model is retired, unknown or unavailable."""
    status = getattr(err, "status_code", None)
    if status in (401, 403, 429):
        return False  # auth / rate-limit errors must never trigger the fallback
    parts = [str(err)]
    body = getattr(err, "body", None)
    if body:
        parts.append(str(body))
    code = getattr(err, "code", None)
    if code:
        parts.append(str(code))
    message = " ".join(parts).lower()
    return any(marker in message for marker in MODEL_UNAVAILABLE_MARKERS)


def build_data_context(filtered_df: pd.DataFrame) -> str:
    """Compact text summary of the filtered data (kept to ~1K tokens)."""
    lines: List[str] = []
    lines.append("ACTIVE FILTERS:")
    lines.append(f"- Infant sex: {st.session_state.get('selected_sex', 'All')}")
    lines.append(f"- Geographies selected: {filtered_df['state_of_residence'].nunique()}")
    lines.append(f"- Months selected: {filtered_df['month'].nunique()}")
    lines.append("")
    lines.append(f"TOTAL BIRTHS: {int(filtered_df['births'].sum()):,}")
    lines.append("")

    lines.append("BIRTHS BY SEX:")
    for sex, val in filtered_df.groupby("sex_of_infant")["births"].sum().items():
        lines.append(f"- {sex}: {int(val):,}")
    lines.append("")

    lines.append("BIRTHS BY MONTH:")
    for month, val in filtered_df.groupby("month", observed=True)["births"].sum().items():
        lines.append(f"- {month}: {int(val):,}")
    lines.append("")

    state_sex = filtered_df.pivot_table(
        index="state_of_residence",
        columns="sex_of_infant",
        values="births",
        aggfunc="sum",
        fill_value=0,
    )
    for col in ("Female", "Male"):
        if col not in state_sex.columns:
            state_sex[col] = 0
    state_sex["Total"] = state_sex["Female"] + state_sex["Male"]
    state_sex = state_sex.sort_values("Total", ascending=False)

    lines.append("BIRTHS BY STATE (ranked high to low) — State: Total | Female | Male")
    for rank, (state, row) in enumerate(state_sex.iterrows(), start=1):
        lines.append(
            f"{rank}. {state}: {int(row['Total']):,} | {int(row['Female']):,} | {int(row['Male']):,}"
        )
    return "\n".join(lines)


def create_completion_stream(client: OpenAI, provider: Dict[str, Any], messages: List[Dict[str, str]]):
    """Start a streaming completion, falling back through models if unavailable."""
    for model in get_candidate_models(provider):
        params: Dict[str, Any] = dict(
            model=model,
            messages=messages,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            stream=True,
        )
        if "gpt-oss" in model:
            params["reasoning_effort"] = "low"
        try:
            stream = client.chat.completions.create(**params)
        except Exception as err:
            if is_model_unavailable_error(err):
                if st.session_state.get("active_model") == model:
                    st.session_state.pop("active_model", None)
                continue  # try the next model
            raise  # any other error is handled by the caller
        st.session_state["active_model"] = model
        return stream
    raise AllModelsUnavailableError()


def stream_text(stream):
    """Yield only the text content from streaming chunks."""
    for chunk in stream:
        if not getattr(chunk, "choices", None):
            continue
        content = chunk.choices[0].delta.content
        if content:
            yield content


def friendly_error_message(err: Exception) -> str:
    """Translate API errors into user-friendly messages."""
    if isinstance(err, AllModelsUnavailableError):
        return (
            "None of the configured AI models are available. Add a LLM_MODEL secret "
            "in Streamlit Cloud (⋮ → Settings → Secrets) with a model name currently "
            "offered by your provider, e.g. LLM_MODEL = \"openai/gpt-oss-20b\"."
        )
    status = getattr(err, "status_code", None)
    text = str(err).lower()
    if status == 401 or "invalid api key" in text or "invalid_api_key" in text:
        return "The API key was rejected. Please check the key saved in your Streamlit Secrets."
    if status == 429 or "rate limit" in text or "rate_limit" in text:
        return "The free-tier rate limit was reached. Please wait a minute and try again."
    return f"Sorry, something went wrong contacting the AI service: {err}"


def render_chatbot(filtered_df: pd.DataFrame) -> None:
    """AI chatbot tab: ask questions about the currently filtered data."""
    st.subheader("Ask the Data Assistant")

    api_key = get_api_key()
    if not api_key:
        st.warning(
            "**AI assistant not configured.** No API key was found.\n\n"
            "To enable it in Streamlit Community Cloud: open your app, click **⋮ → Settings → Secrets**, "
            "add the line below, and click **Save**:\n\n"
            "```toml\nGROQ_API_KEY = \"gsk_your_key_here\"\n```\n\n"
            "The app will restart automatically. The rest of the dashboard works without a key."
        )
        return

    provider = get_provider_config(api_key)
    active_model = get_candidate_models(provider)[0]
    st.caption(
        f"Powered by {provider['name']} · model {active_model}. "
        "Answers are based on the data matching your current sidebar filters. "
        "AI can make mistakes — verify key numbers with the charts."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Suggested questions + clear button
    pending_question: Optional[str] = None
    btn_cols = st.columns(len(SUGGESTED_QUESTIONS) + 1)
    for i, question in enumerate(SUGGESTED_QUESTIONS):
        if btn_cols[i].button(question, key=f"suggested_q_{i}", width="stretch"):
            pending_question = question
    if btn_cols[-1].button("🗑️ Clear chat", key="clear_chat", width="stretch"):
        st.session_state.messages = []
        st.rerun()

    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    typed_question = st.chat_input("Ask a question about the births data…")
    question = typed_question or pending_question
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    system_message = (
        SYSTEM_PROMPT
        + "\nDATA SUMMARY (current sidebar filters):\n"
        + build_data_context(filtered_df)
    )
    api_messages = [{"role": "system", "content": system_message}] + [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[-CHAT_HISTORY_LENGTH:]
    ]

    with st.chat_message("assistant"):
        try:
            client = OpenAI(api_key=api_key, base_url=provider["base_url"])
            with st.spinner("Thinking…"):
                stream = create_completion_stream(client, provider, api_messages)
            reply = st.write_stream(stream_text(stream))
            if isinstance(reply, list):
                reply = "".join(str(part) for part in reply)
            if not reply or not str(reply).strip():
                reply = "The model returned an empty answer. Please try rephrasing your question."
                st.markdown(reply)
        except Exception as err:
            reply = friendly_error_message(err)
            st.error(reply)

    st.session_state.messages.append({"role": "assistant", "content": str(reply)})

# -----------------------------------------------------------------------------
# 7. MAIN APPLICATION EXECUTION
# -----------------------------------------------------------------------------

def main():
    try:
        df_raw = load_and_preprocess_data()
    except Exception as exc:
        st.error(f"Error loading dataset: {exc}")
        st.stop()

    all_states = sorted(df_raw["state_of_residence"].unique().tolist())
    all_months = MONTH_ORDER
    sex_options = ["All", "Female", "Male"]

    # Filter State Callbacks
    if "selected_states" not in st.session_state:
        st.session_state.selected_states = all_states
    if "selected_months" not in st.session_state:
        st.session_state.selected_months = all_months
    if "selected_sex" not in st.session_state:
        st.session_state.selected_sex = "All"

    def reset_filters():
        st.session_state.selected_states = all_states
        st.session_state.selected_months = all_months
        st.session_state.selected_sex = "All"

    def select_all_states():
        st.session_state.selected_states = all_states

    def select_all_months():
        st.session_state.selected_months = all_months

    # Sidebar
    st.sidebar.header("Filter Controls")

    st.sidebar.selectbox("Infant Sex", options=sex_options, key="selected_sex")

    col_s_btn, _ = st.sidebar.columns([1, 1])
    with col_s_btn:
        st.button("Select All States", on_click=select_all_states, width="stretch")

    st.sidebar.multiselect(
        "State / Geography",
        options=all_states,
        key="selected_states",
        help="Select one or multiple geographies.",
    )

    col_m_btn, _ = st.sidebar.columns([1, 1])
    with col_m_btn:
        st.button("Select All Months", on_click=select_all_months, width="stretch")

    st.sidebar.multiselect(
        "Month (Chronological)",
        options=all_months,
        key="selected_months",
        help="Select calendar months.",
    )

    st.sidebar.markdown("---")
    st.sidebar.button("Reset All Filters", on_click=reset_filters, width="stretch")

    st.sidebar.markdown("### Active Filters Summary")
    st.sidebar.caption(f"• **Sex:** {st.session_state.selected_sex}")
    st.sidebar.caption(f"• **Geographies:** {len(st.session_state.selected_states)} of {len(all_states)} selected")
    st.sidebar.caption(f"• **Months:** {len(st.session_state.selected_months)} of {len(all_months)} selected")

    # Header & Context
    st.title("U.S. Provisional Natality Exploration Dashboard (2025)")
    st.markdown(
        "Designed for exploratory data analysis of geographic, monthly, and infant-sex patterns "
        "using CDC vital statistics."
    )

    st.info(
        "**Source & Methodology Notice:**\n\n"
        "- **Data Source:** Centers for Disease Control and Prevention (CDC) National Center for Health Statistics (NCHS).\n"
        "- **Provisional Status:** All counts shown are provisional and subject to reporting revisions and registration delays.\n"
        "- **Metric Definition:** Values represent raw **birth counts**, not birth or fertility rates. "
        "High volumes reflect both birth propensity and underlying state population size."
    )

    # Filter Application
    filtered_df = df_raw.copy()
    if st.session_state.selected_sex != "All":
        filtered_df = filtered_df[filtered_df["sex_of_infant"] == st.session_state.selected_sex]

    filtered_df = filtered_df[
        (filtered_df["state_of_residence"].isin(st.session_state.selected_states)) &
        (filtered_df["month"].isin(st.session_state.selected_months))
    ]

    if filtered_df.empty:
        st.warning("⚠️ No observations match your current filter selections. Please expand your filter criteria in the sidebar.")
        st.stop()

    # Dynamic KPI Cards
    kpis = compute_kpis(filtered_df)
    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
    kpi_col1.metric("Total Births", f"{kpis['total_births']:,}")
    kpi_col2.metric("Selected Geographies", f"{kpis['selected_geographies']}")
    kpi_col3.metric("Avg Births / Month", f"{kpis['avg_monthly_births']:,.0f}")
    kpi_col4.metric("Top Geography", kpis["top_geography_name"], f"{kpis['top_geography_count']:,} births", delta_color="off")
    kpi_col5.metric("Peak Month", kpis["peak_month_name"], f"{kpis['peak_month_count']:,} births", delta_color="off")

    st.markdown("---")

    # Tabs
    tab_overview, tab_geo, tab_monthly_sex, tab_ai, tab_table, tab_about = st.tabs([
        "Overview",
        "Geographic Analysis",
        "Monthly & Sex Analysis",
        "🤖 Ask the Data (AI)",
        "Data Table & Download",
        "About the Data",
    ])

    with tab_overview:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.plotly_chart(plot_top_bottom_geographies(filtered_df, top_n=5), width="stretch")
        with c2:
            st.plotly_chart(plot_macro_trendline(filtered_df), width="stretch")

    with tab_geo:
        st.subheader("Geographic Distribution")
        st.plotly_chart(plot_choropleth_map(filtered_df), width="stretch")
        st.markdown("#### State Volume Rankings")
        st.plotly_chart(plot_state_rankings(filtered_df), width="stretch")

    with tab_monthly_sex:
        st.subheader("Monthly Seasonality & Sex Breakdown")
        st.plotly_chart(plot_monthly_sex_comparison(filtered_df), width="stretch")
        st.markdown("#### Geographic Seasonality Matrix")
        st.plotly_chart(plot_state_month_heatmap(filtered_df), width="stretch")

    with tab_ai:
        render_chatbot(filtered_df)

    with tab_table:
        st.subheader("Searchable Filtered Records")
        display_df = filtered_df[[
            "state_of_residence", "month", "sex_of_infant", "births"
        ]].rename(columns={
            "state_of_residence": "State",
            "month": "Month",
            "sex_of_infant": "Infant Sex",
            "births": "Birth Count",
        })
        st.dataframe(
            display_df.style.format({"Birth Count": "{:,}"}),
            width="stretch",
            hide_index=True,
        )
        csv_buffer = display_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv_buffer,
            file_name="filtered_provisional_natality_2025.csv",
            mime="text/csv",
        )

    with tab_about:
        st.subheader("Data Documentation & Analytics Guidance")
        st.markdown(
            """
            ### Background and Provenance
            This dataset originates from the **Centers for Disease Control and Prevention (CDC)** National Vital Statistics System (NVSS).
            The records document provisional monthly live birth counts categorized by maternal state of residence and infant sex for the year 2025.

            ### Critical Analytical Notes for Students
            1. **Counts vs. Rates:**
               * The figures presented are raw birth counts ($N$).
               * Larger values in states such as California, Texas, and Florida primarily reflect base population rather than higher birth rates.
               * To calculate standardized birth rates in deeper analytics exercises, join these counts with U.S. Census Bureau population estimates:
                 $$\\text{Crude Birth Rate} = \\frac{\\text{Total Births}}{\\text{Total Population}} \\times 1{,}000$$
            2. **Provisional Data Considerations:**
               * Provisional data files reflect ongoing vital record reporting.
               * Counts for the most recent reporting months are subject to upward revisions as late certificates are processed.
            3. **Sex Ratio at Birth:**
               * Across large demographic samples, the natural human secondary sex ratio at birth typically hovers around 105 male births per 100 female births (~51.2% male).
               * Students can test for statistical deviations from this ratio across states using chi-squared goodness-of-fit tests.
            """
        )

if __name__ == "__main__":
    main()
