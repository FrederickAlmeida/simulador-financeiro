

import json
import math
import uuid
from dataclasses import dataclass, asdict
from datetime import date
from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# -------------------------------
# Helpers
# -------------------------------

def _this_month_period() -> pd.Period:
    today = date.today()
    return pd.Period(freq="M", year=today.year, month=today.month)


def parse_year_month(s: str, default: Optional[pd.Period] = None) -> pd.Period:
    """
    Parse strings like '2025-08' or '2025/08' into a pandas Period('M').
    Falls back to 'default' or current month.
    """
    if default is None:
        default = _this_month_period()
    if not isinstance(s, str) or len(s) < 6:
        return default
    s = s.strip().replace("/", "-")
    try:
        y, m = s.split("-")[:2]
        y = int(y)
        m = int(m)
        if 1 <= m <= 12:
            return pd.Period(freq="M", year=y, month=m)
    except Exception:
        pass
    return default


def period_range(start: pd.Period, months: int) -> List[pd.Period]:
    """Return [start, start+1M, ..., start+(months-1)M]"""
    return [start + i for i in range(months)]


def effective_monthly_rate_from_annual(annual_rate_pct: float) -> float:
    """
    Convert an annual return (in %) to effective monthly rate.
    Example: 12%/year -> (1.12)^(1/12) - 1 ≈ 0.9489% per month.
    """
    return (1.0 + annual_rate_pct / 100.0) ** (1.0 / 12.0) - 1.0


# -------------------------------
# Data model
# -------------------------------

@dataclass
class Event:
    id: str
    name: str
    kind: Literal["income", "cost"]  # income increases, cost decreases
    amount: float                    # base monthly amount
    start: str                       # 'YYYY-MM'
    months: int                      # duration in months (>=1)
    growth_rate_pct: float = 0.0     # monthly growth/decay, e.g., 1.5 = +1.5% per month
    notes: str = ""

    def to_dict(self):
        return asdict(self)


# -------------------------------
# Core calculations
# -------------------------------

def expand_event_to_series(ev: Event, horizon: List[pd.Period]) -> pd.Series:
    """
    Expand a single event into a monthly series aligned to 'horizon'.
    Applies monthly growth (compound) if growth_rate_pct != 0.
    """
    start_p = parse_year_month(ev.start)
    monthly_g = ev.growth_rate_pct / 100.0

    # Build the event's own period range
    ev_months = max(1, int(ev.months))
    ev_range = period_range(start_p, ev_months)

    values = {}
    for idx, p in enumerate(ev_range):
        amt = ev.amount * ((1.0 + monthly_g) ** idx)
        values[p] = amt

    # Align to the horizon; zero when outside
    aligned = []
    for p in horizon:
        v = values.get(p, 0.0)
        # Sign by kind
        if ev.kind == "income":
            aligned.append(v)
        else:  # cost
            aligned.append(-v)
    return pd.Series(aligned, index=[p.to_timestamp(how="end") for p in horizon], name=ev.name)


def combine_events(events: List[Event], start: pd.Period, months: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (by_event_df, totals_df)
      - by_event_df: each event as a separate column, indexed by month-end timestamps
      - totals_df: columns ['Income', 'Costs', 'Net'], indexed by month-end timestamps
    """
    horizon = period_range(start, months)
    series_list = []
    for ev in events:
        s = expand_event_to_series(ev, horizon)
        series_list.append(s)

    if series_list:
        by_event = pd.concat(series_list, axis=1)
    else:
        by_event = pd.DataFrame(index=[p.to_timestamp(how="end") for p in horizon])

    # Totals
    # Sum only positive columns for income, negative for costs
    income = by_event.clip(lower=0).sum(axis=1)
    costs = -by_event.clip(upper=0).sum(axis=1)  # make positive
    net = income - costs

    totals = pd.DataFrame({"Income": income, "Costs": costs, "Net": net})
    return by_event, totals


def accumulated_no_interest(net: pd.Series) -> pd.Series:
    return net.cumsum()


def accumulated_with_interest(
    net: pd.Series,
    annual_rate_pct: float,
    contribution_timing: Literal["end_of_month", "beginning_of_month"] = "end_of_month",
) -> pd.Series:
    """
    Compound a rolling balance using net monthly contributions.
    - If contribution_timing == 'end_of_month': balance = balance*(1+r) + net
    - If 'beginning_of_month': balance = (balance + net)*(1+r)
    """
    r = effective_monthly_rate_from_annual(annual_rate_pct)
    bal = 0.0
    out = []
    for v in net.to_list():
        if contribution_timing == "beginning_of_month":
            bal = (bal + v) * (1.0 + r)
        else:
            bal = bal * (1.0 + r) + v
        out.append(bal)
    return pd.Series(out, index=net.index)


# -------------------------------
# Streamlit UI
# -------------------------------

st.set_page_config(page_title="Financial Planner", layout="wide")

st.title("📈 Financial Planner — Incomes, Costs & Accumulated Balance")

with st.expander("About this tool", expanded=False):
    st.markdown(
        """
This local app lets you model monthly **incomes** and **costs**, visualize them on a timeline,
and compute your **accumulated balance** with optional investment returns.

**Highlights**
- Add multiple events (income or cost), fixed or with **monthly growth** (positive or negative).
- Choose a **start month** and **horizon** (number of months).
- Toggle **accumulated balance** with or without interest.
- Export/import your scenario as JSON and download the monthly results as CSV.
        """
    )

# --------- Sidebar: Global settings --------------------------------------------------
st.sidebar.header("⚙️ Global Settings")

# Start month and horizon
col_a, col_b = st.sidebar.columns(2)
with col_a:
    start_year = st.number_input("Start Year", min_value=1970, max_value=2100, value=_this_month_period().year)
with col_b:
    start_month = st.number_input("Start Month", min_value=1, max_value=12, value=_this_month_period().month)
start_period = pd.Period(freq="M", year=int(start_year), month=int(start_month))

horizon_months = st.sidebar.slider("Horizon (months)", min_value=3, max_value=240, value=36, step=1)

# Accumulated / interest options
show_acc_no_int = st.sidebar.checkbox("Show Accumulated (no interest)", value=True)
show_acc_with_int = st.sidebar.checkbox("Show Accumulated (with interest)", value=True)
annual_rate_pct = st.sidebar.number_input("Annual return (%)", min_value=-100.0, max_value=1000.0, value=6.0, step=0.10)
timing = st.sidebar.radio(
    "Contribution timing", options=("end_of_month", "beginning_of_month"), index=0, horizontal=False
)

# --------- Session state: Events -----------------------------------------------------
DEFAULT_EVENTS = [
    Event(id=str(uuid.uuid4()), name="Salary", kind="income", amount=5000.0, start=str(_this_month_period()), months=36, growth_rate_pct=0.0, notes="Monthly net salary"),
    Event(id=str(uuid.uuid4()), name="Rent",   kind="cost",   amount=1800.0, start=str(_this_month_period()), months=36, growth_rate_pct=0.0, notes="Fixed rent"),
    Event(id=str(uuid.uuid4()), name="Groceries", kind="cost", amount=900.0, start=str(_this_month_period()), months=36, growth_rate_pct=0.5, notes="~0.5% monthly inflation"),
    Event(id=str(uuid.uuid4()), name="Streaming", kind="cost", amount=45.0, start=str(_this_month_period()), months=24, growth_rate_pct=0.0, notes="Cancels after 24 months"),
]

if "events" not in st.session_state:
    st.session_state.events = [e.to_dict() for e in DEFAULT_EVENTS]


def events_df() -> pd.DataFrame:
    df = pd.DataFrame(st.session_state.events)
    # Ensure ordered columns
    cols = ["id", "name", "kind", "amount", "start", "months", "growth_rate_pct", "notes"]
    df = df.reindex(columns=cols)
    return df


def write_events_back(df: pd.DataFrame):
    # Ensure IDs exist and are strings
    df = df.copy()
    if "id" not in df.columns:
        df["id"] = [str(uuid.uuid4()) for _ in range(len(df))]
    # Fill missing IDs
    df.loc[df["id"].isna(), "id"] = [str(uuid.uuid4()) for _ in range(df["id"].isna().sum())]

    # Coerce types / sanitize
    df["name"] = df["name"].fillna("").astype(str)
    df["kind"] = df["kind"].fillna("income").astype(str)
    df["kind"] = df["kind"].where(df["kind"].isin(["income", "cost"]), "income")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    df["start"] = df["start"].fillna(str(_this_month_period())).astype(str)
    df["months"] = pd.to_numeric(df["months"], errors="coerce").fillna(1).astype(int).clip(lower=1)
    df["growth_rate_pct"] = pd.to_numeric(df["growth_rate_pct"], errors="coerce").fillna(0.0)
    df["notes"] = df["notes"].fillna("").astype(str)

    st.session_state.events = df.to_dict(orient="records")


st.subheader("🧾 Events")
st.caption("Tip: Growth can be **negative** for declines (e.g., -1%/mo). Dates use `YYYY-MM`.")

edited = st.data_editor(
    events_df(),
    num_rows="dynamic",
    key="events_editor",
    use_container_width=True,
    column_config={
        "id": st.column_config.TextColumn("ID", disabled=True),
        "name": st.column_config.TextColumn("Name", help="Short label for this event"),
        "kind": st.column_config.SelectboxColumn("Kind", options=["income", "cost"], help="Income adds; Cost subtracts"),
        "amount": st.column_config.NumberColumn("Amount / month", step=10.0, format="%.2f"),
        "start": st.column_config.TextColumn("Start (YYYY-MM)", help="First month included, e.g., 2025-08"),
        "months": st.column_config.NumberColumn("Duration (months)", min_value=1, step=1),
        "growth_rate_pct": st.column_config.NumberColumn("Growth % / month", help="e.g., 0 for fixed, 1.5 means +1.5% per month", step=0.1, format="%.2f"),
        "notes": st.column_config.TextColumn("Notes", help="Optional"),
    },
)

c1, c2, c3, c4 = st.columns(4)
with c1:
    if st.button("💾 Save edits", use_container_width=True):
        write_events_back(edited)
        st.success("Events saved in session.", icon="✅")
with c2:
    if st.button("➕ Add example income", use_container_width=True):
        df = events_df()
        new = Event(
            id=str(uuid.uuid4()),
            name="Side Gig",
            kind="income",
            amount=800.0,
            start=str(_this_month_period() + 1),
            months=18,
            growth_rate_pct=0.0,
            notes="Freelance"
        )
        df = pd.concat([df, pd.DataFrame([new.to_dict()])], ignore_index=True)
        write_events_back(df)
        st.experimental_rerun()
with c3:
    if st.button("♻️ Reset to defaults", use_container_width=True):
        st.session_state.events = [e.to_dict() for e in DEFAULT_EVENTS]
        st.experimental_rerun()
with c4:
    uploaded = st.file_uploader("📥 Import events (JSON)", type=["json"], label_visibility="collapsed")
    if uploaded is not None:
        try:
            data = json.load(uploaded)
            # accept list[dict] or {"events": [...]}
            if isinstance(data, dict) and "events" in data:
                data = data["events"]
            assert isinstance(data, list)
            # minimal validation
            df = pd.DataFrame(data)
            write_events_back(df)
            st.success("Imported events.", icon="✅")
        except Exception as e:
            st.error(f"Failed to import: {e}")

# Export events
st.download_button(
    "⬇️ Download events JSON",
    data=json.dumps({"events": st.session_state.events}, indent=2),
    file_name="events.json",
    mime="application/json",
    use_container_width=True,
)

st.divider()

# --------- Compute & Visualize -------------------------------------------------------
by_event_df, totals_df = combine_events(
    events=[Event(**e) for e in st.session_state.events],
    start=start_period,
    months=int(horizon_months),
)

# Accumulated
acc_no_int = accumulated_no_interest(totals_df["Net"]) if show_acc_no_int else None
acc_with_int = accumulated_with_interest(
    totals_df["Net"], annual_rate_pct=annual_rate_pct, contribution_timing=timing
) if show_acc_with_int else None

# Build plot
plot_df = totals_df.copy()
if show_acc_no_int:
    plot_df["Accumulated (no interest)"] = acc_no_int
if show_acc_with_int:
    suffix = "end" if timing == "end_of_month" else "begin"
    plot_df[f"Accumulated (with interest, {annual_rate_pct:.2f}%/yr, {suffix})"] = acc_with_int

# Melt for multi-line plot
long_df = plot_df.reset_index(names="Month").melt(id_vars="Month", var_name="Series", value_name="Amount")

st.subheader("📊 Timeline")
st.caption("Income and Costs are monthly totals; Accumulated lines reflect cumulative net (with optional compounding).")

fig = px.line(long_df, x="Month", y="Amount", color="Series", markers=True)
fig.update_layout(legend_title_text="Series", hovermode="x unified", height=520, yaxis_tickformat=",.2f")
st.plotly_chart(fig, use_container_width=True)

st.subheader("🗃️ Monthly Breakdown")
st.caption("Download as CSV for further analysis.")
full_table = plot_df.round(2)
st.dataframe(full_table, use_container_width=True)
st.download_button(
    "⬇️ Download monthly CSV",
    data=full_table.to_csv(index=True).encode("utf-8"),
    file_name="monthly_breakdown.csv",
    mime="text/csv",
    use_container_width=True,
)

st.subheader("📦 Event Contributions (by event)")
st.caption("Positive values are incomes; negative values are costs.")
st.dataframe(by_event_df.round(2), use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    "Built with ❤️ in Streamlit. Tip: Use **negative growth** for shrinking costs, or **positive growth** to model inflation."
)
