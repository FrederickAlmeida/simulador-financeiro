
import json
import math
import uuid
from dataclasses import dataclass, asdict
from datetime import date
from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
    end: Optional[str] = None        # 'YYYY-MM' (inclusive). If provided, overrides 'months'.
    permanent: bool = False          # if True, ignore months and run to horizon end
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
    Supports permanent events and optional end date.
    Precedence: permanent > end (inclusive) > months
    """
    start_p = parse_year_month(ev.start)
    monthly_g = ev.growth_rate_pct / 100.0

    aligned = []

    def _apply_sign(v):
        return v if ev.kind == "income" else -v

    if ev.permanent:
        for p in horizon:
            if p < start_p:
                aligned.append(0.0)
            else:
                idx = (p.year - start_p.year) * 12 + (p.month - start_p.month)
                aligned.append(_apply_sign(ev.amount * ((1.0 + monthly_g) ** idx)))
        return pd.Series(aligned, index=[p.to_timestamp(how="end") for p in horizon], name=ev.name)

    # End date (inclusive) if provided
    if getattr(ev, "end", None):
        end_p = parse_year_month(ev.end, default=start_p)
        if end_p < start_p:
            end_p = start_p
        for p in horizon:
            if p < start_p or p > end_p:
                aligned.append(0.0)
            else:
                idx = (p.year - start_p.year) * 12 + (p.month - start_p.month)
                aligned.append(_apply_sign(ev.amount * ((1.0 + monthly_g) ** idx)))
        return pd.Series(aligned, index=[p.to_timestamp(how="end") for p in horizon], name=ev.name)

    # Fallback: fixed months
    ev_months = max(1, int(ev.months))
    ev_range = period_range(start_p, ev_months)

    values = {}
    for idx, p in enumerate(ev_range):
        amt = ev.amount * ((1.0 + monthly_g) ** idx)
        values[p] = amt

    for p in horizon:
        v = values.get(p, 0.0)
        aligned.append(_apply_sign(v))

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


def _signed_area_traces(x, y, base_name, line_style=None, pos_alpha=0.25, neg_alpha=0.25):
    """
    Build two traces for a series that changes sign:
    - Positive part: green line + filled area to zero
    - Negative part: red line + filled area to zero
    Returns a list of go.Scatter traces.
    """
    pos = [v if (v is not None and v >= 0) else None for v in y]
    neg = [v if (v is not None and v < 0) else None for v in y]

    pos_trace = go.Scatter(
        x=x, y=pos, name=base_name,
        mode="lines",
        line=dict(color="green"),
        fill="tozeroy",
        fillcolor=f"rgba(0,128,0,{pos_alpha})",
        legendgroup=base_name,
        showlegend=True,
        hovertemplate="%{x|%b %Y}<br>%{y:.2f}<extra>"+base_name+"</extra>",
    )
    if line_style:
        pos_trace.line["dash"] = line_style

    neg_trace = go.Scatter(
        x=x, y=neg, name=base_name,
        mode="lines",
        line=dict(color="red"),
        fill="tozeroy",
        fillcolor=f"rgba(214,39,40,{neg_alpha})",  # Plotly default red-ish
        legendgroup=base_name,
        showlegend=False,  # don't duplicate legend entries
        hovertemplate="%{x|%b %Y}<br>%{y:.2f}<extra>"+base_name+"</extra>",
    )
    if line_style:
        neg_trace.line["dash"] = line_style

    return [pos_trace, neg_trace]

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
    Event(id=str(uuid.uuid4()), name="Salary", kind="income", amount=5000.0, start=str(_this_month_period()), months=36, permanent=False, growth_rate_pct=0.0, notes="Monthly net salary"),
    Event(id=str(uuid.uuid4()), name="Rent",   kind="cost",   amount=1800.0, start=str(_this_month_period()), months=36, permanent=False, growth_rate_pct=0.0, notes="Fixed rent"),
    Event(id=str(uuid.uuid4()), name="Groceries", kind="cost", amount=900.0, start=str(_this_month_period()), months=36, permanent=False, growth_rate_pct=0.5, notes="~0.5% monthly inflation"),
    Event(id=str(uuid.uuid4()), name="Streaming", kind="cost", amount=45.0, start=str(_this_month_period()), months=24, permanent=False, growth_rate_pct=0.0, notes="Cancels after 24 months"),
]

if "events" not in st.session_state:
    st.session_state.events = [e.to_dict() for e in DEFAULT_EVENTS]


def events_df() -> pd.DataFrame:
    df = pd.DataFrame(st.session_state.events)
    # Ensure ordered columns
    cols = ["id", "name", "kind", "amount", "start", "months", "end", "permanent", "growth_rate_pct", "notes"]
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

    # add/clean end column (YYYY-MM as string, inclusive)
    if "end" not in df.columns:
        df["end"] = None
    def _clean_end(v):
        if v is None:
            return None
        s = str(v).strip()
        return s if len(s) >= 6 else None
    df["end"] = df["end"].apply(_clean_end)

    df["months"] = pd.to_numeric(df["months"], errors="coerce").fillna(1).astype(int).clip(lower=1)

    # permanent parsing (robust against strings like "true"/"false")
    if "permanent" not in df.columns:
        df["permanent"] = False
    def _parse_bool(v):
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return v != 0
        if isinstance(v, str):
            s = v.strip().lower()
            return s in ("true", "1", "yes", "y", "t")
        return False
    df["permanent"] = df["permanent"].apply(_parse_bool)

    df["growth_rate_pct"] = pd.to_numeric(df["growth_rate_pct"], errors="coerce").fillna(0.0)
    df["notes"] = df["notes"].fillna("").astype(str)

    st.session_state.events = df.to_dict(orient="records")


st.subheader("🧾 Eventos")

edited = st.data_editor(
    events_df(),
    num_rows="dynamic",
    key="events_editor",
    column_order=["name","kind","amount","start","permanent","months","end","growth_rate_pct","notes"],  # hide 'id' from view
    use_container_width=True,
    column_config={
        # 'id' is intentionally hidden by column_order but still present in the data frame to preserve identity
        "name": st.column_config.TextColumn("Name", help="Short label for this event"),
        "kind": st.column_config.SelectboxColumn("Kind", options=["income", "cost"], help="Income adds; Cost subtracts"),
        "amount": st.column_config.NumberColumn("Amount / month", step=0.01, format="%.2f"),
        "start": st.column_config.TextColumn("Start (YYYY-MM)", help="First month included, e.g., 2025-08"),
        "permanent": st.column_config.CheckboxColumn("Permanent", help="If checked, ignore months and run to horizon end"),
        "months": st.column_config.NumberColumn("Duration (months)", min_value=1, step=1),
        "end": st.column_config.TextColumn("End (YYYY-MM)", help="Inclusive end month; overrides Duration if set"),
        "growth_rate_pct": st.column_config.NumberColumn("Growth % / month", help="e.g., 0 for fixed, 1.5 means +1.5% per month", step=0.01, format="%.2f"),
        "notes": st.column_config.TextColumn("Notes", help="Optional"),
    },
)

c1, c2, c3, c4 = st.columns(4)
with c1:
    if st.button("💾 Atualizar", use_container_width=True):
        write_events_back(edited)
        st.success("Events saved in session.", icon="✅")
with c2:
    if st.button("➕ Adicionar novo evento", use_container_width=True):
        df = events_df()
        new = Event(
            id=str(uuid.uuid4()),
            name="Side Gig",
            kind="income",
            amount=800.0,
            start=str(_this_month_period() + 1),
            months=18,
            permanent=False,
            growth_rate_pct=0.0,
            notes="Freelance"
        )
        df = pd.concat([df, pd.DataFrame([new.to_dict()])], ignore_index=True)
        write_events_back(df)
        st.rerun()
with c3:
    if st.button("♻️ Resetar", use_container_width=True):
        st.session_state.events = [e.to_dict() for e in DEFAULT_EVENTS]
        st.rerun()
with c4:
    uploaded = st.file_uploader("📥 Import events (JSON)", type=["json"], label_visibility="collapsed")
    if uploaded is not None:
        try:
            data = json.load(uploaded)
            if isinstance(data, dict) and "events" in data:
                data = data["events"]
            assert isinstance(data, list)
            df = pd.DataFrame(data)
            write_events_back(df)
            st.success("Imported events.", icon="✅")
        except Exception as e:
            st.error(f"Failed to import: {e}")

# Export events
st.download_button(
    "⬇️ Baixar Eventos",
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

st.subheader("📊 Monthly Income, Costs & Net")
st.caption("Income and Costs are monthly totals (blue/red). Net is shaded: green when positive, red when negative.")

x = totals_df.index
income_y = totals_df["Income"].tolist()
costs_y = totals_df["Costs"].tolist()
net_y = totals_df["Net"].tolist()

fig1 = go.Figure()

# Income (blue line)
fig1.add_trace(go.Scatter(
    x=x, y=income_y, name="Income",
    mode="lines+markers",
    line=dict(color="blue"),
    hovertemplate="%{x|%b %Y}<br>%{y:.2f}<extra>Income</extra>"
))

# Costs (red line)
fig1.add_trace(go.Scatter(
    x=x, y=costs_y, name="Costs",
    mode="lines+markers",
    line=dict(color="red"),
    hovertemplate="%{x|%b %Y}<br>%{y:.2f}<extra>Costs</extra>"
))

# Net (signed area)
for tr in _signed_area_traces(x, net_y, "Net", line_style=None, pos_alpha=0.25, neg_alpha=0.25):
    fig1.add_trace(tr)

fig1.update_layout(
    legend_title_text="Series",
    hovermode="x unified",
    height=520,
    yaxis_tickformat=",.2f",
)

st.plotly_chart(fig1, use_container_width=True)

# ----------------- Accumulated chart -----------------
st.subheader("📈 Accumulated Balance")
st.caption("Shaded by sign (green=positive, red=negative). 'With interest' uses the annual rate and timing from the sidebar.")

fig2 = go.Figure()

# Accumulated (no interest)
if show_acc_no_int:
    acc_no_int_y = acc_no_int.tolist()
    for tr in _signed_area_traces(x, acc_no_int_y, "Accumulated (no interest)", line_style="solid", pos_alpha=0.20, neg_alpha=0.20):
        fig2.add_trace(tr)

# Accumulated (with interest)
if show_acc_with_int:
    acc_with_int_y = acc_with_int.tolist()
    line_style = "dash"  # visually distinguish from no-interest
    for tr in _signed_area_traces(x, acc_with_int_y, f"Accumulated (with interest, {annual_rate_pct:.2f}%/yr)", line_style=line_style, pos_alpha=0.20, neg_alpha=0.20):
        fig2.add_trace(tr)

fig2.update_layout(
    legend_title_text="Series",
    hovermode="x unified",
    height=520,
    yaxis_tickformat=",.2f",
)

st.plotly_chart(fig2, use_container_width=True)

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
