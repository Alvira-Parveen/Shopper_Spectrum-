import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
from sklearn.preprocessing import StandardScaler
import os

# ─── Page Configuration ────────────────────────────────────────────
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Color Palette ─────────────────────────────────────────────────
CHART_COLORS = [
    "#6366f1", "#06b6d4", "#10b981", "#f59e0b",
    "#f43f5e", "#8b5cf6", "#14b8a6", "#ec4899",
]
SEGMENT_META = {
    "High-Value":  dict(color="#10b981", icon="◆", bg="rgba(16,185,129,0.08)",  border="#10b981"),
    "Regular":     dict(color="#6366f1", icon="●", bg="rgba(99,102,241,0.08)",  border="#6366f1"),
    "Occasional":  dict(color="#f59e0b", icon="▲", bg="rgba(245,158,11,0.08)",  border="#f59e0b"),
    "At-Risk":     dict(color="#ef4444", icon="■", bg="rgba(239,68,68,0.08)",   border="#ef4444"),
}
SEGMENT_DESCRIPTIONS = {
    "High-Value": (
        "These are your best customers — they buy often, spend a lot, and were "
        "active recently. Don't take them for granted; a loyalty program or "
        "early access to new arrivals goes a long way."
    ),
    "Regular": (
        "Reliable, steady buyers. They won't break records, but they keep the "
        "lights on. Periodic promotions and a 'thank you' email every now and "
        "then can turn them into high-value shoppers."
    ),
    "Occasional": (
        "They pop in now and then — maybe around holidays or sales. A well-timed "
        "reminder or a personalised discount could nudge them towards buying more often."
    ),
    "At-Risk": (
        "Haven't bought in a while, and their history is thin. They're drifting. "
        "A 'we miss you' campaign with a compelling offer is your best shot "
        "before they're gone for good."
    ),
}

SEGMENT_PLAYBOOK = {
    "High-Value": [
        "Launch a VIP or loyalty tier with early access perks",
        "Send handwritten thank-you notes with orders (seriously, it works)",
        "Offer free shipping or a birthday discount automatically",
        "Ask for product reviews — they're your best advocates",
    ],
    "Regular": [
        "Set up a points-based rewards system to encourage repeat purchases",
        "Send 'back in stock' and 'you might like' emails based on past buys",
        "Offer bundle deals on categories they already shop",
        "Invite them to refer friends for mutual discounts",
    ],
    "Occasional": [
        "Trigger a personalised discount after 30 days of inactivity",
        "Highlight bestsellers and trending products in emails",
        "Use retargeting ads featuring items they've browsed",
        "Create urgency with limited-time offers around their purchase anniversaries",
    ],
    "At-Risk": [
        "Send a 'We miss you' email with a meaningful discount (15-20%)",
        "Run a win-back campaign — feature what's new since their last visit",
        "Survey them: ask why they stopped buying (keep it short, 2 questions max)",
        "If they don't respond in 60 days, consider sunsetting to save email reputation",
    ],
}


# ─── Data & Model Loading (cached) ────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_data(show_spinner="Crunching 400k+ transactions …")
def load_data():
    path = os.path.join(BASE_DIR, "..", "data", "online_retail.csv")
    raw = pd.read_csv(path)
    raw["InvoiceDate"] = pd.to_datetime(raw["InvoiceDate"])
    raw = raw.dropna(subset=["CustomerID", "Description"])
    raw = raw[~raw["InvoiceNo"].astype(str).str.startswith("C")]
    raw = raw[(raw["Quantity"] > 0) & (raw["UnitPrice"] > 0)]
    raw["TotalPrice"] = raw["Quantity"] * raw["UnitPrice"]
    raw["CustomerID"] = raw["CustomerID"].astype(int)
    raw["YearMonth"] = raw["InvoiceDate"].dt.to_period("M").astype(str)
    raw["Hour"] = raw["InvoiceDate"].dt.hour
    raw["DayOfWeek"] = raw["InvoiceDate"].dt.day_name()
    return raw


@st.cache_resource(show_spinner="Loading ML models …")
def load_models():
    model_dir = os.path.join(BASE_DIR, "..", "models")
    km = joblib.load(os.path.join(model_dir, "customer_segmentation_model.pkl"))
    sim = joblib.load(os.path.join(model_dir, "product_similarity.pkl"))
    sim.index = sim.index.str.strip()
    sim.columns = sim.columns.str.strip()
    return km, sim


@st.cache_data(show_spinner="Computing RFM features …")
def compute_rfm(_df):
    ref = _df["InvoiceDate"].max() + pd.Timedelta(days=1)
    return (
        _df.groupby("CustomerID")
        .agg(
            Recency=("InvoiceDate", lambda x: (ref - x.max()).days),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("TotalPrice", "sum"),
        )
        .reset_index()
    )


@st.cache_resource(show_spinner=False)
def build_scaler(_rfm):
    sc = StandardScaler()
    sc.fit(np.log1p(_rfm[["Recency", "Frequency", "Monetary"]]))
    return sc


def _map_clusters(km, sc):
    centroids_real = np.expm1(sc.inverse_transform(km.cluster_centers_))
    cdf = pd.DataFrame(centroids_real, columns=["R", "F", "M"])
    cdf["score"] = -cdf["R"].rank() + cdf["F"].rank() + cdf["M"].rank()
    rank = cdf["score"].rank(ascending=True).astype(int)
    names = {1: "At-Risk", 2: "Occasional", 3: "Regular", 4: "High-Value"}
    return {i: names[rank[i]] for i in range(len(rank))}


# Initialise
df = load_data()
kmeans, similarity_df = load_models()
rfm = compute_rfm(df)
scaler = build_scaler(rfm)
segment_labels = _map_clusters(kmeans, scaler)


# ─── Plotly helper ─────────────────────────────────────────────────
def _layout(**kw):
    base = dict(
        template="plotly_white",
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, sans-serif", color="#334155"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=40, b=0),
        colorway=CHART_COLORS,
        hoverlabel=dict(bgcolor="#1e293b", font_color="#e2e8f0", font_size=12),
    )
    base.update(kw)
    return base


def create_sparkline(data, color):
    fig = go.Figure(go.Scatter(
        y=data, mode="lines", fill="tozeroy",
        fillcolor=f"rgba({int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)},0.15)",
        line=dict(color=color, width=2, shape="spline"),
        hoverinfo="skip"
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=30, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        showlegend=False
    )
    return fig


# ─── Custom CSS ────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
.main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1200px; }
#MainMenu, footer, header { visibility: hidden; }

/* sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span,
[data-testid="stSidebar"] label {
    color: #cbd5e1 !important;
}
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.08); }

/* animations */
@keyframes fadeInUp { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
@keyframes bgShift { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
@keyframes shimmer { 0% { background-position: -200% center; } 100% { background-position: 200% center; } }

.main .block-container > div { animation: fadeInUp 0.6s ease-out forwards; }

/* hero */
.page-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #312e81 100%);
    background-size: 200% 200%;
    animation: bgShift 15s ease infinite;
    border-radius: 16px; padding: 36px 44px; margin-bottom: 1.75rem;
}
.page-header::after {
    content: ''; position: absolute; top: -60%; right: -8%;
    width: 340px; height: 340px;
    background: radial-gradient(circle, rgba(99,102,241,0.18) 0%, transparent 70%);
    border-radius: 50%;
}
.page-header h1 {
    color: #fff; font-size: 1.8rem; font-weight: 800;
}
.page-header p {
    color: #94a3b8; font-size: 0.88rem; margin: 0;
}

/* KPIs */
.kpi-row { display: flex; gap: 16px; margin-bottom: 1.5rem; }
.kpi-card {
    flex: 1; background: #fff; border: 1px solid #e2e8f0;
    transition: transform 0.2s, box-shadow 0.2s;
}
.kpi-card:hover { transform: translateY(-4px); box-shadow: 0 8px 24px rgba(0,0,0,0.06); }
.kpi-card::before { content: ''; position: absolute; inset: 0 0 auto 0; height: 3px; }
.kpi-card:nth-child(1)::before { background: #6366f1; }
.kpi-card:nth-child(2)::before { background: #06b6d4; }
.kpi-card:nth-child(3)::before { background: #10b981; }
.kpi-card:nth-child(4)::before { background: #f59e0b; }
.kpi-label {
    font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.06em; color: #64748b; margin-bottom: 4px;
}
.kpi-value { font-size: 1.75rem; font-weight: 800; color: #0f172a; line-height: 1.2; }
.kpi-sub   { font-size: 0.72rem; color: #94a3b8; margin-top: 4px; }

/* section title */
.section-title {
    font-size: 0.92rem; font-weight: 700; color: #334155;
    margin: 1.75rem 0 0.9rem; padding-bottom: 8px;
    border-bottom: 2px solid transparent; border-image: linear-gradient(to right, #e2e8f0, transparent) 1;
}

/* insight callout */
.insight-box {
    background: #fffbeb; border-left: 4px solid #f59e0b;
    border-radius: 0 10px 10px 0; padding: 14px 20px; margin: 1rem 0 1.5rem;
    font-size: 0.82rem; color: #78350f; line-height: 1.55;
}
.insight-box strong { color: #92400e; }

/* recommendation cards */
.rec-card {
    background: #fff; border: 1px solid #e2e8f0; border-radius: 12px;
    padding: 14px 20px; margin-bottom: 10px;
    display: flex; align-items: center; gap: 16px;
    transition: border-color 0.2s, box-shadow 0.2s, transform 0.2s;
}
.rec-card:hover {
    border-color: #6366f1;
    box-shadow: 0 4px 16px rgba(99,102,241,0.10);
    transform: translateY(-2px);
}
.rec-rank {
    width: 34px; height: 34px; background: #6366f1; color: #fff;
    border-radius: 8px; display: flex; align-items: center;
    justify-content: center; font-weight: 700; font-size: 0.85rem; flex-shrink: 0;
}
.rec-name { font-weight: 500; color: #1e293b; flex: 1; font-size: 0.88rem; }
.rec-score-wrap { width: 130px; flex-shrink: 0; }
.rec-score-lbl { font-size: 0.68rem; color: #94a3b8; margin-bottom: 4px; text-align: right; }
.rec-bar { height: 6px; background: #e2e8f0; border-radius: 3px; overflow: hidden; }
.rec-fill { height: 100%; background: linear-gradient(90deg,#6366f1,#8b5cf6); border-radius: 3px; }

/* product stat pills */
.prod-stats {
    display: flex; gap: 12px; margin: 1rem 0 0.5rem; flex-wrap: wrap;
}
.prod-pill {
    background: #f1f5f9; border-radius: 20px; padding: 6px 16px;
    font-size: 0.78rem; color: #475569;
}
.prod-pill b { color: #1e293b; }

/* quick picks */
.quick-picks { margin-bottom: 1rem; }
.quick-picks p {
    font-size: 0.78rem; color: #64748b; margin-bottom: 6px;
}

/* segment badge */
.seg-result {
    text-align: center; padding: 36px 24px; border-radius: 16px; margin: 1rem 0;
    backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px);
}
.seg-label {
    font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.1em; margin-bottom: 8px;
}
.seg-name { font-size: 2rem; font-weight: 800; margin-bottom: 10px; }
.seg-desc {
    font-size: 0.82rem; opacity: 0.85; max-width: 400px; margin: 0 auto;
    line-height: 1.55;
}

/* playbook */
.playbook {
    background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px;
    padding: 20px 24px; margin-top: 1rem;
}
.playbook h4 {
    font-size: 0.82rem; font-weight: 700; color: #334155;
    margin: 0 0 12px; text-transform: uppercase; letter-spacing: 0.04em;
}
.playbook ul { margin: 0; padding-left: 20px; }
.playbook li {
    font-size: 0.82rem; color: #475569; margin-bottom: 8px; line-height: 1.5;
}

/* preset buttons */
.preset-row { display: flex; gap: 8px; flex-wrap: wrap; margin: 0.8rem 0 1.2rem; }

/* sidebar brand */
.sb-brand {
    padding: 0.8rem 0 1.4rem; border-bottom: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 1.2rem;
}
.sb-brand h2 {
    font-size: 1.15rem; font-weight: 800; margin: 0;
    background: linear-gradient(90deg, #818cf8 0%, #c084fc 40%, #e879f9 60%, #c084fc 80%, #818cf8 100%);
    background-size: 200% auto;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    animation: shimmer 4s linear infinite;
}
.sb-brand p {
    font-size: 0.65rem; text-transform: uppercase;
    letter-spacing: 0.12em; color: #64748b !important; margin: 3px 0 0;
}

/* footer */
.app-footer {
    text-align: center; padding: 2rem 0 0.5rem; color: #94a3b8;
    font-size: 0.72rem; border-top: 1px solid #e2e8f0; margin-top: 3rem;
    line-height: 1.6;
}
.app-footer a { color: #6366f1; text-decoration: none; }

[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
</style>
""",
    unsafe_allow_html=True,
)

# ─── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="sb-brand"><h2>◆ Shopper Spectrum</h2>'
        '<p>Customer Intelligence</p></div>',
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigate",
        ["Overview", "Customer Deep-Dive", "Product Recommendations", "Customer Segmentation", "Data Explorer"],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption("RFM Analysis · K-Means · Collaborative Filtering")
    st.caption(f"Dataset: {len(df):,} clean records · Dec 2022 – Dec 2023")


# ═══════════════════════════════════════════════════════════════════
#  OVERVIEW
# ═══════════════════════════════════════════════════════════════════
if page == "Overview":

    st.markdown(
        '<div class="page-header">'
        "<h1>Dashboard Overview</h1>"
        "<p>A bird's-eye view of what's happening across the store — revenue, "
        "customers, products, and the segments they fall into.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # --- KPIs ---
    total_rev = df["TotalPrice"].sum()
    n_cust = df["CustomerID"].nunique()
    n_prod = df["Description"].nunique()
    n_countries = df["Country"].nunique()
    n_txn = df["InvoiceNo"].nunique()
    avg_ord = df.groupby("InvoiceNo")["TotalPrice"].sum().mean()

    monthly = (
        df.groupby("YearMonth")
        .agg(
            Revenue=("TotalPrice", "sum"), 
            Orders=("InvoiceNo", "nunique"),
            Customers=("CustomerID", "nunique"),
            Products=("Description", "nunique")
        )
        .reset_index()
        .sort_values("YearMonth")
    )

    uk_rev = df[df["Country"] == "United Kingdom"]["TotalPrice"].sum()
    uk_pct = uk_rev / total_rev * 100
    best_month = monthly.loc[monthly["Revenue"].idxmax()]

    # Sparkline data
    rev_spark = monthly["Revenue"].tolist()
    cust_spark = monthly["Customers"].tolist()
    prod_spark = monthly["Products"].tolist()
    order_spark = monthly["Orders"].tolist()

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        with st.container(border=True):
            st.markdown(f'''
            <div style="padding-bottom:5px;">
                <div class="kpi-label">Total Revenue</div>
                <div class="kpi-value">£{total_rev:,.0f}</div>
                <div class="kpi-sub" style="margin-bottom:0px;">{n_txn:,} orders over 13 months</div>
            </div>''', unsafe_allow_html=True)
            st.plotly_chart(create_sparkline(rev_spark, "#6366f1"), use_container_width=True, key="s1", config={"displayModeBar": False})

    with kpi2:
        with st.container(border=True):
            st.markdown(f'''
            <div style="padding-bottom:5px;">
                <div class="kpi-label">Customers</div>
                <div class="kpi-value">{n_cust:,}</div>
                <div class="kpi-sub" style="margin-bottom:0px;">Avg basket £{avg_ord:,.0f}</div>
            </div>''', unsafe_allow_html=True)
            st.plotly_chart(create_sparkline(cust_spark, "#06b6d4"), use_container_width=True, key="s2", config={"displayModeBar": False})

    with kpi3:
        with st.container(border=True):
            st.markdown(f'''
            <div style="padding-bottom:5px;">
                <div class="kpi-label">Catalogue</div>
                <div class="kpi-value">{n_prod:,}</div>
                <div class="kpi-sub" style="margin-bottom:0px;">unique products shipped</div>
            </div>''', unsafe_allow_html=True)
            st.plotly_chart(create_sparkline(prod_spark, "#10b981"), use_container_width=True, key="s3", config={"displayModeBar": False})

    with kpi4:
        with st.container(border=True):
            st.markdown(f'''
            <div style="padding-bottom:5px;">
                <div class="kpi-label">Orders</div>
                <div class="kpi-value">{n_txn:,}</div>
                <div class="kpi-sub" style="margin-bottom:0px;">{n_countries} countries reached</div>
            </div>''', unsafe_allow_html=True)
            st.plotly_chart(create_sparkline(order_spark, "#f59e0b"), use_container_width=True, key="s4", config={"displayModeBar": False})

    st.markdown("""
        <style>
        [data-testid='stPlotlyChart'] {
            margin-top: -20px;
            margin-bottom: -15px;
        }
        </style>
    """, unsafe_allow_html=True)

    # --- Insight callout ---
    st.markdown(
        f'<div class="insight-box">'
        f"<strong>Did you know?</strong> The UK alone accounts for "
        f"<strong>{uk_pct:.0f}%</strong> of total revenue. "
        f"The best month was <strong>{best_month['YearMonth']}</strong> "
        f"with £{best_month['Revenue']:,.0f} in sales across "
        f"{int(best_month['Orders']):,} orders — likely driven by the "
        f"holiday season.</div>",
        unsafe_allow_html=True,
    )

    # --- Row 1: Revenue Trend + Country Revenue ---
    c1, c2 = st.columns([3, 2])

    with c1:
        st.markdown('<div class="section-title">Monthly Revenue Trend</div>', unsafe_allow_html=True)

        fig = go.Figure(
            go.Scatter(
                x=monthly["YearMonth"],
                y=monthly["Revenue"],
                fill="tozeroy",
                fillcolor="rgba(99,102,241,0.10)",
                line=dict(color="#6366f1", width=2.5, shape="spline"),
                mode="lines",
                hovertemplate="<b>%{x}</b><br>Revenue: £%{y:,.0f}<extra></extra>",
            )
        )
        fig.update_layout(
            **_layout(
                height=330,
                xaxis=dict(title="", showgrid=False, tickangle=-30),
                yaxis=dict(title="", gridcolor="#f1f5f9", tickformat=",.0f", tickprefix="£"),
            )
        )
        st.plotly_chart(fig, use_container_width=True, key="trend")

    with c2:
        st.markdown('<div class="section-title">Where the money comes from</div>', unsafe_allow_html=True)

        top_countries = df.groupby("Country")["TotalPrice"].sum().sort_values(ascending=True).tail(10)
        fig2 = go.Figure(
            go.Bar(
                x=top_countries.values,
                y=top_countries.index,
                orientation="h",
                marker=dict(
                    color=top_countries.values,
                    colorscale=[[0, "#a5b4fc"], [1, "#6366f1"]],
                ),
                hovertemplate="<b>%{y}</b><br>£%{x:,.0f}<extra></extra>",
            )
        )
        fig2.update_layout(
            **_layout(
                height=330,
                xaxis=dict(title="", showgrid=True, gridcolor="#f1f5f9", tickformat=",.0s"),
                yaxis=dict(title="", tickfont=dict(size=11)),
            )
        )
        st.plotly_chart(fig2, use_container_width=True, key="countries")

    # --- Geographic Revenue Map ---
    st.markdown('<div class="section-title">Global Revenue Distribution</div>', unsafe_allow_html=True)
    country_rev = df.groupby("Country")["TotalPrice"].sum().reset_index()
    fig_map = go.Figure(data=go.Choropleth(
        locations=country_rev["Country"],
        locationmode="country names",
        z=country_rev["TotalPrice"],
        colorscale=[[0, "#eef2ff"], [0.2, "#a5b4fc"], [0.5, "#6366f1"], [1, "#312e81"]],
        marker_line_color="white",
        marker_line_width=0.5,
        hovertemplate="<b>%{location}</b><br>Revenue: £%{z:,.0f}<extra></extra>",
        showscale=False
    ))
    fig_map.update_layout(
        **_layout(
            height=380,
            geo=dict(
                showframe=False, showcoastlines=False,
                projection_type="natural earth",
                bgcolor="rgba(0,0,0,0)",
                landcolor="#f1f5f9",
            ),
            margin=dict(l=0, r=0, t=10, b=0)
        )
    )
    st.plotly_chart(fig_map, use_container_width=True, key="choropleth")

    # --- Row 2: Top Products + Segment Distribution ---
    c3, c4 = st.columns([3, 2])

    with c3:
        st.markdown('<div class="section-title">Bestsellers (by units shipped)</div>', unsafe_allow_html=True)

        top_prods = df.groupby("Description")["Quantity"].sum().nlargest(15).sort_values(ascending=True)
        labels = [n[:38] + "…" if len(n) > 38 else n for n in top_prods.index]

        fig3 = go.Figure(
            go.Bar(
                x=top_prods.values,
                y=labels,
                orientation="h",
                marker=dict(color="#06b6d4"),
                hovertemplate="<b>%{y}</b><br>%{x:,} units<extra></extra>",
            )
        )
        fig3.update_layout(
            **_layout(
                height=440,
                xaxis=dict(title="", showgrid=True, gridcolor="#f1f5f9", tickformat=","),
                yaxis=dict(title="", tickfont=dict(size=10.5)),
                margin=dict(l=0, r=0, t=10, b=0),
            )
        )
        st.plotly_chart(fig3, use_container_width=True, key="top_prods")

    with c4:
        st.markdown('<div class="section-title">Customer Segments</div>', unsafe_allow_html=True)

        rfm_tmp = rfm.copy()
        rfm_log = np.log1p(rfm_tmp[["Recency", "Frequency", "Monetary"]])
        rfm_tmp["Segment"] = pd.Series(
            kmeans.predict(scaler.transform(rfm_log))
        ).map(segment_labels).values

        seg_counts = rfm_tmp["Segment"].value_counts()
        order = [s for s in ["High-Value", "Regular", "Occasional", "At-Risk"] if s in seg_counts]
        vals = [seg_counts[s] for s in order]
        cols = [SEGMENT_META[s]["color"] for s in order]

        fig4 = go.Figure(
            go.Pie(
                labels=order,
                values=vals,
                hole=0.62,
                marker=dict(colors=cols),
                textinfo="percent+label",
                textposition="outside",
                textfont=dict(size=11),
                hovertemplate="<b>%{label}</b><br>%{value:,} customers<br>%{percent}<extra></extra>",
                pull=[0.03 if s == "High-Value" else 0 for s in order],
            )
        )
        fig4.update_layout(
            **_layout(
                height=440,
                showlegend=False,
                margin=dict(l=30, r=30, t=10, b=30),
            )
        )
        st.plotly_chart(fig4, use_container_width=True, key="seg_pie")

    # --- Row 3: Shopping hour heatmap ---
    st.markdown(
        '<div class="section-title">When do customers shop?</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Each cell shows the number of orders placed on a given day-of-week "
        "and hour. Darker = busier. Useful for scheduling email campaigns and flash sales."
    )

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Sunday"]
    hour_day = (
        df.groupby(["DayOfWeek", "Hour"])["InvoiceNo"]
        .nunique()
        .reset_index()
        .rename(columns={"InvoiceNo": "Orders"})
    )
    heatmap_pivot = hour_day.pivot(index="DayOfWeek", columns="Hour", values="Orders").fillna(0)
    heatmap_pivot = heatmap_pivot.reindex([d for d in day_order if d in heatmap_pivot.index])

    fig_heat = go.Figure(
        go.Heatmap(
            z=heatmap_pivot.values,
            x=[f"{h}:00" for h in heatmap_pivot.columns],
            y=heatmap_pivot.index,
            colorscale=[[0, "#eef2ff"], [0.5, "#818cf8"], [1, "#3730a3"]],
            hovertemplate="<b>%{y} %{x}</b><br>%{z:,} orders<extra></extra>",
            showscale=False,
        )
    )
    fig_heat.update_layout(
        **_layout(
            height=260,
            xaxis=dict(title="", showgrid=False, side="top", tickfont=dict(size=10)),
            yaxis=dict(title="", showgrid=False, autorange="reversed", tickfont=dict(size=11)),
            margin=dict(l=0, r=0, t=30, b=0),
        )
    )
    st.plotly_chart(fig_heat, use_container_width=True, key="heatmap")

    # --- Cohort Retention Heatmap ---
    st.markdown('<div class="section-title">Cohort Retention Rate</div>', unsafe_allow_html=True)
    st.caption("Percentage of customers from each monthly cohort who made a purchase in subsequent months.")
    
    # 1. Identify first purchase month for each customer
    first_purchase = df.groupby('CustomerID')['InvoiceDate'].min().dt.to_period('M').reset_index()
    first_purchase.columns = ['CustomerID', 'CohortMonth']
    
    # 2. Merge back to get the cohort month for each transaction
    df_cohort = pd.merge(df, first_purchase, on='CustomerID')
    
    # 3. Calculate months elapsed
    df_cohort['TransactionMonth'] = df_cohort['InvoiceDate'].dt.to_period('M')
    df_cohort['CohortIndex'] = (df_cohort['TransactionMonth'] - df_cohort['CohortMonth']).apply(lambda x: x.n)
    
    # 4. Group by cohort and month index
    cohort_data = df_cohort.groupby(['CohortMonth', 'CohortIndex'])['CustomerID'].nunique().reset_index()
    
    # 5. Pivot into a matrix
    cohort_counts = cohort_data.pivot(index='CohortMonth', columns='CohortIndex', values='CustomerID')
    cohort_sizes = cohort_counts.iloc[:, 0]
    retention = cohort_counts.divide(cohort_sizes, axis=0) * 100
    retention = retention.round(1)
    
    # 6. Heatmap
    y_labels = [str(x) for x in retention.index]
    x_labels = [f"M{c}" for c in retention.columns]
    
    # Replace NaN with 0 for rendering or leave as NaN to show empty, leaving as is allows gaps
    # But for better text formatting, we can replace NaN text with ""
    ret_text = retention.map(lambda x: f"{x:.0f}%" if pd.notnull(x) else "")
    
    fig_ret = go.Figure(data=go.Heatmap(
        z=retention.values,
        x=x_labels,
        y=y_labels,
        colorscale=[[0, "#f8fafc"], [0.2, "#a5b4fc"], [1, "#312e81"]],
        hovertemplate="<b>Cohort: %{y}</b><br>%{x}<br>Retention: %{text}<extra></extra>",
        showscale=False,
        text=ret_text.values,
        texttemplate="%{text}",
        xgap=2, ygap=2
    ))
    fig_ret.update_layout(
        **_layout(
            height=400,
            xaxis=dict(title="", showgrid=False, side="top", tickfont=dict(size=10)),
            yaxis=dict(title="", showgrid=False, autorange="reversed", tickfont=dict(size=11), type="category"),
            margin=dict(l=0, r=0, t=30, b=0)
        )
    )
    st.plotly_chart(fig_ret, use_container_width=True, key="retention")


# ═══════════════════════════════════════════════════════════════════
#  CUSTOMER DEEP-DIVE
# ═══════════════════════════════════════════════════════════════════
elif page == "Customer Deep-Dive":

    st.markdown(
        '<div class="page-header">'
        "<h1>Customer Deep-Dive</h1>"
        "<p>Look up a specific customer by ID to view their full purchase history, "
        "favourite products, and lifetime value metrics.</p></div>",
        unsafe_allow_html=True,
    )

    all_custs = sorted(df["CustomerID"].dropna().unique().astype(int).tolist())
    
    col_search, _ = st.columns([1, 2])
    with col_search:
        selected_cust = st.selectbox(
            "Select or search for a Customer ID",
            options=[""] + all_custs,
            format_func=lambda x: "Type an ID …" if x == "" else f"Customer #{x}",
        )

    if selected_cust:
        cust_df = df[df["CustomerID"] == selected_cust].copy()
        
        c_rev = cust_df["TotalPrice"].sum()
        c_orders = cust_df["InvoiceNo"].nunique()
        c_items = cust_df["Quantity"].sum()
        c_first = cust_df["InvoiceDate"].min()
        c_last = cust_df["InvoiceDate"].max()
        c_country = cust_df["Country"].iloc[0]
        
        # Profile header
        st.markdown(f'<div class="section-title">Customer #{selected_cust} Profile</div>', unsafe_allow_html=True)
        
        pr1, pr2, pr3, pr4 = st.columns(4)
        pr1.metric("Lifetime Value", f"£{c_rev:,.2f}")
        pr2.metric("Total Orders", f"{c_orders:,}")
        pr3.metric("Items Bought", f"{c_items:,}")
        pr4.metric("Location", c_country)
        
        st.caption(f"**First purchase:** {c_first.strftime('%d %b %Y')} &nbsp;•&nbsp; **Last purchase:** {c_last.strftime('%d %b %Y')} ({ (c_last - c_first).days } days active)")
        
        col_hist, col_top = st.columns([2, 1])
        
        with col_hist:
            st.markdown('<div class="section-title">Purchase Timeline</div>', unsafe_allow_html=True)
            order_timeline = cust_df.groupby(cust_df["InvoiceDate"].dt.date)["TotalPrice"].sum().reset_index()
            fig_time = go.Figure(go.Scatter(
                x=order_timeline["InvoiceDate"],
                y=order_timeline["TotalPrice"],
                mode="markers+lines",
                marker=dict(size=8, color="#6366f1", line=dict(width=1, color="#312e81")),
                line=dict(color="#a5b4fc", width=2, dash="dot"),
                hovertemplate="<b>%{x|%d %b %Y}</b><br>Spent: £%{y:,.2f}<extra></extra>"
            ))
            fig_time.update_layout(
                **_layout(
                    height=280,
                    xaxis=dict(title="", showgrid=False),
                    yaxis=dict(title="Order Value (£)", gridcolor="#f1f5f9"),
                )
            )
            st.plotly_chart(fig_time, use_container_width=True, key="timeline")
            
        with col_top:
            st.markdown('<div class="section-title">Top Products</div>', unsafe_allow_html=True)
            top_p = cust_df.groupby("Description")["Quantity"].sum().nlargest(5).sort_values(ascending=True)
            labels_p = [n[:25] + "…" if len(n) > 25 else n for n in top_p.index]
            fig_p = go.Figure(go.Bar(
                x=top_p.values, y=labels_p, orientation="h",
                marker=dict(color="#14b8a6"),
                hovertemplate="<b>%{y}</b><br>%{x} units<extra></extra>"
            ))
            fig_p.update_layout(
                **_layout(
                    height=280,
                    xaxis=dict(title="Units", showgrid=True, gridcolor="#f1f5f9"),
                    yaxis=dict(title="", tickfont=dict(size=10)),
                    margin=dict(l=0, r=0, t=10, b=0)
                )
            )
            st.plotly_chart(fig_p, use_container_width=True, key="top_cust_prod")

        # Segmentation lookup
        st.markdown('<div class="section-title">Segment Assignment</div>', unsafe_allow_html=True)
        
        cust_rfm = rfm[rfm["CustomerID"] == selected_cust]
        if not cust_rfm.empty:
            r = cust_rfm["Recency"].values[0]
            f = cust_rfm["Frequency"].values[0]
            m = cust_rfm["Monetary"].values[0]
            
            inp = np.log1p(np.array([[r, f, m]]))
            cluster_id = int(kmeans.predict(scaler.transform(inp))[0])
            seg = segment_labels[cluster_id]
            meta = SEGMENT_META[seg]
            
            sc1, sc2 = st.columns([1, 2])
            with sc1:
                st.markdown(
                    f"""
                <div class="seg-result" style="background:{meta['bg']};border:2px solid {meta['border']}; margin-top:0;">
                    <div class="seg-label" style="color:{meta['color']}">Assigned Segment</div>
                    <div class="seg-name" style="color:{meta['color']}">{meta['icon']} {seg}</div>
                </div>""",
                    unsafe_allow_html=True,
                )
            with sc2:
                # Radar Chart specific to this customer
                centroids_real = np.expm1(scaler.inverse_transform(kmeans.cluster_centers_))
                all_pts = np.vstack([centroids_real, [[r, f, m]]])
                normed = all_pts.copy().astype(float)
                normed[:, 0] = normed[:, 0].max() - normed[:, 0]
                for j in range(3):
                    lo, hi = normed[:, j].min(), normed[:, j].max()
                    normed[:, j] = (normed[:, j] - lo) / (hi - lo + 1e-9)

                cats = ["Recency<br>(lower=better)", "Frequency", "Monetary"]
                fig_rc = go.Figure()
                
                # Add background centroids
                for i in range(len(centroids_real)):
                    sn = segment_labels[i]
                    if sn == seg:
                        r_vals = normed[i].tolist() + [normed[i][0]]
                        fig_rc.add_trace(go.Scatterpolar(
                            r=r_vals, theta=cats + [cats[0]], name=f"Avg {sn}",
                            line=dict(color=meta["color"], width=2, dash="dot"),
                            opacity=0.5, fill="none",
                        ))
                        
                # Add this customer
                inp_rc = normed[-1].tolist() + [normed[-1][0]]
                fig_rc.add_trace(go.Scatterpolar(
                    r=inp_rc, theta=cats + [cats[0]], name="This Customer",
                    line=dict(color="#0f172a", width=3),
                    fill="toself", fillcolor="rgba(15,23,42,0.1)",
                ))
                
                fig_rc.update_layout(
                    **_layout(
                        height=220, showlegend=True,
                        legend=dict(orientation="v", y=0.5, x=1.1, font=dict(size=10)),
                        polar=dict(
                            bgcolor="rgba(0,0,0,0)",
                            radialaxis=dict(visible=True, range=[0, 1], showticklabels=False, gridcolor="#e2e8f0"),
                            angularaxis=dict(gridcolor="#e2e8f0"),
                        ),
                        margin=dict(l=40, r=40, t=20, b=20),
                    )
                )
                st.plotly_chart(fig_rc, use_container_width=True, key="radar_cust")


# ═══════════════════════════════════════════════════════════════════
#  PRODUCT RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════
elif page == "Product Recommendations":

    st.markdown(
        '<div class="page-header">'
        "<h1>Product Recommendations</h1>"
        "<p>Pick any product from the catalogue and we'll find the five items "
        "most frequently bought by the same customers — powered by cosine "
        f"similarity across {len(similarity_df):,} products.</p></div>",
        unsafe_allow_html=True,
    )

    # Quick picks
    st.markdown(
        '<div class="quick-picks"><p>Try one of these popular items:</p></div>',
        unsafe_allow_html=True,
    )
    qp_cols = st.columns(5)
    quick_picks = [
        "JUMBO BAG RED RETROSPOT",
        "WHITE HANGING HEART T-LIGHT HOLDER",
        "REGENCY CAKESTAND 3 TIER",
        "PARTY BUNTING",
        "SET OF 3 CAKE TINS PANTRY DESIGN",
    ]
    picked_qp = None
    for i, qp in enumerate(quick_picks):
        with qp_cols[i]:
            if st.button(qp.title(), key=f"qp_{i}", use_container_width=True):
                picked_qp = qp

    all_products = sorted(similarity_df.index.tolist())

    default_idx = 0
    if picked_qp and picked_qp in all_products:
        default_idx = all_products.index(picked_qp) + 1  # +1 for empty option

    selected = st.selectbox(
        "Or search the full catalogue",
        options=[""] + all_products,
        index=default_idx,
        format_func=lambda x: "Type to search …" if x == "" else x.title(),
    )

    if selected:
        sims = similarity_df[selected].sort_values(ascending=False)
        top5 = sims.iloc[1:6]

        # Product stats
        prod_rows = df[df["Description"].str.strip() == selected]
        times_bought = len(prod_rows)
        unique_buyers = prod_rows["CustomerID"].nunique()
        total_qty = prod_rows["Quantity"].sum()
        avg_price = prod_rows["UnitPrice"].mean()

        st.markdown(
            f'<div class="prod-stats">'
            f'<span class="prod-pill">Bought <b>{times_bought:,}</b> times</span>'
            f'<span class="prod-pill"><b>{unique_buyers:,}</b> unique customers</span>'
            f'<span class="prod-pill"><b>{total_qty:,}</b> total units</span>'
            f'<span class="prod-pill">Avg price <b>£{avg_price:.2f}</b></span>'
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-title">Customers also bought</div>', unsafe_allow_html=True)

        for rank, (prod, score) in enumerate(top5.items(), 1):
            pct = max(score * 100, 0)
            st.markdown(
                f"""
            <div class="rec-card">
                <div class="rec-rank">{rank}</div>
                <div class="rec-name">{prod.strip().title()}</div>
                <div class="rec-score-wrap">
                    <div class="rec-score-lbl">{pct:.0f}% match</div>
                    <div class="rec-bar"><div class="rec-fill" style="width:{pct}%"></div></div>
                </div>
            </div>""",
                unsafe_allow_html=True,
            )

        with st.expander("Similarity distribution for this product"):
            all_scores = sims.drop(selected)
            above_50 = (all_scores > 0.5).sum()
            st.caption(
                f"{above_50:,} products have >50% similarity. "
                f"Median similarity: {all_scores.median():.3f}."
            )
            fig_h = go.Figure(
                go.Histogram(
                    x=all_scores.values,
                    nbinsx=60,
                    marker=dict(color="#6366f1", line=dict(color="#4f46e5", width=0.5)),
                    hovertemplate="Score: %{x:.3f}<br>Count: %{y}<extra></extra>",
                )
            )
            fig_h.update_layout(
                **_layout(
                    height=240,
                    xaxis=dict(title="Cosine Similarity", showgrid=False),
                    yaxis=dict(title="Products", gridcolor="#f1f5f9"),
                    margin=dict(l=0, r=0, t=10, b=0),
                )
            )
            st.plotly_chart(fig_h, use_container_width=True, key="sim_hist")

    with st.expander("How does this work?"):
        st.markdown(
            """
**The short version:** we look at what customers buy together, then use maths
to rank products by how often they share the same shoppers.

**The longer version:**

1. Every customer's purchase history is turned into a row in a big matrix
   (customers × products).
2. We compute [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity)
   between every pair of product columns — this tells us how "aligned"
   their buyer audiences are.
3. A score of **1.0** means two products are purchased by an almost
   identical set of customers; **0** means completely different audiences.

This is the same core idea behind "customers who bought X also bought Y"
on sites like Amazon — just without the billion-dollar infrastructure.
"""
        )


# ═══════════════════════════════════════════════════════════════════
#  CUSTOMER SEGMENTATION
# ═══════════════════════════════════════════════════════════════════
elif page == "Customer Segmentation":

    st.markdown(
        '<div class="page-header">'
        "<h1>Customer Segmentation</h1>"
        "<p>Enter a customer's RFM profile and the model will classify them "
        "into one of four segments. Use the presets to explore typical profiles, "
        "or dial in your own numbers.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Presets
    st.markdown("**Quick presets** — try a typical customer profile:", unsafe_allow_html=False)

    presets = {
        "VIP Shopper":       (5, 120, 12000.0),
        "Steady Regular":    (40, 20, 800.0),
        "Holiday Buyer":     (90, 4, 200.0),
        "Ghost Customer":    (350, 2, 50.0),
    }

    preset_cols = st.columns(len(presets))
    chosen_preset = None
    for i, (label, vals) in enumerate(presets.items()):
        with preset_cols[i]:
            if st.button(label, key=f"preset_{i}", use_container_width=True):
                chosen_preset = vals

    # Defaults
    default_r, default_f, default_m = chosen_preset if chosen_preset else (30, 15, 1500.0)

    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        st.markdown('<div class="section-title">RFM Parameters</div>', unsafe_allow_html=True)

        recency = st.slider("Recency — days since last purchase", 1, 400, default_r)
        frequency = st.slider("Frequency — number of transactions", 1, 250, default_f)
        monetary = st.slider("Monetary — total spend (£)", 10.0, 50000.0, default_m, step=50.0)

        # How many real customers look like this?
        r_band = rfm[(rfm["Recency"].between(recency * 0.7, recency * 1.3)) &
                      (rfm["Frequency"].between(max(1, frequency - 5), frequency + 5))]
        similar_count = len(r_band)

        st.caption(
            f"~{similar_count:,} customers in the dataset have a similar "
            f"RFM profile (±30% recency, ±5 frequency)."
        )

        # Predict
        inp = np.log1p(np.array([[recency, frequency, monetary]]))
        cluster_id = int(kmeans.predict(scaler.transform(inp))[0])
        seg = segment_labels[cluster_id]
        meta = SEGMENT_META[seg]

    with col_out:
        st.markdown('<div class="section-title">Predicted Segment</div>', unsafe_allow_html=True)

        st.markdown(
            f"""
        <div class="seg-result" style="background:{meta['bg']};border:2px solid {meta['border']};">
            <div class="seg-label" style="color:{meta['color']}">Predicted Segment</div>
            <div class="seg-name" style="color:{meta['color']}">{meta['icon']} {seg}</div>
            <div class="seg-desc" style="color:{meta['color']}">{SEGMENT_DESCRIPTIONS[seg]}</div>
        </div>""",
            unsafe_allow_html=True,
        )

        # Radar Chart
        centroids_real = np.expm1(scaler.inverse_transform(kmeans.cluster_centers_))
        all_pts = np.vstack([centroids_real, [[recency, frequency, monetary]]])

        normed = all_pts.copy().astype(float)
        normed[:, 0] = normed[:, 0].max() - normed[:, 0]
        for j in range(3):
            lo, hi = normed[:, j].min(), normed[:, j].max()
            normed[:, j] = (normed[:, j] - lo) / (hi - lo + 1e-9)

        cats = ["Recency<br>(lower=better)", "Frequency", "Monetary"]
        fig_r = go.Figure()
        for i in range(len(centroids_real)):
            sn = segment_labels[i]
            r_vals = normed[i].tolist() + [normed[i][0]]
            fig_r.add_trace(
                go.Scatterpolar(
                    r=r_vals, theta=cats + [cats[0]], name=sn,
                    line=dict(color=SEGMENT_META[sn]["color"], width=1.5, dash="dot"),
                    opacity=0.35, fill="none",
                )
            )
        inp_r = normed[-1].tolist() + [normed[-1][0]]
        fig_r.add_trace(
            go.Scatterpolar(
                r=inp_r, theta=cats + [cats[0]], name="Your Input",
                line=dict(color="#0f172a", width=3),
                fill="toself", fillcolor="rgba(15,23,42,0.06)",
            )
        )
        fig_r.update_layout(
            **_layout(
                height=310, showlegend=True,
                legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center", font=dict(size=9.5)),
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=True, range=[0, 1], showticklabels=False, gridcolor="#e2e8f0"),
                    angularaxis=dict(gridcolor="#e2e8f0"),
                ),
                margin=dict(l=60, r=60, t=20, b=50),
            )
        )
        st.plotly_chart(fig_r, use_container_width=True, key="radar")

    # --- Marketing Playbook ---
    st.markdown('<div class="section-title">What to do with this segment</div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="playbook"><h4>Marketing playbook for {seg} customers</h4><ul>'
        + "".join(f"<li>{tip}</li>" for tip in SEGMENT_PLAYBOOK[seg])
        + "</ul></div>",
        unsafe_allow_html=True,
    )

    # --- Segment Profiles Table ---
    st.markdown('<div class="section-title">All Segment Profiles</div>', unsafe_allow_html=True)

    rfm_all = rfm.copy()
    rfm_log_all = np.log1p(rfm_all[["Recency", "Frequency", "Monetary"]])
    rfm_all["Segment"] = pd.Series(
        kmeans.predict(scaler.transform(rfm_log_all))
    ).map(segment_labels).values

    seg_table = (
        rfm_all.groupby("Segment")
        .agg(
            Customers=("CustomerID", "count"),
            Avg_Recency=("Recency", "mean"),
            Avg_Frequency=("Frequency", "mean"),
            Avg_Monetary=("Monetary", "mean"),
        )
        .round(1)
    )
    seg_table.columns = ["Customers", "Avg Recency (days)", "Avg Frequency", "Avg Monetary (£)"]
    seg_table = seg_table.reindex(
        [s for s in ["High-Value", "Regular", "Occasional", "At-Risk"] if s in seg_table.index]
    )

    st.dataframe(
        seg_table.style.format(
            {"Customers": "{:,.0f}", "Avg Recency (days)": "{:.0f}",
             "Avg Frequency": "{:.0f}", "Avg Monetary (£)": "£{:,.0f}"}
        ),
        use_container_width=True,
    )


# ═══════════════════════════════════════════════════════════════════
#  DATA EXPLORER
# ═══════════════════════════════════════════════════════════════════
elif page == "Data Explorer":

    st.markdown(
        '<div class="page-header">'
        "<h1>Data Explorer</h1>"
        "<p>Dig into the raw transactions. Filter by country, date, or "
        "minimum spend — the summary stats update in real time.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        countries_list = ["All countries"] + sorted(df["Country"].unique().tolist())
        country_sel = st.selectbox("Country", countries_list, key="de_country")
    with fc2:
        d_min, d_max = df["InvoiceDate"].min().date(), df["InvoiceDate"].max().date()
        d_range = st.date_input(
            "Date range", value=(d_min, d_max),
            min_value=d_min, max_value=d_max, key="de_dates",
        )
    with fc3:
        min_val = st.number_input(
            "Min line total (£)", value=0.0, step=10.0, key="de_minval",
            help="Filter to rows where Quantity × UnitPrice ≥ this amount.",
        )

    filt = df.copy()
    if country_sel != "All countries":
        filt = filt[filt["Country"] == country_sel]
    if isinstance(d_range, (list, tuple)) and len(d_range) == 2:
        filt = filt[
            (filt["InvoiceDate"].dt.date >= d_range[0])
            & (filt["InvoiceDate"].dt.date <= d_range[1])
        ]
    if min_val > 0:
        filt = filt[filt["TotalPrice"] >= min_val]

    st.markdown('<div class="section-title">At a glance</div>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Records", f"{len(filt):,}")
    m2.metric("Customers", f"{filt['CustomerID'].nunique():,}")
    m3.metric("Revenue", f"£{filt['TotalPrice'].sum():,.0f}")
    m4.metric("Avg Unit Price", f"£{filt['UnitPrice'].mean():.2f}")

    # Quick insights for filtered data
    if len(filt) > 0:
        top_prod_filt = filt["Description"].value_counts().head(1)
        top_cust_filt = filt.groupby("CustomerID")["TotalPrice"].sum().idxmax()
        top_cust_spend = filt.groupby("CustomerID")["TotalPrice"].sum().max()

        st.markdown(
            f'<div class="insight-box">'
            f"<strong>Quick look:</strong> "
            f'The most frequent item is "<strong>{top_prod_filt.index[0].title()}</strong>" '
            f"({top_prod_filt.values[0]:,} line items). "
            f"Top spender is customer <strong>#{top_cust_filt}</strong> "
            f"at £{top_cust_spend:,.0f}.</div>",
            unsafe_allow_html=True,
        )

    # Revenue mini-chart for filtered data
    if len(filt) > 100:
        filt_monthly = filt.groupby("YearMonth")["TotalPrice"].sum().reset_index().sort_values("YearMonth")
        fig_mini = go.Figure(
            go.Bar(
                x=filt_monthly["YearMonth"],
                y=filt_monthly["TotalPrice"],
                marker=dict(color="#06b6d4"),
                hovertemplate="<b>%{x}</b><br>£%{y:,.0f}<extra></extra>",
            )
        )
        fig_mini.update_layout(
            **_layout(
                height=200,
                xaxis=dict(title="", showgrid=False, tickangle=-30, tickfont=dict(size=10)),
                yaxis=dict(title="", gridcolor="#f1f5f9", tickformat=",.0s", tickprefix="£"),
                margin=dict(l=0, r=0, t=10, b=0),
            )
        )
        st.plotly_chart(fig_mini, use_container_width=True, key="filt_rev")

    st.markdown('<div class="section-title">Transaction Data</div>', unsafe_allow_html=True)
    show_cols = [
        "InvoiceNo", "StockCode", "Description", "Quantity",
        "UnitPrice", "TotalPrice", "CustomerID", "Country", "InvoiceDate",
    ]
    st.dataframe(filt[show_cols].head(5000), use_container_width=True, height=480)


# ─── Footer ────────────────────────────────────────────────────────
st.markdown(
    '<div class="app-footer">'
    "Shopper Spectrum · Built by "
    '<a href="https://github.com/Alvira-Parveen" target="_blank">Alvira Parveen</a>'
    "<br>RFM segmentation + collaborative filtering on the Online Retail dataset"
    "</div>",
    unsafe_allow_html=True,
)