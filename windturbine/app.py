import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Wind Turbine Fault Detector",
                   page_icon="🌬️", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; max-width: 1250px;}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0F2038 0%,#0B1220 100%);
    border-right: 1px solid #1E2D4A;}
.hero {background: linear-gradient(120deg,#0072FF 0%,#00C6FF 55%,#00E5A0 100%);
    padding: 28px 32px; border-radius: 18px; margin-bottom: 22px;
    box-shadow: 0 10px 30px rgba(0,150,255,.25);}
.hero h1 {color:#fff; margin:0; font-size:2rem; font-weight:800;}
.hero p {color:#EAF6FF; margin:6px 0 0 0; font-size:1rem;}
.kpi {background: linear-gradient(145deg,#16233A,#111B2D);
    border:1px solid #22314F; border-radius:16px; padding:16px 18px;
    box-shadow:0 6px 18px rgba(0,0,0,.35); margin-bottom:10px;}
.kpi-l {color:#8FA3C4; font-size:.75rem; letter-spacing:.08em;
    text-transform:uppercase;}
.kpi-v {color:#fff; font-size:1.9rem; font-weight:800; line-height:1.25;}
.kpi-s {color:#8FA3C4; font-size:.8rem;}
.result {background:#111B2D; border:1px solid #22314F; border-radius:14px;
    padding:16px 20px; margin-top:12px;}
.result b {font-size:1.15rem;}
.sec {color:#fff; font-size:1.25rem; font-weight:700; margin:18px 0 8px 0;
    padding-left:10px; border-left:4px solid #00D4FF;}
</style>
""", unsafe_allow_html=True)

COLORS = {"Critical": "#FF4D4F", "Warning": "#FFA940", "OK": "#36CFC9"}
CAUSE_COL = {"Pitch Fault / Icing": "#00D4FF",
             "Gearbox Friction / Overheating": "#FF7A45",
             "Sensor Drift / Other": "#FFC53D", "Healthy": "#36CFC9",
             "Normal": "#36CFC9"}
PALETTE = px.colors.qualitative.Bold


@st.cache_data
def load():
    return (pd.read_csv("wind_scada.csv"), pd.read_csv("wind_clean.csv"),
            pd.read_csv("ranking.csv"), pd.read_csv("diagnosis.csv"),
            pd.read_csv("wind_diag.csv"))


@st.cache_resource
def load_model():
    if os.path.exists("model.pkl"):
        return joblib.load("model.pkl")
    return None


raw, df, rank, diag, wd = load()
bundle = load_model()


def status(loss):
    return "Critical" if loss >= 15 else ("Warning" if loss >= 5 else "OK")


rank["Status"] = rank["Loss_%"].apply(status)


def kpi(label, value, sub="", color="#00D4FF"):
    return (f'<div class="kpi" style="border-top:3px solid {color}">'
            f'<div class="kpi-l">{label}</div><div class="kpi-v">{value}</div>'
            f'<div class="kpi-s">{sub}</div></div>')


def hero(title, sub):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{sub}</p></div>',
                unsafe_allow_html=True)


def sec(text):
    st.markdown(f'<div class="sec">{text}</div>', unsafe_allow_html=True)


def style(fig, h=450):
    fig.update_layout(template="plotly_dark", height=h,
                      paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(19,28,46,0.6)",
                      margin=dict(l=20, r=20, t=50, b=20),
                      legend=dict(orientation="h", y=1.12))
    return fig


def theo(v):
    p = np.where(v < 3, 0, 2000 * ((v - 3) / 9) ** 3)
    p = np.where(v >= 12, 2000, p)
    return np.where(v >= 25, 0, p)


# ---------------- Sidebar ----------------
st.sidebar.markdown("## 🌬️ Turbine Doctor")
st.sidebar.caption("Wind Turbine Underperformance & Fault Detector")
page = st.sidebar.radio("Navigate", [
    "🏠 Overview", "📈 Power Curve", "🏆 Ranking",
    "🩺 Diagnosis", "📊 Charts & Models"])
st.sidebar.divider()
if bundle is not None:
    st.sidebar.success(f"ML model loaded: {bundle['name']}")
else:
    st.sidebar.warning("model.pkl not found")
st.sidebar.caption("Data: simulated SCADA | 8 turbines | 10-min intervals")
st.sidebar.caption("Built with Python, Pandas, Scikit-Learn, Plotly, Streamlit")

# ---------------- Overview ----------------
if page == "🏠 Overview":
    hero("Wind Turbine Fault Detector",
         "Turbines can silently lose power without any ON/OFF alarm. "
         "This tool compares actual vs theoretical output and finds the probable cause.")
    worst = rank.sort_values("Loss_%", ascending=False).iloc[0]
    n_bad = int((rank["Status"] != "OK").sum())
    c = st.columns(4)
    c[0].markdown(kpi("Turbines monitored", df["Turbine_ID"].nunique(),
                      "8 in the fleet"), unsafe_allow_html=True)
    c[1].markdown(kpi("Clean records", f"{len(df):,}",
                      "after preprocessing", "#36CFC9"), unsafe_allow_html=True)
    c[2].markdown(kpi("Need attention", n_bad, "warning or critical",
                      "#FFA940"), unsafe_allow_html=True)
    c[3].markdown(kpi("Worst turbine", worst["Turbine_ID"],
                      f"{worst['Loss_%']}% energy lost", "#FF4D4F"),
                  unsafe_allow_html=True)

    a, b = st.columns(2)
    with a:
        sec("Fleet health")
        cnt = rank["Status"].value_counts().reset_index()
        cnt.columns = ["Status", "Turbines"]
        fig = px.pie(cnt, names="Status", values="Turbines", hole=0.55,
                     color="Status", color_discrete_map=COLORS)
        st.plotly_chart(style(fig, 330))
    with b:
        sec("Energy loss by turbine")
        r = rank.copy()
        r["Loss_%"] = r["Loss_%"].clip(lower=0)
        fig = px.bar(r.sort_values("Loss_%", ascending=False),
                     x="Turbine_ID", y="Loss_%", color="Status",
                     color_discrete_map=COLORS)
        st.plotly_chart(style(fig, 330))

    sec("Data exploration (Exp 1 and 2)")
    with st.expander("Raw data preview"):
        st.dataframe(raw.head(20))
    with st.expander("Descriptive statistics"):
        st.dataframe(raw.describe())
    with st.expander("Missing values and duplicates in raw data"):
        x, y = st.columns(2)
        x.dataframe(raw.isnull().sum().rename("Missing values"))
        y.metric("Duplicate rows", int(raw.duplicated().sum()))

# ---------------- Power curve ----------------
elif page == "📈 Power Curve":
    hero("Power Curve Overlay",
         "Actual output vs the factory power curve. The bigger the gap, the bigger the loss.")
    ids = sorted(df["Turbine_ID"].unique())
    pick = st.multiselect("Select turbines to compare", ids, default=["T1", "T7"])

    fig = go.Figure()
    ws = np.linspace(0, 26, 261)
    for t in pick:
        i = ids.index(t)
        s = df[df["Turbine_ID"] == t]
        smp = s.sample(1200, random_state=1)
        fig.add_trace(go.Scatter(
            x=smp["WindSpeed"], y=smp["ActivePower"], mode="markers",
            name=f"{t} readings",
            marker=dict(size=4, opacity=0.35, color=PALETTE[i % len(PALETTE)])))
        s2 = s.copy()
        s2["b"] = (s2["WindSpeed"] * 2).round() / 2
        m = s2.groupby("b")["ActivePower"].mean().reset_index()
        fig.add_trace(go.Scatter(x=m["b"], y=m["ActivePower"], mode="lines",
                                 name=f"{t} average",
                                 line=dict(width=3, color=PALETTE[i % len(PALETTE)])))
    fig.add_trace(go.Scatter(x=ws, y=theo(ws), mode="lines",
                             name="Theoretical curve",
                             line=dict(color="#FFD666", width=4, dash="dash")))
    fig.update_layout(xaxis_title="Wind speed (m/s)", yaxis_title="Power (kW)")
    st.plotly_chart(style(fig, 560))

    if pick:
        cols = st.columns(len(pick))
        for col, t in zip(cols, pick):
            row = rank[rank["Turbine_ID"] == t].iloc[0]
            stt = row["Status"]
            col.markdown(kpi(f"{t} energy loss", f"{max(row['Loss_%'], 0)}%",
                             stt, COLORS[stt]), unsafe_allow_html=True)

# ---------------- Ranking ----------------
elif page == "🏆 Ranking":
    hero("Underperformance Leaderboard",
         "Turbines ranked by the percentage of energy they failed to generate.")
    price = st.slider("Electricity price (Rs per kWh)", 1.0, 15.0, 5.0, 0.5)
    r = rank.copy()
    r["Lost_kWh"] = ((r["Theoretical_kWh"] - r["Actual_kWh"]) / 6).clip(lower=0).round(0)
    r["Revenue_Lost_Rs"] = (r["Lost_kWh"] * price).round(0)
    r["Loss"] = r["Loss_%"].clip(lower=0)
    r = r.sort_values("Loss_%", ascending=False).reset_index(drop=True)
    r.insert(0, "Rank", range(1, len(r) + 1))

    medals = ["🥇", "🥈", "🥉"]
    c = st.columns(3)
    for i in range(3):
        row = r.iloc[i]
        c[i].markdown(kpi(f"{medals[i]} {row['Turbine_ID']}  -  {row['Status']}",
                          f"{row['Loss']}% lost",
                          f"Rs {int(row['Revenue_Lost_Rs']):,} lost in 30 days",
                          COLORS[row["Status"]]), unsafe_allow_html=True)

    sec("Full leaderboard")
    st.dataframe(
        r[["Rank", "Turbine_ID", "Loss", "Lost_kWh", "Revenue_Lost_Rs", "Status"]],
        hide_index=True,
        column_config={
            "Loss": st.column_config.ProgressColumn(
                "Energy loss %", min_value=0, max_value=40, format="%.2f%%"),
            "Lost_kWh": st.column_config.NumberColumn("Lost energy (kWh)"),
            "Revenue_Lost_Rs": st.column_config.NumberColumn(
                "Revenue lost (Rs)", format="Rs %d")})
    st.caption("Tiny negative losses are measurement noise and are shown as 0.")

    sec("Loss comparison")
    fig = px.bar(r, x="Turbine_ID", y="Loss", color="Status", text="Loss",
                 color_discrete_map=COLORS)
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(yaxis_title="Energy loss %")
    st.plotly_chart(style(fig, 400))
    st.markdown(kpi("Total revenue lost (30 days)",
                    f"Rs {int(r['Revenue_Lost_Rs'].sum()):,}",
                    f"at Rs {price} per kWh", "#FF4D4F"), unsafe_allow_html=True)

# ---------------- Diagnosis ----------------
elif page == "🩺 Diagnosis":
    hero("Probable Cause Diagnostic",
         "A trained machine learning model predicts the cause from live sensor readings.")
    d = diag.copy()
    d["Loss_%"] = d["Loss_%"].clip(lower=0)
    sec("Fleet diagnosis")
    st.dataframe(d, hide_index=True, column_config={
        "Underperf_Time_pct": st.column_config.ProgressColumn(
            "Time underperforming", min_value=0, max_value=100, format="%.1f%%"),
        "Loss_%": st.column_config.ProgressColumn(
            "Energy loss", min_value=0, max_value=40, format="%.2f%%")})

    ids = sorted(wd["Turbine_ID"].unique())
    a, b = st.columns([1, 2])
    with a:
        t = st.selectbox("Turbine detail", ids,
                         index=ids.index("T7") if "T7" in ids else 0)
    sub = wd[(wd["Turbine_ID"] == t) & (wd["Underperforming"] == 1)]
    with b:
        if len(sub):
            fig = px.pie(sub, names="Cause", hole=0.55, color="Cause",
                         color_discrete_map=CAUSE_COL,
                         title=f"What is hurting {t}?")
            st.plotly_chart(style(fig, 320))
        else:
            st.success(f"{t} shows no significant underperformance.")

    sec("Live fault tester (uses the trained ML model)")
    c1, c2, c3, c4 = st.columns(4)
    wsv = c1.number_input("Wind speed (m/s)", 3.0, 25.0, 10.0)
    act = c2.number_input("Actual power (kW)", 0.0, 2200.0, 400.0)
    rpm = c3.number_input("Rotor RPM", 0.0, 20.0, 5.0)
    temp = c4.number_input("Generator temp (C)", 0.0, 120.0, 60.0)
    if st.button("🔍 Diagnose", type="primary"):
        th = float(theo(np.array([wsv]))[0])
        loss = (th - act) / th if th > 0 else 0
        if th <= 0:
            st.info("Wind speed is below cut-in (3 m/s). "
                    "The turbine is not expected to produce power.")
        elif bundle is not None:
            X = pd.DataFrame([[wsv, th, act, rpm, temp, loss]],
                             columns=bundle["features"])
            model = bundle["model"]
            pred = model.predict(X)[0]
            proba = pd.Series(model.predict_proba(X)[0],
                              index=model.classes_).sort_values()
            col = CAUSE_COL.get(pred, "#00D4FF")
            g, h = st.columns([1, 2])
            gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=max(loss, 0) * 100,
                number={"suffix": "%"}, title={"text": "Power loss"},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": col},
                       "steps": [{"range": [0, 10], "color": "#12372A"},
                                 {"range": [10, 25], "color": "#4A3A12"},
                                 {"range": [25, 100], "color": "#4A1A1A"}]}))
            g.plotly_chart(style(gauge, 260))
            h.markdown(
                f'<div class="result" style="border-left:6px solid {col}">'
                f'Theoretical power: <b>{th:.0f} kW</b><br><br>'
                f'ML model prediction ({bundle["name"]}):<br>'
                f'<b style="color:{col}">{pred}</b><br>'
                f'Confidence: <b>{proba.max()*100:.1f}%</b></div>',
                unsafe_allow_html=True)
            fig = px.bar(x=proba.values * 100, y=proba.index,
                         orientation="h", color=proba.index,
                         color_discrete_map=CAUSE_COL,
                         labels={"x": "Probability (%)", "y": ""})
            fig.update_layout(showlegend=False)
            st.plotly_chart(style(fig, 260))
        else:
            st.error("model.pkl not found. Run train_model.py first.")

# ---------------- Charts ----------------
else:
    hero("Charts & Model Results",
         "Visualization, feature engineering, regression and classification outputs.")
    names = {
        "1_histogram.png": "Histogram",
        "2_boxplot.png": "Box plot",
        "3_scatter.png": "Scatter",
        "4_pairplot.png": "Pair plot",
        "5_heatmap.png": "Heatmap",
        "6_pca.png": "PCA",
        "7_residuals.png": "Residuals",
        "8_confusion.png": "Confusion matrix",
        "9_model_comparison.png": "Model comparison",
        "10_feature_importance.png": "Feature importance",
        "11_cause_confusion.png": "Cause confusion matrix"}
    notes = {
        "1_histogram.png": "Wind speed is right-skewed, as in real wind farms.",
        "2_boxplot.png": "T4, T5, T6 and T7 sit far below zero = lost power.",
        "3_scatter.png": "Faulty T7 plateaus near 1300 kW instead of 2000 kW.",
        "4_pairplot.png": "Shows relationships between all sensor readings.",
        "5_heatmap.png": "Wind speed, power and RPM are strongly correlated.",
        "6_pca.png": "2 components capture about 94% of the variance.",
        "7_residuals.png": "Used to check regression assumptions.",
        "8_confusion.png": "Logistic regression accuracy is about 95%.",
        "9_model_comparison.png": "Three algorithms compared on accuracy, F1 and cross-validation.",
        "10_feature_importance.png": "Which sensor readings matter most to the model.",
        "11_cause_confusion.png": "How well the best model separates each fault cause."}
    avail = {k: v for k, v in names.items()
             if os.path.exists(os.path.join("charts", k))}
    tabs = st.tabs(list(avail.values()))
    for tab, f in zip(tabs, avail):
        with tab:
            left, right = st.columns([2, 1])
            left.image(os.path.join("charts", f))
            right.markdown(kpi("Insight", "", notes.get(f, "Trained ML model result.")),
                           unsafe_allow_html=True)

    c = st.columns(4)
    c[0].markdown(kpi("Simple regression", "R² 0.876", "RMSE 244.7 kW"),
                  unsafe_allow_html=True)
    c[1].markdown(kpi("Multiple regression", "R² 0.955", "RMSE 146.7 kW",
                      "#36CFC9"), unsafe_allow_html=True)
    c[2].markdown(kpi("Logistic regression", "95%", "underperformance accuracy",
                      "#FFA940"), unsafe_allow_html=True)
    if bundle is not None:
        c[3].markdown(kpi("Best cause model", bundle["name"],
                          f"Accuracy {bundle['accuracy']*100:.1f}% | "
                          f"F1 {bundle['f1']*100:.1f}%", "#00D4FF"),
                      unsafe_allow_html=True)