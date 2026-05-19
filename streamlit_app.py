import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from scipy.integrate import solve_ivp

st.set_page_config(page_title="DO Crash Explorer", layout="wide")
st.title("Dissolved Oxygen (DO) Crash Explorer: Oxygen Transfer (OTR) vs Oxygen Uptake (OUR)")

with st.sidebar:
    st.header("Mass Transfer")
    kLa = st.slider("Volumetric Mass Transfer Coefficient (1/h)", 0.0, 500.0, 120.0, 1.0)
    Cstar = st.slider("Saturated DO Concentration (mM)", 0.0, 0.50, 0.25, 0.01)
    C0 = st.slider("Initial DO Concentration (mM)", 0.0, 0.50, 0.20, 0.01)

    st.header("Oxygen Demand")
    OUR = st.slider("Oxygen Usage Rate (mM/h)", 0.0, 50.0, 10.0, 0.1)

    st.header("Simulation")
    tf = st.slider("End Time (h)", 0.1, 20.0, 6.0, 0.1)
    n = st.slider("Time Points", 100, 2000, 600, 50)
    DO_min = st.slider("Minimum Safe DO Concentration (mM)", 0.0, 0.30, 0.05, 0.01)

def rhs(t, y):
    C = y[0]
    return [kLa * (Cstar - C) - OUR]

t_eval = np.linspace(0, tf, n)

sol = solve_ivp(
    fun=rhs,
    t_span=(0.0, tf),
    y0=[C0],
    t_eval=t_eval,
    method="LSODA",
)

if not sol.success:
    st.error(sol.message)
    st.stop()

t_h = sol.t
C = sol.y[0]

show_minutes = st.sidebar.checkbox("Show time in minutes", value=True)
t_plot = 60.0 * t_h if show_minutes else t_h
t_label = "t (min)" if show_minutes else "t (h)"

OTR = kLa * (Cstar - C)
OUR_vec = np.full_like(t_h, OUR)

df = pd.DataFrame({
    t_label: t_plot,
    "C (mM)": C,
    "OTR (mM/h)": OTR,
    "OUR (mM/h)": OUR_vec
})

# This allows the label and plot to be dynamic of the user's choosing.

# Metrics
C_min = float(C.min())
time_below = float(np.trapezoid((C < DO_min).astype(float), t_h)) # integrate in hours

C_ss = np.nan
if kLa > 1e-12:
    C_ss = Cstar - OUR / kLa  # steady-state if it exists

col1, col2 = st.columns(2)

with col1:
    fig1 = px.line(df, x=t_label, y="C (mM)", title="Dissolved Oxygen Over Time")
    fig1.add_hline(y=DO_min, line_dash="dash")

    fig1.update_layout(
        title_font_size=20,
        title_x=0.25, # Centers the chart title
        font_family="Computer Modern", 
        font_color="black",
        
        xaxis=dict(
            title_font=dict(color="black"),
            tickfont=dict(color="black")
        ),
        
        yaxis=dict(
            title_font=dict(color="black"),
            tickfont=dict(color="black")
        )
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.line(df, x=t_label, y=["OTR (mM/h)", "OUR (mM/h)"], title="Oxygen Rates Over Time", labels={"value": "Rate (mM/h)", "variable": "Oxygen Rate"})
    
    fig2.update_layout(
        title_font_size=20,
        title_x=0.25, # Centers the chart title
        font_family="Computer Modern",  
        font_color="black",
        
        legend=dict(
            orientation="h", # Horizontal layout
            yanchor="bottom",
            y=0.2,                 
            xanchor="right",
            x=1
            ),
        
        xaxis=dict(
            title_font=dict(color="black"),
            tickfont=dict(color="black")
        ),
    
        yaxis=dict(
            title_font=dict(color="black"),
            tickfont=dict(color="black")
        )
    )
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("Results")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Safe DO Threshold", f"{C_min:.2f} mM")
c2.metric("Hours below DO threshold", f"{time_below:.2f} hr")
c3.metric("Predicted steady-state DO", f"{C_ss:.2f} mM")
c4.metric("Initial OTR", f"{(kLa*(Cstar-C0)):.2f} mM/h")

if C_ss < 0:
    st.warning("Steady-state's dissolved oxygen is negative: oxygen demand exceeds transfer capacity.")
elif C_ss < DO_min:
    st.warning("Steady-state's dissolved oxygen is below dissolved oxygen threshold: biological population decay expected.")
else:
    st.success("Steady-state's dissolved oxygen is above dissolved oxygen threshold under constant conditions.")
