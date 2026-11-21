import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Nanomaterial Gas Sensor Digital Twin", layout="wide")

st.title("📡 Nanomaterial Metal-Oxide Gas Sensor Digital Twin")
st.markdown(
    "Interactive **Chemiresistive Thin-Film Sensor & Embedded ADC Simulator** modeling "
    "surface depletion layer physics, Arrhenius thermal activation, Freundlich power-law response, "
    "and microcontroller voltage-divider signal conditioning."
)

# ---------------------------------------------------------------------------
# Nanomaterial Sensing Layer Database
# ---------------------------------------------------------------------------
MATERIALS = {
    "SnO₂ Nanowires (n-type)": {
        "semiconductor_type": "n-type",
        "grain_size_nm": 18.0,
        "debye_length_nm": 3.2,
        "R0_ref_kohm": 45.0,       # Baseline resistance in clean air at T_opt (kOhm)
        "T_opt_C": 280.0,         # Optimal micro-heater temperature (°C)
        "sigma_T": 65.0,          # Thermal sensitivity bell-width (°C)
        "Ea_eV": 0.42,            # Activation energy for baseline resistance (eV)
        "tau_resp_ref_s": 14.0,   # Characteristic response time at T_opt (s)
        "tau_rec_ref_s": 28.0,    # Characteristic recovery time at T_opt (s)
        "gases": {
            "Carbon Monoxide (CO) — Reducing": {"nature": "reducing", "S_max": 12.5, "beta": 0.62},
            "Ethanol Vapor (C₂H₅OH) — Reducing": {"nature": "reducing", "S_max": 28.0, "beta": 0.70},
            "Hydrogen (H₂) — Reducing": {"nature": "reducing", "S_max": 18.0, "beta": 0.65},
            "Nitrogen Dioxide (NO₂) — Oxidizing": {"nature": "oxidizing", "S_max": 22.0, "beta": 0.75},
        },
    },
    "ZnO Nanorods (n-type)": {
        "semiconductor_type": "n-type",
        "grain_size_nm": 24.0,
        "debye_length_nm": 4.5,
        "R0_ref_kohm": 82.0,
        "T_opt_C": 320.0,
        "sigma_T": 70.0,
        "Ea_eV": 0.48,
        "tau_resp_ref_s": 18.0,
        "tau_rec_ref_s": 35.0,
        "gases": {
            "Carbon Monoxide (CO) — Reducing": {"nature": "reducing", "S_max": 9.0, "beta": 0.58},
            "Ethanol Vapor (C₂H₅OH) — Reducing": {"nature": "reducing", "S_max": 34.0, "beta": 0.72},
            "Hydrogen (H₂) — Reducing": {"nature": "reducing", "S_max": 15.0, "beta": 0.60},
            "Nitrogen Dioxide (NO₂) — Oxidizing": {"nature": "oxidizing", "S_max": 19.0, "beta": 0.68},
        },
    },
    "CuO Nanoflakes (p-type)": {
        "semiconductor_type": "p-type",
        "grain_size_nm": 22.0,
        "debye_length_nm": 2.8,
        "R0_ref_kohm": 16.0,
        "T_opt_C": 240.0,
        "sigma_T": 55.0,
        "Ea_eV": 0.35,
        "tau_resp_ref_s": 12.0,
        "tau_rec_ref_s": 24.0,
        "gases": {
            "Carbon Monoxide (CO) — Reducing": {"nature": "reducing", "S_max": 6.5, "beta": 0.52},
            "Ethanol Vapor (C₂H₅OH) — Reducing": {"nature": "reducing", "S_max": 11.0, "beta": 0.58},
            "Hydrogen (H₂) — Reducing": {"nature": "reducing", "S_max": 8.0, "beta": 0.55},
            "Nitrogen Dioxide (NO₂) — Oxidizing": {"nature": "oxidizing", "S_max": 26.0, "beta": 0.78},
        },
    },
}

# ---------------------------------------------------------------------------
# Sidebar Controls: Nanomaterial, Environment & Electronics Interface
# ---------------------------------------------------------------------------
st.sidebar.header("1. Nanomaterial Sensing Layer")
selected_mat = st.sidebar.selectbox("Thin-Film Nanostructure", list(MATERIALS.keys()), index=0)
mat = MATERIALS[selected_mat]

grain_nm = st.sidebar.slider(
    "Nanocrystallite Grain Size D (nm)",
    min_value=6.0,
    max_value=60.0,
    value=float(mat["grain_size_nm"]),
    step=1.0,
    help="Smaller grains (D approaching 2 × Debye length λ_D) dramatically amplify chemiresistive sensitivity.",
)

st.sidebar.header("2. Gas Exposure & Micro-Heater")
selected_gas = st.sidebar.selectbox("Target Gas Species", list(mat["gases"].keys()), index=1)
gas_info = mat["gases"][selected_gas]

gas_ppm = st.sidebar.slider("Gas Concentration (ppm)", min_value=5, max_value=500, value=100, step=5)
heater_temp_C = st.sidebar.slider("Micro-Heater Temperature (°C)", min_value=100, max_value=450, value=int(mat["T_opt_C"]), step=5)
rh_percent = st.sidebar.slider("Ambient Relative Humidity (% RH)", min_value=10, max_value=90, value=40, step=5)

st.sidebar.header("3. Embedded Electronics & ADC")
vcc = st.sidebar.selectbox("Supply Voltage V_CC (V)", [3.3, 5.0], index=0)
r_load_kohm = st.sidebar.slider("Load Resistor R_L (kΩ)", min_value=1.0, max_value=200.0, value=47.0, step=1.0)
adc_bits = st.sidebar.radio("MCU ADC Resolution", [10, 12], index=1, format_func=lambda b: f"{b}-bit ({2**b} levels)")
add_noise = st.sidebar.checkbox("Simulate Thermal / ADC Noise", value=True)

# ---------------------------------------------------------------------------
# Physics Model: Surface Depletion, Arrhenius Baseline & Gas Sensitivity
# ---------------------------------------------------------------------------
k_B_eV = 8.617333e-5
T_K = heater_temp_C + 273.15
T_opt_K = mat["T_opt_C"] + 273.15

# 1. Baseline resistance in clean air R_a (Arrhenius temperature dependence + humidity cross-sensitivity)
humidity_factor = 1.0 - 0.0022 * (rh_percent - 40.0)
R_air_kohm = mat["R0_ref_kohm"] * np.exp((mat["Ea_eV"] / k_B_eV) * (1.0 / T_K - 1.0 / T_opt_K)) * humidity_factor

# 2. Grain-size enhancement factor (Neck/Depletion-controlled regime when D ~ 2 * lambda_D)
lambda_D = mat["debye_length_nm"]
size_gain = (mat["grain_size_nm"] / grain_nm) ** 0.85

# 3. Thermal volcano curve (adsorption vs desorption kinetics peak near T_opt)
thermal_efficiency = np.exp(-0.5 * ((heater_temp_C - mat["T_opt_C"]) / mat["sigma_T"]) ** 2)

# 4. Freundlich power-law response factor: S = 1 + A(T, D) * C^beta
S_magnitude = 1.0 + (gas_info["S_max"] * thermal_efficiency * size_gain * humidity_factor) * ((gas_ppm / 100.0) ** gas_info["beta"])

# Direction of resistance change (n-type vs p-type & reducing vs oxidizing gas)
is_n_type = mat["semiconductor_type"] == "n-type"
is_reducing = gas_info["nature"] == "reducing"

if (is_n_type and is_reducing) or (not is_n_type and not is_reducing):
    # Resistance decreases upon gas exposure (R_g = R_a / S)
    R_gas_kohm = R_air_kohm / S_magnitude
    direction_label = "Resistance Decreases (R_a / R_g)"
else:
    # Resistance increases upon gas exposure (R_g = R_a * S)
    R_gas_kohm = R_air_kohm * S_magnitude
    direction_label = "Resistance Increases (R_g / R_a)"

# 5. Arrhenius-accelerated response & recovery time constants
speedup = np.exp(0.012 * (heater_temp_C - mat["T_opt_C"]))
tau_resp = np.clip(mat["tau_resp_ref_s"] / speedup, 2.0, 90.0)
tau_rec = np.clip(mat["tau_rec_ref_s"] / speedup, 4.0, 150.0)

# t90 metrics (time to reach 90% of step change = 2.3026 * tau)
t90_resp_s = 2.3026 * tau_resp
t90_rec_s = 2.3026 * tau_rec

# ---------------------------------------------------------------------------
# Time-Domain Transient Simulation (0 to 300 seconds)
# ---------------------------------------------------------------------------
t = np.linspace(0, 300, 601)  # 0.5 s sampling
t_on = 50.0    # Gas valve opens at t = 50 s
t_off = 170.0  # Clean air purge begins at t = 170 s

R_t = np.full_like(t, R_air_kohm)

# Exposure phase
mask_exp = (t >= t_on) & (t < t_off)
R_t[mask_exp] = R_gas_kohm + (R_air_kohm - R_gas_kohm) * np.exp(-(t[mask_exp] - t_on) / tau_resp)

# Recovery phase
R_at_off = R_gas_kohm + (R_air_kohm - R_gas_kohm) * np.exp(-(t_off - t_on) / tau_resp)
mask_rec = t >= t_off
R_t[mask_rec] = R_air_kohm + (R_at_off - R_air_kohm) * np.exp(-(t[mask_rec] - t_off) / tau_rec)

# Optional realistic sensor noise
rng = np.random.default_rng(42)
if add_noise:
    R_t = R_t * (1.0 + rng.normal(0.0, 0.006, size=len(t)))
    R_t = np.maximum(R_t, 0.05)

# Voltage Divider Circuit: V_out = V_CC * (R_L / (R_sensor + R_L))
V_out = vcc * (r_load_kohm / (R_t + r_load_kohm))
adc_max = (2 ** adc_bits) - 1
adc_counts = np.round((V_out / vcc) * adc_max).astype(int)
V_quantized = (adc_counts / adc_max) * vcc

# ---------------------------------------------------------------------------
# Top KPI Metrics Row
# ---------------------------------------------------------------------------
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Semiconductor Type", f"{mat['semiconductor_type']} ({direction_label.split()[1]})")
k2.metric("Sensor Response (S)", f"{S_magnitude:.2f}×")
k3.metric("Baseline R_air → R_gas", f"{R_air_kohm:.1f} → {R_gas_kohm:.2f} kΩ")
k4.metric("Response Time (t₉₀)", f"{t90_resp_s:.1f} s")
k5.metric("Recovery Time (t₉₀)", f"{t90_rec_s:.1f} s")

# ---------------------------------------------------------------------------
# Interactive Plots: Transient Resistance/ADC & Calibration Curve
# ---------------------------------------------------------------------------
col_main, col_cal = st.columns([1.3, 1.0])

with col_main:
    st.subheader("Real-Time Sensor Resistance & MCU ADC Transient")
    fig_trans = make_subplots(specs=[[{"secondary_y": True}]])

    # Shaded gas exposure zone
    fig_trans.add_vrect(
        x0=t_on,
        x1=t_off,
        fillcolor="rgba(37, 99, 235, 0.08)",
        layer="below",
        line_width=0,
        annotation_text=f"Target Gas ON ({gas_ppm} ppm)",
        annotation_position="top left",
    )

    fig_trans.add_trace(
        go.Scatter(
            x=t,
            y=R_t,
            name="Sensor Resistance R_s (kΩ)",
            line=dict(color="#2563EB", width=2.4),
        ),
        secondary_y=False,
    )

    fig_trans.add_trace(
        go.Scatter(
            x=t,
            y=V_quantized,
            name=f"MCU ADC Voltage ({adc_bits}-bit)",
            line=dict(color="#EA580C", width=2.0, dash="dot"),
        ),
        secondary_y=True,
    )

    fig_trans.update_layout(
        height=440,
        margin=dict(l=10, r=10, t=30, b=10),
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig_trans.update_xaxes(title_text="Time (seconds)")
    fig_trans.update_yaxes(title_text="Sensor Resistance R_s (kΩ)", secondary_y=False)
    fig_trans.update_yaxes(title_text="ADC Input Voltage V_out (V)", range=[0, vcc * 1.05], secondary_y=True)
    st.plotly_chart(fig_trans, use_container_width=True)

with col_cal:
    st.subheader("Sensitivity vs. Gas Concentration & Heater Temp")
    ppm_grid = np.linspace(5, 500, 100)
    S_curve = 1.0 + (gas_info["S_max"] * thermal_efficiency * size_gain * humidity_factor) * ((ppm_grid / 100.0) ** gas_info["beta"])

    fig_cal = go.Figure()
    fig_cal.add_trace(
        go.Scatter(
            x=ppm_grid,
            y=S_curve,
            mode="lines",
            name=f"T = {heater_temp_C} °C",
            line=dict(color="#0D9488", width=2.5),
        )
    )
    fig_cal.add_trace(
        go.Scatter(
            x=[gas_ppm],
            y=[S_magnitude],
            mode="markers+text",
            name="Operating Point",
            text=[f"  ({gas_ppm} ppm, S={S_magnitude:.1f})"],
            textposition="middle right",
            marker=dict(size=11, color="#DC2626"),
        )
    )
    fig_cal.update_layout(
        height=440,
        margin=dict(l=10, r=10, t=30, b=10),
        template="plotly_white",
        xaxis_title="Gas Concentration (ppm)",
        yaxis_title="Sensitivity S (R_a/R_g or R_g/R_a)",
        showlegend=False,
    )
    st.plotly_chart(fig_cal, use_container_width=True)

# ---------------------------------------------------------------------------
# Electronics Interface Summary & CSV Telemetry Export
# ---------------------------------------------------------------------------
st.subheader("Embedded Hardware & Signal Conditioning Summary")
e1, e2, e3, e4 = st.columns(4)
e1.metric("Clean Air V_out", f"{V_out[0]:.3f} V (ADC: {adc_counts[0]})")
e2.metric("Steady-State Gas V_out", f"{V_out[np.argmin(np.abs(t - 165))]:.3f} V (ADC: {adc_counts[np.argmin(np.abs(t - 165))]})")
e3.metric("Voltage Swing ΔV", f"{abs(V_out[np.argmin(np.abs(t - 165))] - V_out[0]):.3f} V")
e4.metric("Debye Ratio (D / 2λ_D)", f"{grain_nm / (2.0 * lambda_D):.2f}")

export_df = pd.DataFrame({
    "Time_s": np.round(t, 2),
    "Resistance_kOhm": np.round(R_t, 4),
    "Vout_V": np.round(V_quantized, 4),
    "ADC_Code": adc_counts,
})
st.download_button(
    label="📥 Download Simulated Sensor & ADC Telemetry (CSV)",
    data=export_df.to_csv(index=False).encode("utf-8"),
    file_name=f"gas_sensor_twin_{selected_mat.split()[0].lower()}.csv",
    mime="text/csv",
)