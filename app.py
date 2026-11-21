import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Nanomaterial Gas Sensor Digital Twin", layout="wide")

st.title("📡 Nanomaterial Metal-Oxide Gas Sensor Digital Twin")
st.markdown(
    "Interactive **Chemiresistive Thin-Film Sensor & Embedded ADC Simulator** with animated "
    "**nanograin depletion-layer dynamics**, real-time **transient response playback**, and "
    "microcontroller signal conditioning."
)

# ---------------------------------------------------------------------------
# Nanomaterial Sensing Layer Database
# ---------------------------------------------------------------------------
MATERIALS = {
    "SnO₂ Nanowires (n-type)": {
        "semiconductor_type": "n-type",
        "grain_size_nm": 18.0,
        "debye_length_nm": 3.2,
        "R0_ref_kohm": 45.0,
        "T_opt_C": 280.0,
        "sigma_T": 65.0,
        "Ea_eV": 0.42,
        "tau_resp_ref_s": 14.0,
        "tau_rec_ref_s": 28.0,
        "core_color": "#2563EB",
        "gases": {
            "Carbon Monoxide (CO) — Reducing": {"nature": "reducing", "S_max": 12.5, "beta": 0.62, "color": "#F97316"},
            "Ethanol Vapor (C₂H₅OH) — Reducing": {"nature": "reducing", "S_max": 28.0, "beta": 0.70, "color": "#10B981"},
            "Hydrogen (H₂) — Reducing": {"nature": "reducing", "S_max": 18.0, "beta": 0.65, "color": "#06B6D4"},
            "Nitrogen Dioxide (NO₂) — Oxidizing": {"nature": "oxidizing", "S_max": 22.0, "beta": 0.75, "color": "#EF4444"},
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
        "core_color": "#0284C7",
        "gases": {
            "Carbon Monoxide (CO) — Reducing": {"nature": "reducing", "S_max": 9.0, "beta": 0.58, "color": "#F97316"},
            "Ethanol Vapor (C₂H₅OH) — Reducing": {"nature": "reducing", "S_max": 34.0, "beta": 0.72, "color": "#10B981"},
            "Hydrogen (H₂) — Reducing": {"nature": "reducing", "S_max": 15.0, "beta": 0.60, "color": "#06B6D4"},
            "Nitrogen Dioxide (NO₂) — Oxidizing": {"nature": "oxidizing", "S_max": 19.0, "beta": 0.68, "color": "#EF4444"},
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
        "core_color": "#7C3AED",
        "gases": {
            "Carbon Monoxide (CO) — Reducing": {"nature": "reducing", "S_max": 6.5, "beta": 0.52, "color": "#F97316"},
            "Ethanol Vapor (C₂H₅OH) — Reducing": {"nature": "reducing", "S_max": 11.0, "beta": 0.58, "color": "#10B981"},
            "Hydrogen (H₂) — Reducing": {"nature": "reducing", "S_max": 8.0, "beta": 0.55, "color": "#06B6D4"},
            "Nitrogen Dioxide (NO₂) — Oxidizing": {"nature": "oxidizing", "S_max": 26.0, "beta": 0.78, "color": "#EF4444"},
        },
    },
}

# ---------------------------------------------------------------------------
# Sidebar Controls
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
)

st.sidebar.header("2. Gas Exposure & Micro-Heater")
selected_gas = st.sidebar.selectbox("Target Gas Species", list(mat["gases"].keys()), index=1)
gas_info = mat["gases"][selected_gas]

gas_ppm = st.sidebar.slider("Gas Concentration (ppm)", min_value=5, max_value=500, value=120, step=5)
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

humidity_factor = 1.0 - 0.0022 * (rh_percent - 40.0)
R_air_kohm = mat["R0_ref_kohm"] * np.exp((mat["Ea_eV"] / k_B_eV) * (1.0 / T_K - 1.0 / T_opt_K)) * humidity_factor

lambda_D = mat["debye_length_nm"]
size_gain = (mat["grain_size_nm"] / grain_nm) ** 0.85
thermal_efficiency = np.exp(-0.5 * ((heater_temp_C - mat["T_opt_C"]) / mat["sigma_T"]) ** 2)

S_magnitude = 1.0 + (gas_info["S_max"] * thermal_efficiency * size_gain * humidity_factor) * ((gas_ppm / 100.0) ** gas_info["beta"])

is_n_type = mat["semiconductor_type"] == "n-type"
is_reducing = gas_info["nature"] == "reducing"
resistance_drops = (is_n_type and is_reducing) or (not is_n_type and not is_reducing)

if resistance_drops:
    R_gas_kohm = R_air_kohm / S_magnitude
    direction_label = "Resistance Decreases"
else:
    R_gas_kohm = R_air_kohm * S_magnitude
    direction_label = "Resistance Increases"

speedup = np.exp(0.012 * (heater_temp_C - mat["T_opt_C"]))
tau_resp = np.clip(mat["tau_resp_ref_s"] / speedup, 2.0, 90.0)
tau_rec = np.clip(mat["tau_rec_ref_s"] / speedup, 4.0, 150.0)

t90_resp_s = 2.3026 * tau_resp
t90_rec_s = 2.3026 * tau_rec

# ---------------------------------------------------------------------------
# Time-Domain Transient Simulation (0 to 300 seconds)
# ---------------------------------------------------------------------------
t = np.linspace(0, 300, 601)
t_on = 50.0
t_off = 170.0

R_t = np.full_like(t, R_air_kohm)
mask_exp = (t >= t_on) & (t < t_off)
R_t[mask_exp] = R_gas_kohm + (R_air_kohm - R_gas_kohm) * np.exp(-(t[mask_exp] - t_on) / tau_resp)

R_at_off = R_gas_kohm + (R_air_kohm - R_gas_kohm) * np.exp(-(t_off - t_on) / tau_resp)
mask_rec = t >= t_off
R_t[mask_rec] = R_air_kohm + (R_at_off - R_air_kohm) * np.exp(-(t[mask_rec] - t_off) / tau_rec)

rng = np.random.default_rng(42)
if add_noise:
    R_t = R_t * (1.0 + rng.normal(0.0, 0.006, size=len(t)))
    R_t = np.maximum(R_t, 0.05)

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
# ANIMATION 1: Live Nanoscale Grain-Boundary Depletion & Gas Diffusion Chamber
# ---------------------------------------------------------------------------
st.subheader("🔬 Live Animated Nanograin Depletion Layer & Gas Chamber Simulation")
st.caption("Click **▶ Play** below to watch target gas molecules enter the chamber, react with chemisorbed oxygen ions (O⁻/O²⁻), and modulate the grain-boundary depletion shells & electron conduction bridges.")

# Fixed grid of 6 nanograins inside a [0, 10] x [0, 5] micro-chamber
grain_centers = np.array([
    [2.2, 1.5], [5.0, 1.5], [7.8, 1.5],
    [2.2, 3.5], [5.0, 3.5], [7.8, 3.5],
])
grain_radius = 0.85

# Pre-generate 36 gas molecule trajectories drifting from left inlet to right exhaust
n_molecules = 36
mol_rng = np.random.default_rng(101)
mol_x0 = mol_rng.uniform(-4.0, 9.5, n_molecules)
mol_y0 = mol_rng.uniform(0.35, 4.65, n_molecules)
mol_vy = mol_rng.uniform(-0.18, 0.18, n_molecules)
mol_vx = mol_rng.uniform(0.32, 0.55, n_molecules)

# Generate 36 animation frames spanning t = 0 s to t = 300 s
frame_indices = np.linspace(0, len(t) - 1, 36, dtype=int)
anim_frames = []

for f_idx, ti in enumerate(frame_indices):
    curr_t = t[ti]
    curr_R = R_t[ti]
    curr_V = V_quantized[ti]
    gas_active = (curr_t >= t_on) and (curr_t < t_off)

    # Normalized conductance state (0 = high resistance air, 1 = peak gas response)
    if resistance_drops:
        conductance_norm = np.clip((R_air_kohm - curr_R) / max(R_air_kohm - R_gas_kohm, 1e-3), 0.0, 1.0)
    else:
        conductance_norm = np.clip(1.0 - (curr_R - R_air_kohm) / max(R_gas_kohm - R_air_kohm, 1e-3), 0.0, 1.0)

    # Depletion shell thickness shrinks when conductance is high, expands when resistance is high
    depletion_width = 0.38 * (1.0 - 0.72 * conductance_norm)
    shell_radius = grain_radius + depletion_width

    # Gas particle positions (visible when gas valve is ON or purging)
    if gas_active:
        mx = (mol_x0 + f_idx * mol_vx) % 10.2
        my = np.clip(mol_y0 + 0.35 * np.sin(f_idx * 0.4 + mol_vy * 10), 0.25, 4.75)
        n_vis = max(6, int(n_molecules * min(gas_ppm / 250.0, 1.0)))
        mx_vis, my_vis = mx[:n_vis], my[:n_vis]
        valve_status = f"GAS VALVE: OPEN ({gas_ppm} ppm)"
    elif curr_t >= t_off and curr_t < t_off + 35:
        fade = max(0.0, 1.0 - (curr_t - t_off) / 35.0)
        n_vis = int(n_molecules * 0.5 * fade)
        mx_vis = (mol_x0[:n_vis] + f_idx * mol_vx[:n_vis]) % 10.2
        my_vis = mol_y0[:n_vis]
        valve_status = "CLEAN AIR PURGE"
    else:
        mx_vis, my_vis = [], []
        valve_status = "CLEAN AIR BASELINE"

    # Build circle polygons for depletion shells & semiconducting cores
    theta_c = np.linspace(0, 2 * np.pi, 40)
    shell_x, shell_y = [], []
    core_x, core_y = [], []
    for gx, gy in grain_centers:
        shell_x.extend(list(gx + shell_radius * np.cos(theta_c)) + [None])
        shell_y.extend(list(gy + shell_radius * np.sin(theta_c)) + [None])
        core_x.extend(list(gx + grain_radius * np.cos(theta_c)) + [None])
        core_y.extend(list(gy + grain_radius * np.sin(theta_c)) + [None])

    # Conduction channel bridges between adjacent nanograins
    bridge_pairs = [(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)]
    bx, by = [], []
    for i, j in bridge_pairs:
        bx.extend([grain_centers[i, 0], grain_centers[j, 0], None])
        by.extend([grain_centers[i, 1], grain_centers[j, 1], None])

    bridge_width = 2.0 + 9.0 * conductance_norm
    bridge_opacity = 0.20 + 0.75 * conductance_norm

    frame_data = [
        # Trace 0: Electron conduction bridges
        go.Scatter(
            x=bx, y=by, mode="lines",
            name="Conduction Channels",
            line=dict(color=f"rgba(16, 185, 129, {bridge_opacity:.2f})", width=bridge_width),
            hoverinfo="skip",
        ),
        # Trace 1: Depletion layer shells
        go.Scatter(
            x=shell_x, y=shell_y, mode="lines", fill="toself",
            fillcolor="rgba(245, 158, 11, 0.24)",
            line=dict(color="#F59E0B", width=1.5, dash="dot"),
            name="Electron Depletion Layer (λ_D)",
            hoverinfo="skip",
        ),
        # Trace 2: Nanograin cores
        go.Scatter(
            x=core_x, y=core_y, mode="lines", fill="toself",
            fillcolor=mat["core_color"],
            line=dict(color="#1E293B", width=1.5),
            name=f"{selected_mat.split()[0]} Nanograins",
            hoverinfo="skip",
        ),
        # Trace 3: Diffusing target gas molecules
        go.Scatter(
            x=mx_vis, y=my_vis, mode="markers",
            name="Target Gas Molecules",
            marker=dict(size=10, color=gas_info["color"], line=dict(width=1, color="#FFFFFF")),
            hoverinfo="skip",
        ),
        # Trace 4: Live OLED telemetry HUD readout inside chamber
        go.Scatter(
            x=[5.0], y=[4.72], mode="text",
            text=[
                f"<b>t = {curr_t:5.1f} s  |  {valve_status}  |  R_s = {curr_R:6.2f} kΩ  |  ADC V_out = {curr_V:.3f} V</b>"
            ],
            textfont=dict(size=13, color="#0F172A"),
            showlegend=False,
            hoverinfo="skip",
        ),
    ]
    anim_frames.append(go.Frame(data=frame_data, name=f"{int(curr_t)}s"))

fig_chamber = go.Figure(
    data=anim_frames[0].data,
    layout=go.Layout(
        height=340,
        margin=dict(l=15, r=15, t=20, b=15),
        xaxis=dict(range=[0, 10], showgrid=False, zeroline=False, showticklabels=False, title="Gas Flow Inlet → Sensing Thin Film → Exhaust"),
        yaxis=dict(range=[0, 5.1], showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="#F8FAFC",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        shapes=[
            # Chamber border & bottom micro-heater plate
            dict(type="rect", x0=0.05, y0=0.25, x1=9.95, y1=4.45, line=dict(color="#94A3B8", width=2)),
            dict(type="rect", x0=0.8, y0=0.05, x1=9.2, y1=0.25, fillcolor="#EF4444", line=dict(width=0)),
        ],
        annotations=[
            dict(
                x=5.0, y=0.14, text=f"Micro-Heater Substrate ({heater_temp_C} °C)",
                showarrow=False, font=dict(color="white", size=10),
            )
        ],
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.0,
                y=-0.16,
                xanchor="left",
                buttons=[
                    dict(
                        label="▶ Play Simulation",
                        method="animate",
                        args=[None, {"frame": {"duration": 140, "redraw": True}, "fromcurrent": True, "transition": {"duration": 60}}],
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                x=0.22,
                y=-0.13,
                len=0.78,
                currentvalue={"prefix": "Timeline: ", "visible": True, "xanchor": "right"},
                steps=[
                    dict(
                        method="animate",
                        args=[[f.name], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                        label=f.name,
                    )
                    for f in anim_frames
                ],
            )
        ],
    ),
    frames=anim_frames,
)

st.plotly_chart(fig_chamber, use_container_width=True)

# ---------------------------------------------------------------------------
# ANIMATION 2: Real-Time Resistance & MCU ADC Transient + Calibration Curve
# ---------------------------------------------------------------------------
col_main, col_cal = st.columns([1.3, 1.0])

with col_main:
    st.subheader("Sensor Resistance & MCU ADC Transient Curve")
    fig_trans = make_subplots(specs=[[{"secondary_y": True}]])

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
        height=410,
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
    st.subheader("Sensitivity vs. Gas Concentration")
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
        height=410,
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