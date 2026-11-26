import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Advanced Nanomaterial Gas Sensor Digital Twin", layout="wide")

st.title("📡 Advanced Nanomaterial Metal-Oxide Gas Sensor Digital Twin")
st.markdown(
    "Research-grade **Chemiresistive Thin-Film & Embedded MEMS Sensor Simulator** modeling "
    "surface ionosorption ($\text{O}_2^- / \text{O}^- / \text{O}^{2-}$), grain-boundary **double-Schottky band bending** ($qV_s$), "
    "**mixed-gas interference**, **heater-pulse modulation**, and **MCU firmware EMA filtering**."
)

# ---------------------------------------------------------------------------
# Nanomaterial Sensing Layer Database
# ---------------------------------------------------------------------------
MATERIALS = {
    "SnO₂ Nanowires (n-type)": {
        "semiconductor_type": "n-type",
        "band_gap_eV": 3.60,
        "grain_size_nm": 18.0,
        "debye_length_nm": 3.2,
        "R0_ref_kohm": 45.0,
        "T_opt_C": 280.0,
        "sigma_T": 62.0,
        "Ea_eV": 0.42,
        "tau_resp_ref_s": 12.0,
        "tau_rec_ref_s": 24.0,
        "core_color": "#2563EB",
        "gases": {
            "Carbon Monoxide (CO)": {"short": "CO", "nature": "reducing", "S_max": 13.0, "beta": 0.62, "color": "#F97316", "T_peak": 295.0},
            "Ethanol Vapor (C₂H₅OH)": {"short": "EtOH", "nature": "reducing", "S_max": 29.0, "beta": 0.70, "color": "#10B981", "T_peak": 275.0},
            "Hydrogen (H₂)": {"short": "H₂", "nature": "reducing", "S_max": 19.0, "beta": 0.65, "color": "#06B6D4", "T_peak": 245.0},
            "Nitrogen Dioxide (NO₂)": {"short": "NO₂", "nature": "oxidizing", "S_max": 24.0, "beta": 0.76, "color": "#EF4444", "T_peak": 205.0},
            "Ammonia (NH₃)": {"short": "NH₃", "nature": "reducing", "S_max": 10.5, "beta": 0.58, "color": "#A855F7", "T_peak": 260.0},
        },
    },
    "ZnO Nanorods (n-type)": {
        "semiconductor_type": "n-type",
        "band_gap_eV": 3.37,
        "grain_size_nm": 24.0,
        "debye_length_nm": 4.5,
        "R0_ref_kohm": 82.0,
        "T_opt_C": 320.0,
        "sigma_T": 68.0,
        "Ea_eV": 0.48,
        "tau_resp_ref_s": 15.0,
        "tau_rec_ref_s": 30.0,
        "core_color": "#0284C7",
        "gases": {
            "Carbon Monoxide (CO)": {"short": "CO", "nature": "reducing", "S_max": 9.5, "beta": 0.58, "color": "#F97316", "T_peak": 330.0},
            "Ethanol Vapor (C₂H₅OH)": {"short": "EtOH", "nature": "reducing", "S_max": 35.0, "beta": 0.72, "color": "#10B981", "T_peak": 315.0},
            "Hydrogen (H₂)": {"short": "H₂", "nature": "reducing", "S_max": 16.0, "beta": 0.60, "color": "#06B6D4", "T_peak": 295.0},
            "Nitrogen Dioxide (NO₂)": {"short": "NO₂", "nature": "oxidizing", "S_max": 20.0, "beta": 0.68, "color": "#EF4444", "T_peak": 235.0},
            "Ammonia (NH₃)": {"short": "NH₃", "nature": "reducing", "S_max": 12.0, "beta": 0.60, "color": "#A855F7", "T_peak": 300.0},
        },
    },
    "CuO Nanoflakes (p-type)": {
        "semiconductor_type": "p-type",
        "band_gap_eV": 1.35,
        "grain_size_nm": 22.0,
        "debye_length_nm": 2.8,
        "R0_ref_kohm": 16.0,
        "T_opt_C": 240.0,
        "sigma_T": 55.0,
        "Ea_eV": 0.35,
        "tau_resp_ref_s": 10.0,
        "tau_rec_ref_s": 22.0,
        "core_color": "#7C3AED",
        "gases": {
            "Carbon Monoxide (CO)": {"short": "CO", "nature": "reducing", "S_max": 6.8, "beta": 0.52, "color": "#F97316", "T_peak": 250.0},
            "Ethanol Vapor (C₂H₅OH)": {"short": "EtOH", "nature": "reducing", "S_max": 11.5, "beta": 0.58, "color": "#10B981", "T_peak": 240.0},
            "Hydrogen (H₂)": {"short": "H₂", "nature": "reducing", "S_max": 8.5, "beta": 0.55, "color": "#06B6D4", "T_peak": 225.0},
            "Nitrogen Dioxide (NO₂)": {"short": "NO₂", "nature": "oxidizing", "S_max": 28.0, "beta": 0.78, "color": "#EF4444", "T_peak": 195.0},
            "Ammonia (NH₃)": {"short": "NH₃", "nature": "reducing", "S_max": 9.0, "beta": 0.54, "color": "#A855F7", "T_peak": 215.0},
        },
    },
    "WO₃ Nanosheets (n-type)": {
        "semiconductor_type": "n-type",
        "band_gap_eV": 2.75,
        "grain_size_nm": 20.0,
        "debye_length_nm": 3.8,
        "R0_ref_kohm": 110.0,
        "T_opt_C": 220.0,
        "sigma_T": 58.0,
        "Ea_eV": 0.45,
        "tau_resp_ref_s": 13.0,
        "tau_rec_ref_s": 26.0,
        "core_color": "#0D9488",
        "gases": {
            "Carbon Monoxide (CO)": {"short": "CO", "nature": "reducing", "S_max": 7.5, "beta": 0.54, "color": "#F97316", "T_peak": 260.0},
            "Ethanol Vapor (C₂H₅OH)": {"short": "EtOH", "nature": "reducing", "S_max": 16.0, "beta": 0.64, "color": "#10B981", "T_peak": 250.0},
            "Hydrogen (H₂)": {"short": "H₂", "nature": "reducing", "S_max": 14.0, "beta": 0.62, "color": "#06B6D4", "T_peak": 230.0},
            "Nitrogen Dioxide (NO₂)": {"short": "NO₂", "nature": "oxidizing", "S_max": 38.0, "beta": 0.80, "color": "#EF4444", "T_peak": 190.0},
            "Ammonia (NH₃)": {"short": "NH₃", "nature": "reducing", "S_max": 18.5, "beta": 0.66, "color": "#A855F7", "T_peak": 210.0},
        },
    },
}

# ---------------------------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------------------------
st.sidebar.header("1. Nanomaterial Architecture")
selected_mat = st.sidebar.selectbox("Sensing Thin Film", list(MATERIALS.keys()), index=0)
mat = MATERIALS[selected_mat]

grain_nm = st.sidebar.slider("Crystallite Grain Size D (nm)", 6.0, 60.0, float(mat["grain_size_nm"]), 1.0)
noble_catalyst = st.sidebar.selectbox(
    "Surface Sensitization / Decoration",
    ["Pristine (Unmodified)", "Pd Nanoparticles (1 wt% — H₂/CO Boost)", "Au Nanoparticles (1 wt% — VOC/EtOH Boost)", "Pt Clusters (1 wt% — Low-Temp Activation)"],
    index=0,
)

st.sidebar.header("2. Mixed-Gas Chamber Environment")
gas_names = list(mat["gases"].keys())
target_gas_name = st.sidebar.selectbox("Primary Target Gas", gas_names, index=1)
target_gas = mat["gases"][target_gas_name]
target_ppm = st.sidebar.slider("Target Gas Concentration (ppm)", 5, 500, 120, 5)

interferent_options = ["None (Pure Target Gas)"] + [g for g in gas_names if g != target_gas_name]
interferent_name = st.sidebar.selectbox("Background Interferent Gas", interferent_options, index=0)
interferent_ppm = st.sidebar.slider("Interferent Concentration (ppm)", 0, 200, 30 if interferent_name != "None (Pure Target Gas)" else 0, 5)
rh_percent = st.sidebar.slider("Ambient Relative Humidity (% RH)", 10, 90, 40, 5)

st.sidebar.header("3. MEMS Micro-Heater Mode")
heater_mode = st.sidebar.radio("Heater Drive Mode", ["Isothermal (Constant Temp)", "Dynamic Square-Wave Pulse"], index=0)
if heater_mode == "Isothermal (Constant Temp)":
    heater_temp_C = st.sidebar.slider("Heater Temperature (°C)", 100, 450, int(mat["T_opt_C"]), 5)
    t_high_C = heater_temp_C
    t_low_C = heater_temp_C
else:
    t_high_C = st.sidebar.slider("High Pulse Temp T_high (°C)", 220, 420, 310, 5)
    t_low_C = st.sidebar.slider("Low Pulse Temp T_low (°C)", 120, 260, 180, 5)
    heater_temp_C = t_high_C

st.sidebar.header("4. MCU Signal Conditioning & Firmware")
vcc = st.sidebar.selectbox("Supply Voltage V_CC (V)", [3.3, 5.0], index=0)
r_load_kohm = st.sidebar.slider("Load Resistor R_L (kΩ)", 1.0, 200.0, 47.0, 1.0)
adc_bits = st.sidebar.radio("ADC Resolution", [10, 12], index=1, format_func=lambda b: f"{b}-bit ({2**b} levels)")
ema_alpha = st.sidebar.slider("Firmware EMA Filter α (1.0 = Raw)", 0.05, 1.0, 0.25, 0.05)
alarm_delta_v = st.sidebar.slider("Firmware Alarm Threshold ΔV (V)", 0.10, 1.50, 0.45, 0.05)

# ---------------------------------------------------------------------------
# Physics Engine: Catalyst Gain, Surface Ionosorption & Band Bending
# ---------------------------------------------------------------------------
k_B_eV = 8.617333e-5
lambda_D = mat["debye_length_nm"]
debye_ratio = grain_nm / (2.0 * lambda_D)
size_gain = (mat["grain_size_nm"] / grain_nm) ** 0.88
humidity_factor = 1.0 - 0.0024 * (rh_percent - 40.0)

# Catalyst spillover & electronic sensitization multipliers
cat_gain_map = {
    "Pristine (Unmodified)": {"CO": 1.0, "EtOH": 1.0, "H₂": 1.0, "NO₂": 1.0, "NH₃": 1.0, "t_shift": 0.0},
    "Pd Nanoparticles (1 wt% — H₂/CO Boost)": {"CO": 1.75, "EtOH": 1.15, "H₂": 2.40, "NO₂": 0.90, "NH₃": 1.05, "t_shift": -35.0},
    "Au Nanoparticles (1 wt% — VOC/EtOH Boost)": {"CO": 1.40, "EtOH": 2.15, "H₂": 1.10, "NO₂": 1.05, "NH₃": 1.25, "t_shift": -25.0},
    "Pt Clusters (1 wt% — Low-Temp Activation)": {"CO": 1.95, "EtOH": 1.65, "H₂": 2.10, "NO₂": 1.15, "NH₃": 1.30, "t_shift": -45.0},
}
cat_cfg = cat_gain_map[noble_catalyst]

def calc_R_air(temp_c):
    tk = temp_c + 273.15
    t_opt_k = mat["T_opt_C"] + 273.15
    return mat["R0_ref_kohm"] * np.exp((mat["Ea_eV"] / k_B_eV) * (1.0 / tk - 1.0 / t_opt_k)) * humidity_factor

def calc_gas_response_factor(g_dict, temp_c, ppm_val):
    if ppm_val <= 0:
        return 1.0
    shifted_peak = g_dict["T_peak"] + cat_cfg["t_shift"]
    t_eff = np.exp(-0.5 * ((temp_c - shifted_peak) / mat["sigma_T"]) ** 2)
    cat_mult = cat_cfg[g_dict["short"]]
    return 1.0 + (g_dict["S_max"] * cat_mult * t_eff * size_gain * humidity_factor) * ((ppm_val / 100.0) ** g_dict["beta"])

def calc_equilibrium_resistance(temp_c, t_ppm, int_gas_dict, int_ppm):
    r_air = calc_R_air(temp_c)
    is_n = mat["semiconductor_type"] == "n-type"

    # Target gas contribution
    s_targ = calc_gas_response_factor(target_gas, temp_c, t_ppm)
    targ_reduces = (is_n and target_gas["nature"] == "reducing") or (not is_n and target_gas["nature"] == "oxidizing")
    r_after_target = (r_air / s_targ) if targ_reduces else (r_air * s_targ)

    # Background interferent gas contribution (multiplicative surface site competition)
    if int_gas_dict is not None and int_ppm > 0:
        s_int = calc_gas_response_factor(int_gas_dict, temp_c, int_ppm)
        int_reduces = (is_n and int_gas_dict["nature"] == "reducing") or (not is_n and int_gas_dict["nature"] == "oxidizing")
        r_final = (r_after_target / (1.0 + 0.65 * (s_int - 1.0))) if int_reduces else (r_after_target * (1.0 + 0.65 * (s_int - 1.0)))
    else:
        r_final = r_after_target

    return r_air, r_final

int_gas_obj = mat["gases"][interferent_name] if interferent_name != "None (Pure Target Gas)" else None

R_air_nominal, R_gas_nominal = calc_equilibrium_resistance(heater_temp_C, target_ppm, int_gas_obj, interferent_ppm)
S_effective = max(R_air_nominal / R_gas_nominal, R_gas_nominal / R_air_nominal)

T_K = heater_temp_C + 273.15
if heater_temp_C < 150:
    oxygen_species = "O₂⁻ (Molecular Chemisorption)"
elif heater_temp_C <= 300:
    oxygen_species = "O⁻ (Atomic Ionosorption — Peak Reactivity)"
else:
    oxygen_species = "O²⁻ (High-Temp Lattice Ionosorption)"

# Schottky barrier height in air (~0.65 eV baseline) and barrier shift ΔΦ_B
phi_air_eV = 0.65 * np.clip(1.0 + 0.15 * (1.0 - debye_ratio), 0.7, 1.35)
delta_phi_eV = k_B_eV * T_K * np.log(R_gas_nominal / R_air_nominal)
phi_gas_eV = max(0.08, phi_air_eV + delta_phi_eV)

# ---------------------------------------------------------------------------
# Time-Domain Transient ODE Simulation (0 to 300 seconds, dt = 0.5 s)
# ---------------------------------------------------------------------------
t = np.linspace(0, 300, 601)
dt = t[1] - t[0]
t_on = 50.0
t_off = 175.0

T_profile = np.full_like(t, float(heater_temp_C))
if heater_mode == "Dynamic Square-Wave Pulse":
    # 30-second period square wave between t_low_C and t_high_C
    T_profile = np.where((t % 30.0) < 15.0, float(t_high_C), float(t_low_C))

R_t = np.zeros_like(t)
R_t[0] = calc_equilibrium_resistance(T_profile[0], 0.0, int_gas_obj, interferent_ppm)[1]

for i in range(1, len(t)):
    curr_t = t[i]
    curr_T = T_profile[i]
    active_ppm = float(target_ppm) if (t_on <= curr_t < t_off) else 0.0
    _, R_eq = calc_equilibrium_resistance(curr_T, active_ppm, int_gas_obj, interferent_ppm)

    speedup = np.exp(0.012 * (curr_T - mat["T_opt_C"]))
    tau = (mat["tau_resp_ref_s"] if active_ppm > 0 else mat["tau_rec_ref_s"]) / speedup
    tau = np.clip(tau, 1.8, 90.0)

    # First-order relaxation toward instantaneous equilibrium resistance
    R_t[i] = R_t[i -