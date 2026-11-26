import json
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="3D Nanomaterial Gas Sensor Digital Twin", layout="wide")

st.title("🧊 3D Nanomaterial Gas Sensor & MEMS Digital Twin")
st.markdown(
    "Cross-platform **3D Chemiresistive Sensor & Embedded Hardware Simulator** — drag or touch to rotate 360°, "
    "scroll or pinch to zoom, and inspect real-time **3D gas diffusion**, **depletion shell ($\\lambda_D$) "
    "modulation**, and **MCU ADC signal conditioning**."
)

# ---------------------------------------------------------------------------
# Nanomaterial Sensing Layer Database
# ---------------------------------------------------------------------------
MATERIALS = {
    "SnO₂ Nanowire Network (n-type)": {
        "type": "n-type",
        "eg_eV": 3.60,
        "grain_nm": 18.0,
        "debye_nm": 3.2,
        "R0_kohm": 45.0,
        "T_opt": 280.0,
        "sigma_T": 62.0,
        "Ea_eV": 0.42,
        "tau_resp": 12.0,
        "tau_rec": 24.0,
        "core_hex": "#2563EB",
        "gases": {
            "Ethanol Vapor (C₂H₅OH)": {"short": "EtOH", "nature": "reducing", "S_max": 29.0, "beta": 0.70, "hex": "#10B981", "T_peak": 275.0},
            "Carbon Monoxide (CO)": {"short": "CO", "nature": "reducing", "S_max": 13.0, "beta": 0.62, "hex": "#F97316", "T_peak": 295.0},
            "Hydrogen (H₂)": {"short": "H₂", "nature": "reducing", "S_max": 19.0, "beta": 0.65, "hex": "#06B6D4", "T_peak": 245.0},
            "Nitrogen Dioxide (NO₂)": {"short": "NO₂", "nature": "oxidizing", "S_max": 24.0, "beta": 0.76, "hex": "#EF4444", "T_peak": 205.0},
        },
    },
    "ZnO Nanorod Array (n-type)": {
        "type": "n-type",
        "eg_eV": 3.37,
        "grain_nm": 24.0,
        "debye_nm": 4.5,
        "R0_kohm": 82.0,
        "T_opt": 320.0,
        "sigma_T": 68.0,
        "Ea_eV": 0.48,
        "tau_resp": 15.0,
        "tau_rec": 30.0,
        "core_hex": "#0284C7",
        "gases": {
            "Ethanol Vapor (C₂H₅OH)": {"short": "EtOH", "nature": "reducing", "S_max": 35.0, "beta": 0.72, "hex": "#10B981", "T_peak": 315.0},
            "Carbon Monoxide (CO)": {"short": "CO", "nature": "reducing", "S_max": 9.5, "beta": 0.58, "hex": "#F97316", "T_peak": 330.0},
            "Hydrogen (H₂)": {"short": "H₂", "nature": "reducing", "S_max": 16.0, "beta": 0.60, "hex": "#06B6D4", "T_peak": 295.0},
            "Nitrogen Dioxide (NO₂)": {"short": "NO₂", "nature": "oxidizing", "S_max": 20.0, "beta": 0.68, "hex": "#EF4444", "T_peak": 235.0},
        },
    },
    "CuO Nanoparticle Film (p-type)": {
        "type": "p-type",
        "eg_eV": 1.35,
        "grain_nm": 22.0,
        "debye_nm": 2.8,
        "R0_kohm": 16.0,
        "T_opt": 240.0,
        "sigma_T": 55.0,
        "Ea_eV": 0.35,
        "tau_resp": 10.0,
        "tau_rec": 22.0,
        "core_hex": "#7C3AED",
        "gases": {
            "Ethanol Vapor (C₂H₅OH)": {"short": "EtOH", "nature": "reducing", "S_max": 11.5, "beta": 0.58, "hex": "#10B981", "T_peak": 240.0},
            "Carbon Monoxide (CO)": {"short": "CO", "nature": "reducing", "S_max": 6.8, "beta": 0.52, "hex": "#F97316", "T_peak": 250.0},
            "Hydrogen (H₂)": {"short": "H₂", "nature": "reducing", "S_max": 8.5, "beta": 0.55, "hex": "#06B6D4", "T_peak": 225.0},
            "Nitrogen Dioxide (NO₂)": {"short": "NO₂", "nature": "oxidizing", "S_max": 28.0, "beta": 0.78, "hex": "#EF4444", "T_peak": 195.0},
        },
    },
    "WO₃ Nanosheet Cluster (n-type)": {
        "type": "n-type",
        "eg_eV": 2.75,
        "grain_nm": 20.0,
        "debye_nm": 3.8,
        "R0_kohm": 110.0,
        "T_opt": 220.0,
        "sigma_T": 58.0,
        "Ea_eV": 0.45,
        "tau_resp": 13.0,
        "tau_rec": 26.0,
        "core_hex": "#0D9488",
        "gases": {
            "Ethanol Vapor (C₂H₅OH)": {"short": "EtOH", "nature": "reducing", "S_max": 16.0, "beta": 0.64, "hex": "#10B981", "T_peak": 250.0},
            "Carbon Monoxide (CO)": {"short": "CO", "nature": "reducing", "S_max": 7.5, "beta": 0.54, "hex": "#F97316", "T_peak": 260.0},
            "Hydrogen (H₂)": {"short": "H₂", "nature": "reducing", "S_max": 14.0, "beta": 0.62, "hex": "#06B6D4", "T_peak": 230.0},
            "Nitrogen Dioxide (NO₂)": {"short": "NO₂", "nature": "oxidizing", "S_max": 38.0, "beta": 0.80, "hex": "#EF4444", "T_peak": 190.0},
        },
    },
}

# ---------------------------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------------------------
st.sidebar.header("1. 3D Nanomaterial & Catalyst")
selected_mat = st.sidebar.selectbox("Sensing Nanostructure", list(MATERIALS.keys()), index=0)
mat = MATERIALS[selected_mat]

grain_nm = st.sidebar.slider("Nanograin Diameter D (nm)", 6.0, 60.0, float(mat["grain_nm"]), 1.0)
catalyst = st.sidebar.selectbox(
    "Noble Metal Decoration",
    ["Pristine (Unmodified)", "Pd Nanoparticles (H₂/CO Boost)", "Au Nanoparticles (VOC/EtOH Boost)", "Pt Clusters (Low-Temp Activation)"],
    index=1,
)

st.sidebar.header("2. Gas Chamber & 3D Display")
gas_names = list(mat["gases"].keys())
target_gas_name = st.sidebar.selectbox("Target Analyte Gas", gas_names, index=0)
target_gas = mat["gases"][target_gas_name]
target_ppm = st.sidebar.slider("Gas Concentration (ppm)", 5, 500, 150, 5)
heater_temp_C = st.sidebar.slider("MEMS Heater Temp (°C)", 100, 450, int(mat["T_opt"]), 5)
rh_percent = st.sidebar.slider("Relative Humidity (% RH)", 10, 90, 40, 5)

show_depletion = st.sidebar.checkbox("Show 3D Depletion Shells (λ_D)", value=True)
show_molecules = st.sidebar.checkbox("Show 3D Gas & O⁻ Ion Spheres", value=True)

st.sidebar.header("3. MCU Signal Conditioning")
vcc = st.sidebar.selectbox("Supply Voltage V_CC (V)", [3.3, 5.0], index=0)
r_load_kohm = st.sidebar.slider("Load Resistor R_L (kΩ)", 1.0, 200.0, 47.0, 1.0)
adc_bits = st.sidebar.radio("ADC Resolution", [10, 12], index=1, format_func=lambda b: f"{b}-bit")

# ---------------------------------------------------------------------------
# Physics Calculations
# ---------------------------------------------------------------------------
k_B_eV = 8.617333e-5
T_K = heater_temp_C + 273.15
T_opt_K = mat["T_opt"] + 273.15

cat_map = {
    "Pristine (Unmodified)": {"CO": 1.0, "EtOH": 1.0, "H₂": 1.0, "NO₂": 1.0, "t_shift": 0.0, "hex": None},
    "Pd Nanoparticles (H₂/CO Boost)": {"CO": 1.75, "EtOH": 1.15, "H₂": 2.40, "NO₂": 0.90, "t_shift": -35.0, "hex": "#CBD5E1"},
    "Au Nanoparticles (VOC/EtOH Boost)": {"CO": 1.40, "EtOH": 2.15, "H₂": 1.10, "NO₂": 1.05, "t_shift": -25.0, "hex": "#FACC15"},
    "Pt Clusters (Low-Temp Activation)": {"CO": 1.95, "EtOH": 1.65, "H₂": 2.10, "NO₂": 1.15, "t_shift": -45.0, "hex": "#E2E8F0"},
}
cat_cfg = cat_map[catalyst]

humidity_factor = 1.0 - 0.0024 * (rh_percent - 40.0)
R_air = mat["R0_kohm"] * np.exp((mat["Ea_eV"] / k_B_eV) * (1.0 / T_K - 1.0 / T_opt_K)) * humidity_factor

debye_ratio = grain_nm / (2.0 * mat["debye_nm"])
size_gain = (mat["grain_nm"] / grain_nm) ** 0.88

def calc_sensitivity(g_dict, temp_c, ppm_val):
    t_eff = np.exp(-0.5 * ((temp_c - (g_dict["T_peak"] + cat_cfg["t_shift"])) / mat["sigma_T"]) ** 2)
    return 1.0 + (g_dict["S_max"] * cat_cfg[g_dict["short"]] * t_eff * size_gain * humidity_factor) * ((ppm_val / 100.0) ** g_dict["beta"])

S_val = calc_sensitivity(target_gas, heater_temp_C, target_ppm)
is_n = mat["type"] == "n-type"
is_red = target_gas["nature"] == "reducing"
res_drops = (is_n and is_red) or (not is_n and not is_red)

R_gas = (R_air / S_val) if res_drops else (R_air * S_val)
delta_phi_meV = (-1 if res_drops else 1) * k_B_eV * T_K * np.log(S_val) * 1000.0

speedup = np.exp(0.012 * (heater_temp_C - mat["T_opt"]))
tau_resp = np.clip(mat["tau_resp"] / speedup, 1.8, 90.0)
tau_rec = np.clip(mat["tau_rec"] / speedup, 2.5, 120.0)
t90_resp = 2.3026 * tau_resp
t90_rec = 2.3026 * tau_rec
heater_mw = 0.14 * (heater_temp_C - 25.0)

# Transient array (0..300 s)
t = np.linspace(0, 300, 601)
t_on, t_off = 50.0, 175.0
R_t = np.full_like(t, R_air)
m_exp = (t >= t_on) & (t < t_off)
R_t[m_exp] = R_gas + (R_air - R_gas) * np.exp(-(t[m_exp] - t_on) / tau_resp)
R_end = R_gas + (R_air - R_gas) * np.exp(-(t_off - t_on) / tau_resp)
m_rec = t >= t_off
R_t[m_rec] = R_air + (R_end - R_air) * np.exp(-(t[m_rec] - t_off) / tau_rec)

rng = np.random.default_rng(42)
R_noisy = np.maximum(R_t * (1.0 + rng.normal(0.0, 0.005, len(t))), 0.05)
V_out = vcc * (r_load_kohm / (R_noisy + r_load_kohm))
adc_max = (2 ** adc_bits) - 1
adc_codes = np.round((V_out / vcc) * adc_max).astype(int)
V_adc = (adc_codes / adc_max) * vcc

# ---------------------------------------------------------------------------
# Top KPI Metrics Row
# ---------------------------------------------------------------------------
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Semiconductor", f"{mat['type']} (Eg {mat['eg_eV']} eV)")
k2.metric("Response (S)", f"{S_val:.2f}×")
k3.metric("R_air → R_gas", f"{R_air:.1f} → {R_gas:.1f} kΩ")
k4.metric("Barrier Shift ΔΦ_B", f"{delta_phi_meV:+.1f} meV")
k5.metric("t₉₀ Resp / Rec", f"{t90_resp:.1f}s / {t90_rec:.1f}s")
k6.metric("Heater Power", f"{heater_mw:.1f} mW")

# ---------------------------------------------------------------------------
# Side-by-Side Zero-Dependency 60-FPS 3D Scene & Live Transient Response
# ---------------------------------------------------------------------------
col_3d, col_plot = st.columns([1.18, 1.02])

with col_3d:
    st.subheader("3D Animated MEMS Chip & Nanostructure Array")
    st.caption("Drag/touch to rotate 360°, scroll/pinch to zoom, and scrub time—100% offline & browser-compatible.")

    heat_norm = float(np.clip((heater_temp_C - 100.0) / 350.0, 0.0, 1.0))
    sub_r = int(65 + 185 * heat_norm)
    sub_g = int(75 + 25 * (1.0 - heat_norm))
    sub_b = int(95 * (1.0 - heat_norm))
    heater_hex = f"rgb({sub_r},{sub_g},{sub_b})"

    sim_payload = {
        "matShort": selected_mat.split()[0],
        "coreColor": mat["core_hex"],
        "gasShort": target_gas["short"],
        "gasColor": target_gas["hex"],
        "gasPpm": int(target_ppm),
        "heaterTemp": int(heater_temp_C),
        "heaterColor": heater_hex,
        "grainRadius": float(np.clip(0.34 + (grain_nm - 6.0) * 0.0055, 0.34, 0.62)),
        "showDepletion": bool(show_depletion),
        "showMolecules": bool(show_molecules),
        "catalystColor": cat_cfg["hex"],
        "resDrops": bool(res_drops),
        "R_air": float(R_air),
        "R_gas": float(R_gas),
        "tauResp": float(tau_resp),
        "tauRec": float(tau_rec),
        "vcc": float(vcc),
        "rLoad": float(r_load_kohm),
        "nMolecules": int(np.clip(18 + target_ppm // 10, 18, 60)),
    }

    # Self-contained HTML5 3D Perspective Engine (zero external CDN requests)
    compatible_3d_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      <style>
        * {{ box-sizing: border-box; user-select: none; -webkit-user-select: none; }}
        html, body {{
          margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden;
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
          background: transparent;
        }}
        #wrap {{
          position: relative; width: 100%; height: 540px;
          border-radius: 10px; border: 1px solid #cbd5e1;
          background: radial-gradient(circle at 50% 35%, #ffffff 0%, #eef2f6 100%);
          overflow: hidden; touch-action: none;
        }}
        canvas {{ display: block; width: 100%; height: 100%; cursor: grab; }}
        canvas:active {{ cursor: grabbing; }}
        #hud {{
          position: absolute; top: 10px; left: 10px; right: 10px;
          display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 6px;
          background: rgba(15, 23, 42, 0.88); color: #f8fafc;
          padding: 7px 12px; border-radius: 8px; font-size: 12px; pointer-events: none;
        }}
        .badge {{ padding: 2px 8px; border-radius: 999px; font-weight: 600; font-size: 11px; }}
        #legend {{
          position: absolute; top: 52px; left: 10px;
          background: rgba(255, 255, 255, 0.88); border: 1px solid #e2e8f0;
          border-radius: 6px; padding: 6px 9px; font-size: 11px; color: #334155;
          pointer-events: none; line-height: 1.5;
        }}
        .dot {{ display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 5px; }}
        #controls {{
          position: absolute; bottom: 10px; left: 10px; right: 10px;
          display: flex; align-items: center; gap: 8px;
          background: rgba(255, 255, 255, 0.94); border: 1px solid #cbd5e1;
          padding: 7px 10px; border-radius: 8px;
        }}
        button {{
          background: #2563eb; color: #fff; border: none; border-radius: 6px;
          padding: 5px 11px; font-size: 12px; font-weight: 600; cursor: pointer; white-space: nowrap;
        }}
        button:hover {{ background: #1d4ed8; }}
        button.sec {{ background: #475569; }}
        input[type=range] {{ flex: 1; min-width: 80px; cursor: pointer; }}
      </style>
    </head>
    <body>
      <div id="wrap">
        <canvas id="cv"></canvas>
        <div id="hud">
          <div>
            <span id="state-badge" class="badge" style="background:#334155;">CLEAN AIR BASELINE</span>
            <span style="margin-left:8px;">t = <strong id="hud-t">0.0</strong> s</span>
          </div>
          <div>
            R<sub>s</sub> = <strong id="hud-r">0.0</strong> kΩ &nbsp;|&nbsp;
            ADC V<sub>out</sub> = <strong id="hud-v">0.00</strong> V
          </div>
        </div>
        <div id="legend">
          <div><span class="dot" id="leg-core"></span><span id="leg-mat"></span> Nanograins</div>
          <div><span class="dot" style="background:#f59e0b;"></span>Depletion Shell (&lambda;<sub>D</sub>)</div>
          <div><span class="dot" style="background:#10b981;"></span>Conduction Channels</div>
          <div><span class="dot" id="leg-gas"></span><span id="leg-gas-txt"></span> Gas / O<sup>-</sup> Ions</div>
        </div>
        <div id="controls">
          <button id="btn-play">⏸ Pause</button>
          <button id="btn-reset" class="sec">↺ Reset</button>
          <input id="slider" type="range" min="0" max="300" step="0.5" value="0" />
          <span id="slider-lbl" style="font-size:11px; color:#475569; min-width:44px; text-align:right;">0 s</span>
        </div>
      </div>

      <script>
        const CFG = {json.dumps(sim_payload)};
        document.getElementById("leg-core").style.background = CFG.coreColor;
        document.getElementById("leg-mat").textContent = CFG.matShort;
        document.getElementById("leg-gas").style.background = CFG.gasColor;
        document.getElementById("leg-gas-txt").textContent = CFG.gasShort;

        const wrap = document.getElementById("wrap");
        const canvas = document.getElementById("cv");
        const ctx = canvas.getContext("2d");

        function resize() {{
          const dpr = Math.min(window.devicePixelRatio || 1, 2);
          canvas.width = wrap.clientWidth * dpr;
          canvas.height = wrap.clientHeight * dpr;
          ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        }}
        resize();
        window.addEventListener("resize", resize);

        // Camera Orbit State (360° Yaw, Pitch, Zoom)
        let yaw = 0.62;
        let pitch = 0.44;
        let zoom = 64.0;
        let isDragging = false;
        let lastX = 0, lastY = 0;
        let pinchDist = 0;

        canvas.addEventListener("pointerdown", (e) => {{
          isDragging = true;
          lastX = e.clientX;
          lastY = e.clientY;
          canvas.setPointerCapture(e.pointerId);
        }});
        canvas.addEventListener("pointermove", (e) => {{
          if (!isDragging) return;
          const dx = e.clientX - lastX;
          const dy = e.clientY - lastY;
          lastX = e.clientX;
          lastY = e.clientY;
          yaw += dx * 0.009;
          pitch = Math.max(-1.25, Math.min(1.35, pitch + dy * 0.009));
        }});
        canvas.addEventListener("pointerup", (e) => {{
          isDragging = false;
          canvas.releasePointerCapture(e.pointerId);
        }});
        canvas.addEventListener("wheel", (e) => {{
          e.preventDefault();
          zoom = Math.max(30, Math.min(140, zoom * (e.deltaY > 0 ? 0.92 : 1.08)));
        }}, {{ passive: false }});

        // Touch pinch-to-zoom support
        canvas.addEventListener("touchstart", (e) => {{
          if (e.touches.length === 2) {{
            pinchDist = Math.hypot(
              e.touches[0].clientX - e.touches[1].clientX,
              e.touches[0].clientY - e.touches[1].clientY
            );
          }}
        }}, {{ passive: true }});
        canvas.addEventListener("touchmove", (e) => {{
          if (e.touches.length === 2) {{
            const d = Math.hypot(
              e.touches[0].clientX - e.touches[1].clientX,
              e.touches[0].clientY - e.touches[1].clientY
            );
            if (pinchDist > 0) {{
              zoom = Math.max(30, Math.min(140, zoom * (d / pinchDist)));
            }}
            pinchDist = d;
          }}
        }}, {{ passive: true }});

        // 3D -> 2D Perspective Projection
        function project(x, y, z) {{
          const cy = Math.cos(yaw), sy = Math.sin(yaw);
          const cp = Math.cos(pitch), sp = Math.sin(pitch);

          // Center vertically around y = 1.15
          const yc = y - 1.15;
          const x1 = x * cy - z * sy;
          const z1 = x * sy + z * cy;
          const y2 = yc * cp - z1 * sp;
          const z2 = yc * sp + z1 * cp;

          const camDist = 11.5;
          const depth = Math.max(1.5, camDist - z2);
          const scale = (zoom * 11.5) / depth;
          return {{
            sx: wrap.clientWidth * 0.5 + x1 * scale,
            sy: wrap.clientHeight * 0.48 - y2 * scale,
            scale: scale,
            depth: z2
          }};
        }}

        // 3x3x2 Nanograin Coordinates
        const grains = [];
        [-1.85, 0.0, 1.85].forEach(x => {{
          [-1.85, 0.0, 1.85].forEach(z => {{
            [0.75, 2.05].forEach(y => grains.push({{ x, y, z }}));
          }});
        }});

        // Neighbor pairs for conduction channels
        const bridges = [];
        for (let i = 0; i < grains.length; i++) {{
          for (let j = i + 1; j < grains.length; j++) {{
            const dx = grains[i].x - grains[j].x;
            const dy = grains[i].y - grains[j].y;
            const dz = grains[i].z - grains[j].z;
            if (Math.hypot(dx, dy, dz) <= 1.95) bridges.push([grains[i], grains[j]]);
          }}
        }}

        // Gas Molecules Pool
        const molecules = Array.from({{ length: CFG.nMolecules }}, (_, idx) => ({{
          x: (Math.random() - 0.5) * 6.2,
          y: 0.4 + Math.random() * 2.8,
          z: (Math.random() - 0.5) * 6.2,
          vx: 0.025 + Math.random() * 0.03,
          phase: idx * 0.7
        }}));

        // Playback Controls
        let simTime = 0.0;
        let isPlaying = true;
        const tOn = 50.0, tOff = 175.0;
        const btnPlay = document.getElementById("btn-play");
        const btnReset = document.getElementById("btn-reset");
        const slider = document.getElementById("slider");
        const sliderLbl = document.getElementById("slider-lbl");

        btnPlay.onclick = () => {{
          isPlaying = !isPlaying;
          btnPlay.textContent = isPlaying ? "⏸ Pause" : "▶ Play";
        }};
        btnReset.onclick = () => {{ yaw = 0.62; pitch = 0.44; zoom = 64.0; }};
        slider.oninput = (e) => {{ simTime = parseFloat(e.target.value); }};

        function getR(tSec) {{
          if (tSec < tOn) return CFG.R_air;
          const rEnd = CFG.R_gas + (CFG.R_air - CFG.R_gas) * Math.exp(-(tOff - tOn) / CFG.tauResp);
          if (tSec < tOff) return CFG.R_gas + (CFG.R_air - CFG.R_gas) * Math.exp(-(tSec - tOn) / CFG.tauResp);
          return CFG.R_air + (rEnd - CFG.R_air) * Math.exp(-(tSec - tOff) / CFG.tauRec);
        }}

        function drawQuad(pts, fill, stroke) {{
          const p = pts.map(pt => project(pt[0], pt[1], pt[2]));
          ctx.beginPath();
          ctx.moveTo(p[0].sx, p[0].sy);
          for (let i = 1; i < p.length; i++) ctx.lineTo(p[i].sx, p[i].sy);
          ctx.closePath();
          if (fill) {{ ctx.fillStyle = fill; ctx.fill(); }}
          if (stroke) {{ ctx.strokeStyle = stroke; ctx.lineWidth = 1.2; ctx.stroke(); }}
        }}

        function drawShadedSphere(sx, sy, r, baseHex, alpha = 1.0) {{
          if (r <= 0.5) return;
          ctx.save();
          ctx.globalAlpha = alpha;
          const grad = ctx.createRadialGradient(sx - r * 0.32, sy - r * 0.32, r * 0.1, sx, sy, r);
          grad.addColorStop(0, "#ffffff");
          grad.addColorStop(0.32, baseHex);
          grad.addColorStop(1, "#0f172a");
          ctx.beginPath();
          ctx.arc(sx, sy, r, 0, Math.PI * 2);
          ctx.fillStyle = grad;
          ctx.fill();
          ctx.restore();
        }}

        let lastMs = performance.now();
        function render(nowMs) {{
          requestAnimationFrame(render);
          const dt = Math.min((nowMs - lastMs) / 1000.0, 0.1);
          lastMs = nowMs;

          if (isPlaying) {{
            simTime += dt * 18.0;
            if (simTime > 300.0) simTime = 0.0;
            slider.value = simTime.toFixed(1);
          }}

          const currR = getR(simTime);
          const currV = CFG.vcc * (CFG.rLoad / (currR + CFG.rLoad));
          const gasActive = simTime >= tOn && simTime < tOff;
          const purging = simTime >= tOff && simTime < tOff + 35.0;

          let condNorm = CFG.resDrops
            ? Math.max(0, Math.min(1, (CFG.R_air - currR) / Math.max(CFG.R_air - CFG.R_gas, 0.001)))
            : Math.max(0, Math.min(1, 1.0 - (currR - CFG.R_air) / Math.max(CFG.R_gas - CFG.R_air, 0.001)));

          ctx.clearRect(0, 0, wrap.clientWidth, wrap.clientHeight);

          // 1. Draw MEMS Silicon Substrate & Glowing Micro-Heater Plate
          drawQuad([[-3.5, -0.15, -3.5], [3.5, -0.15, -3.5], [3.5, -0.15, 3.5], [-3.5, -0.15, 3.5]], "#1e293b", "#0f172a");
          drawQuad([[-3.1, 0.0, -3.1], [3.1, 0.0, -3.1], [3.1, 0.0, 3.1], [-3.1, 0.0, 3.1]], CFG.heaterColor, "#991b1b");

          // 2. Gold (Au) Interdigitated Electrodes
          drawQuad([[-2.9, 0.03, -2.7], [-2.55, 0.03, -2.7], [-2.55, 0.03, 2.7], [-2.9, 0.03, 2.7]], "#eab308", "#a16207");
          drawQuad([[2.55, 0.03, -2.7], [2.9, 0.03, -2.7], [2.9, 0.03, 2.7], [2.55, 0.03, 2.7]], "#eab308", "#a16207");
          [-2.0, -0.67, 0.67, 2.0].forEach((fz, idx) => {{
            const x0 = idx % 2 === 0 ? -2.55 : -1.7;
            const x1 = idx % 2 === 0 ? 1.7 : 2.55;
            drawQuad([[x0, 0.03, fz - 0.1], [x1, 0.03, fz - 0.1], [x1, 0.03, fz + 0.1], [x0, 0.03, fz + 0.1]], "#facc15", null);
          }});

          // 3. Grain-Boundary Conduction Channels
          bridges.forEach(([a, b]) => {{
            const pa = project(a.x, a.y, a.z);
            const pb = project(b.x, b.y, b.z);
            ctx.save();
            ctx.beginPath();
            ctx.moveTo(pa.sx, pa.sy);
            ctx.lineTo(pb.sx, pb.sy);
            ctx.strokeStyle = `rgba(16, 185, 129, ${{0.22 + 0.68 * condNorm}})`;
            ctx.lineWidth = (0.08 + 0.22 * condNorm) * ((pa.scale + pb.scale) * 0.5);
            ctx.lineCap = "round";
            ctx.stroke();
            ctx.restore();
          }});

          // 4. Depth-Sorted 3D Render Queue (Depletion Shells, Nanograin Cores, Catalysts, Gas/O- Ions)
          const renderList = [];
          const depScale = 1.0 + 0.55 * (1.0 - 0.78 * condNorm);

          grains.forEach((g, idx) => {{
            const p = project(g.x, g.y, g.z);
            if (CFG.showDepletion) {{
              renderList.push({{ type: "shell", depth: p.depth - 0.01, sx: p.sx, sy: p.sy, r: CFG.grainRadius * depScale * p.scale }});
            }}
            renderList.push({{ type: "core", depth: p.depth, sx: p.sx, sy: p.sy, r: CFG.grainRadius * p.scale }});

            if (CFG.catalystColor) {{
              const pc = project(g.x + CFG.grainRadius * 0.7, g.y + CFG.grainRadius * 0.65, g.z + 0.2);
              renderList.push({{ type: "cat", depth: pc.depth, sx: pc.sx, sy: pc.sy, r: 0.11 * pc.scale }});
            }}
            if (CFG.showMolecules && (!gasActive || idx % 3 === 0)) {{
              const jy = Math.sin(nowMs * 0.005 + idx) * 0.03;
              const po = project(g.x - CFG.grainRadius * 0.72, g.y + CFG.grainRadius * 0.55 + jy, g.z - 0.25);
              renderList.push({{ type: "ox", depth: po.depth, sx: po.sx, sy: po.sy, r: 0.11 * po.scale }});
            }}
          }});

          if (CFG.showMolecules && (gasActive || purging)) {{
            const keepRatio = gasActive ? 1.0 : Math.max(0, 1.0 - (simTime - tOff) / 35.0);
            molecules.forEach((m, idx) => {{
              m.x += m.vx * (purging ? 1.6 : 1.0);
              if (m.x > 3.1) m.x = -3.1;
              if (idx < molecules.length * keepRatio) {{
                const my = m.y + Math.sin(nowMs * 0.004 + m.phase) * 0.08;
                const pm = project(m.x, my, m.z);
                renderList.push({{ type: "gas", depth: pm.depth, sx: pm.sx, sy: pm.sy, r: 0.11 * pm.scale }});
              }}
            }});
          }}

          // Painter's Algorithm Sort (Back-to-Front)
          renderList.sort((a, b) => a.depth - b.depth);

          renderList.forEach(item => {{
            if (item.type === "shell") {{
              ctx.save();
              ctx.beginPath();
              ctx.arc(item.sx, item.sy, item.r, 0, Math.PI * 2);
              ctx.fillStyle = "rgba(245, 158, 11, 0.22)";
              ctx.fill();
              ctx.strokeStyle = "rgba(245, 158, 11, 0.55)";
              ctx.lineWidth = 1.1;
              ctx.stroke();
              ctx.restore();
            }} else if (item.type === "core") {{
              drawShadedSphere(item.sx, item.sy, item.r, CFG.coreColor, 0.96);
            }} else if (item.type === "cat") {{
              drawShadedSphere(item.sx, item.sy, item.r, CFG.catalystColor, 1.0);
            }} else if (item.type === "ox") {{
              drawShadedSphere(item.sx, item.sy, item.r, "#ef4444", 0.95);
            }} else if (item.type === "gas") {{
              drawShadedSphere(item.sx, item.sy, item.r, CFG.gasColor, 0.95);
            }}
          }});

          // Update HUD
          sliderLbl.textContent = `${{Math.round(simTime)}} s`;
          document.getElementById("hud-t").textContent = simTime.toFixed(1);
          document.getElementById("hud-r").textContent = currR.toFixed(2);
          document.getElementById("hud-v").textContent = currV.toFixed(3);

          const badge = document.getElementById("state-badge");
          if (gasActive) {{
            badge.textContent = `GAS ON (${{CFG.gasShort}} ${{CFG.gasPpm}} ppm)`;
            badge.style.background = "#059669";
          }} else if (purging) {{
            badge.textContent = "CLEAN AIR PURGE";
            badge.style.background = "#d97706";
          }} else {{
            badge.textContent = "CLEAN AIR BASELINE";
            badge.style.background = "#334155";
          }}
        }}

        requestAnimationFrame(render);
      </script>
    </body>
    </html>
    """

    components.html(compatible_3d_html, height=555)

with col_plot:
    st.subheader("Real-Time Sensor R_s(t) & MCU ADC Signal")
    fig_trans = make_subplots(specs=[[{"secondary_y": True}]])

    fig_trans.add_vrect(
        x0=t_on, x1=t_off, fillcolor="rgba(37, 99, 235, 0.08)", layer="below", line_width=0,
        annotation_text=f"{target_gas['short']} ON ({target_ppm} ppm)", annotation_position="top left",
    )
    fig_trans.add_trace(
        go.Scatter(x=t, y=R_noisy, name="Sensor Resistance R_s (kΩ)", line=dict(color="#2563EB", width=2.4)),
        secondary_y=False,
    )
    fig_trans.add_trace(
        go.Scatter(x=t, y=V_adc, name=f"MCU ADC V_out ({adc_bits}-bit)", line=dict(color="#EA580C", width=2.1, dash="dot")),
        secondary_y=True,
    )

    fig_trans.update_layout(
        height=555,
        margin=dict(l=10, r=10, t=25, b=10),
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    fig_trans.update_xaxes(title_text="Time (seconds)")
    fig_trans.update_yaxes(title_text="Sensor Resistance R_s (kΩ)", secondary_y=False)
    fig_trans.update_yaxes(title_text="MCU ADC Voltage V_out (V)", range=[0, vcc * 1.05], secondary_y=True)
    st.plotly_chart(fig_trans, width="stretch")

# ---------------------------------------------------------------------------
# Bottom Row: Double-Schottky Band Bending & Cross-Selectivity Comparison
# ---------------------------------------------------------------------------
col_band, col_sel = st.columns([1.05, 1.15])

with col_band:
    st.subheader("Grain-Boundary Double-Schottky Band Bending")
    phi_air = 0.65 * np.clip(1.0 + 0.15 * (1.0 - debye_ratio), 0.7, 1.35)
    phi_gas = max(0.08, phi_air + (delta_phi_meV / 1000.0))

    x_pos = np.linspace(-1.0, 1.0, 200)
    barrier = np.exp(-((x_pos / 0.32) ** 2))
    Ec_air = 1.0 + phi_air * barrier
    Ec_gas = 1.0 + phi_gas * barrier

    fig_band = go.Figure()
    fig_band.add_trace(go.Scatter(x=x_pos, y=Ec_air, mode="lines", name=f"E_c Clean Air ({phi_air:.2f} eV)", line=dict(color="#64748B", width=2.4, dash="dash")))
    fig_band.add_trace(go.Scatter(x=x_pos, y=Ec_gas, mode="lines", name=f"E_c {target_gas['short']} ({phi_gas:.2f} eV)", line=dict(color="#2563EB", width=2.8)))
    fig_band.add_trace(go.Scatter(x=x_pos, y=np.full_like(x_pos, 0.72), mode="lines", name="Fermi Level (E_F)", line=dict(color="#DC2626", width=1.8, dash="dot")))
    fig_band.update_layout(
        height=320, margin=dict(l=10, r=10, t=25, b=10), template="plotly_white",
        xaxis_title="Normalized Distance Across Grain Interface (x / λ_D)",
        yaxis_title="Electron Energy (eV)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig_band, width="stretch")

with col_sel:
    st.subheader(f"Cross-Selectivity at {heater_temp_C} °C ({target_ppm} ppm)")
    g_labels, g_vals, g_cols = [], [], []
    for _, g_d in mat["gases"].items():
        g_labels.append(g_d["short"])
        g_vals.append(calc_sensitivity(g_d, heater_temp_C, target_ppm))
        g_cols.append(g_d["color"])

    fig_sel = go.Figure(
        data=[go.Bar(x=g_labels, y=g_vals, marker_color=g_cols, text=[f"{v:.1f}×" for v in g_vals], textposition="outside")]
    )
    fig_sel.update_layout(
        height=320, margin=dict(l=10, r=10, t=25, b=10), template="plotly_white",
        xaxis_title="Analyte Gas Species", yaxis_title="Sensitivity Response S",
        yaxis=dict(range=[0, max(g_vals) * 1.24]),
    )
    st.plotly_chart(fig_sel, width="stretch")

# ---------------------------------------------------------------------------
# CSV Telemetry Export
# ---------------------------------------------------------------------------
export_df = pd.DataFrame({
    "Time_s": np.round(t, 2),
    "Resistance_kOhm": np.round(R_noisy, 4),
    "V_ADC_V": np.round(V_adc, 4),
    "ADC_Code": adc_codes,
})
st.download_button(
    label="📥 Download 3D Sensor & MCU ADC Telemetry (CSV)",
    data=export_df.to_csv(index=False).encode("utf-8"),
    file_name=f"3d_gas_sensor_twin_{selected_mat.split()[0].lower()}.csv",
    mime="text/csv",
)