
# 🧊 3D Nanomaterial Gas Sensor & MEMS Digital Twin

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nanomaterial-gas-sensor-twin-dybkyeraufheb85rmktxtb.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Interactive_Charts-3F4F75?logo=plotly&logoColor=white)
![HTML5 Canvas 3D](https://img.shields.io/badge/3D_Engine-Zero--CDN_60FPS-0D9488)

An interactive, cross-platform **Chemiresistive Thin-Film Gas Sensor & Embedded MEMS Hardware Digital Twin** that bridges solid-state nanomaterials physics with microcontroller signal conditioning.

🚀 **Live Interactive App:** [Launch 3D Nanomaterial Gas Sensor Digital Twin](https://nanomaterial-gas-sensor-twin-dybkyeraufheb85rmktxtb.streamlit.app/)

---

## ✨ Key Features

### 1. Zero-CDN 60-FPS Interactive 3D MEMS & Nanostructure Viewport
- **360° Orbit, Pan & Pinch/Scroll Zoom:** Built with a self-contained HTML5 3D projection engine (zero external CDN dependencies) for instant offline and cross-platform rendering across desktop, trackpad, and mobile touchscreens.
- **Dynamic Depletion Layer (λ<sub>D</sub>) & Conduction Channels:** Watch translucent electron depletion shells expand and contract around 3D nanograins in real time as target gas molecules diffuse across a heated substrate and gold (Au) interdigitated electrodes (IDE).
- **Timeline Scrubber & Playback:** Play, pause, or scrub across a 300-second exposure and clean-air recovery cycle (`t = 50 s` gas valve ON → `t = 175 s` clean-air purge) with live telemetry overlays (R<sub>s</sub> and V<sub>ADC</sub>).

### 2. Solid-State Semiconductor & Surface Ionosorption Physics
- **4 Nanostructured Metal-Oxide Sensing Materials:**
  - **SnO₂ Nanowire Network** (*n*-type, E<sub>g</sub> = 3.60 eV)
  - **ZnO Nanorod Array** (*n*-type, E<sub>g</sub> = 3.37 eV)
  - **CuO Nanoparticle Film** (*p*-type, E<sub>g</sub> = 1.35 eV)
  - **WO₃ Nanosheet Cluster** (*n*-type, E<sub>g</sub> = 2.75 eV)
- ***n*-Type vs. *p*-Type Resistance Inversion:** Accurately models resistance decreases for reducing gases (C₂H₅OH, CO, H₂) on *n*-type oxides and resistance increases on *p*-type CuO, with inverse behavior for oxidizing NO₂.
- **Grain-Size Scaling & Noble-Metal Sensitization:** Adjust nanocrystallite diameter *D* (6–60 nm) relative to 2λ<sub>D</sub>, and simulate surface decoration with **Pd**, **Au**, or **Pt** catalyst clusters.
- **Double-Schottky Barrier Bending (ΔΦ<sub>B</sub>):** Visualizes grain-boundary conduction band (E<sub>c</sub>) barrier height shifts in clean air vs. target gas exposure.

### 3. Embedded Hardware & MCU ADC Signal Conditioning
- **Voltage Divider & Quantization Model:** Configure supply voltage (V<sub>CC</sub> = 3.3 V or 5.0 V), load resistor (R<sub>L</sub>), and **10-bit / 12-bit MCU ADC resolution**.
- **Transient Kinetics & Cross-Selectivity:** Computes t₉₀ response and recovery times, MEMS micro-heater power consumption (mW), and multi-gas cross-selectivity comparisons.
- **CSV Telemetry Export:** Download complete time-series resistance (R<sub>s</sub>), quantized voltage (V<sub>ADC</sub>), and raw integer ADC codes for embedded firmware testing.

---

## 🛠️ Local Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/DulsaraPrasad/nanomaterial-gas-sensor-twin.git](https://github.com/DulsaraPrasad/nanomaterial-gas-sensor-twin.git)
   cd nanomaterial-gas-sensor-twin
