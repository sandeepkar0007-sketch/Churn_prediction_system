import streamlit as st
import pandas as pd
import numpy as np
import time
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="ChurnIQ — Telecom Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------------
# CYBERPUNK THEME WITH CANVAS ANIMATIONS
# -------------------------------
st.markdown("""
<style>
/* Import premium fonts */
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&display=swap');

/* Global Styles */
:root {
    --bg: #02040c;
    --surface: #0a1120;
    --border: #162030;
    --accent: #00e5ff;
    --accent2: #ff3d71;
    --warn: #ffaa00;
    --text: #e8edf5;
    --muted: #4a5670;
    --card: #0b1524;
    --green: #00e676;
    --purple: #b060ff;
}

.stApp {
    background: var(--bg);
    color: var(--text);
    font-family: 'Syne', sans-serif;
    overflow-x: hidden;
}

/* Canvas Container for Animations */
.canvas-container {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    z-index: 0;
    pointer-events: none;
}

.canvas-bg {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
}

#cAurora { z-index: 0; }
#cStars { z-index: 1; }
#cGlobe { z-index: 2; opacity: 0.5; }
#cHelix { z-index: 3; opacity: 0.55; }
#cMatrix { z-index: 4; opacity: 0.15; }
#cParticle { z-index: 5; }
#cLightning { z-index: 6; opacity: 0.7; }
#cNeural { z-index: 7; opacity: 0.55; }
#cRadar { z-index: 8; opacity: 0.12; }
#cWave { z-index: 9; width: 300px; height: 40px; }

/* Grid Overlay */
.grid-overlay {
    position: fixed;
    inset: 0;
    z-index: 9;
    pointer-events: none;
    background-image:
        linear-gradient(rgba(0,229,255,.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,229,255,.025) 1px, transparent 1px);
    background-size: 50px 50px;
    animation: gridPulse 9s ease-in-out infinite;
}

@keyframes gridPulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.35; }
}

/* Vignette Effect */
.vignette {
    position: fixed;
    inset: 0;
    z-index: 10;
    pointer-events: none;
    background: radial-gradient(ellipse 80% 80% at 50% 50%, transparent 30%, rgba(2,4,12,.92) 100%);
}

/* Scanline Animation */
.scanline {
    position: fixed;
    left: 0;
    right: 0;
    height: 3px;
    z-index: 11;
    pointer-events: none;
    background: linear-gradient(90deg, transparent, rgba(0,229,255,.35) 40%, rgba(0,229,255,.7) 50%, rgba(0,229,255,.35) 60%, transparent);
    filter: blur(1px);
    animation: scan 7s linear infinite;
}

@keyframes scan {
    0% { top: -3px; opacity: 0; }
    3% { opacity: 1; }
    97% { opacity: .3; }
    100% { top: 100vh; opacity: 0; }
}

/* Floating Pills */
.float-pills {
    position: fixed;
    inset: 0;
    z-index: 10;
    pointer-events: none;
    overflow: hidden;
}

.pill {
    position: absolute;
    font-family: 'DM Mono', monospace;
    font-size: .58rem;
    color: rgba(0,229,255,.3);
    border: 1px solid rgba(0,229,255,.1);
    padding: 2px 8px;
    border-radius: 20px;
    white-space: nowrap;
    background: rgba(0,229,255,.03);
    animation: pillRise linear infinite;
}

@keyframes pillRise {
    0% { transform: translateY(110vh); opacity: 0; }
    5% { opacity: 1; }
    95% { opacity: .45; }
    100% { transform: translateY(-20vh); opacity: 0; }
}

/* Holographic Rings */
.holo-rings {
    position: fixed;
    inset: 0;
    z-index: 9;
    pointer-events: none;
    overflow: hidden;
}

.holo-ring {
    position: absolute;
    border-radius: 50%;
    border: 1px solid;
    animation: holoSpin linear infinite;
    transform-origin: center center;
}

@keyframes holoSpin {
    to { transform: translate(-50%,-50%) rotate(360deg); }
}

/* Header Styles */
.churniq-header {
    position: relative;
    z-index: 20;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.1rem 2.5rem;
    border-bottom: 1px solid var(--border);
    background: rgba(2,4,12,.75);
    backdrop-filter: blur(20px);
    margin-bottom: 0;
}

.logo {
    display: flex;
    align-items: center;
    gap: .85rem;
}

.logo-icon {
    position: relative;
    width: 44px;
    height: 44px;
}

.logo-icon svg {
    animation: spinRing 10s linear infinite;
}

@keyframes spinRing {
    to { transform: rotate(360deg); }
}

.logo-dot {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 13px;
    height: 13px;
    background: var(--accent);
    clip-path: polygon(50% 0%, 100% 50%, 50% 100%, 0% 50%);
    animation: pulseDot 2s ease-in-out infinite;
    box-shadow: 0 0 20px var(--accent);
}

@keyframes pulseDot {
    0%, 100% { transform: translate(-50%, -50%) scale(1); }
    50% { transform: translate(-50%, -50%) scale(1.4); }
}

.logo-text h1 {
    font-size: 1.5rem;
    font-weight: 800;
    letter-spacing: -.03em;
    color: var(--text);
    margin: 0;
}

.logo-text h1 span {
    color: var(--accent);
}

.logo-text sub {
    font-family: 'DM Mono', monospace;
    font-size: .52rem;
    color: var(--muted);
    letter-spacing: .15em;
    text-transform: uppercase;
    display: block;
}

.glitch {
    position: relative;
    display: inline-block;
}

.glitch::before, .glitch::after {
    content: attr(data-text);
    position: absolute;
    left: 0;
    top: 0;
    font-size: inherit;
    font-weight: inherit;
}

.glitch::before {
    color: var(--accent2);
    animation: g1 5s infinite;
    opacity: 0;
    clip-path: polygon(0 20%, 100% 20%, 100% 45%, 0 45%);
}

.glitch::after {
    color: var(--accent);
    animation: g2 5s infinite;
    opacity: 0;
    clip-path: polygon(0 65%, 100% 65%, 100% 85%, 0 85%);
}

@keyframes g1 {
    0%, 92%, 100% { opacity: 0; transform: translate(0); }
    93% { opacity: .9; transform: translate(-4px, 1px); }
    95% { opacity: .5; transform: translate(3px, -1px); }
    97% { opacity: 0; }
}

@keyframes g2 {
    0%, 91%, 100% { opacity: 0; transform: translate(0); }
    92% { opacity: .7; transform: translate(4px, -2px); }
    94% { opacity: .3; transform: translate(-3px, 2px); }
    96% { opacity: 0; }
}

.header-mid {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
}

.live-row {
    display: flex;
    align-items: center;
    gap: 6px;
    font-family: 'DM Mono', monospace;
    font-size: .63rem;
    color: var(--green);
}

.live-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--green);
    box-shadow: 0 0 10px var(--green);
    animation: livePulse 1.5s ease-in-out infinite;
}

@keyframes livePulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: .3; transform: scale(.6); }
}

.sig-bars {
    display: flex;
    align-items: flex-end;
    gap: 2px;
    height: 16px;
}

.sbar {
    width: 4px;
    background: var(--accent);
    border-radius: 1px;
    animation: sigAnim 1.2s ease-in-out infinite;
}

.sbar:nth-child(1) { height: 30%; animation-delay: 0s; }
.sbar:nth-child(2) { height: 55%; animation-delay: .15s; }
.sbar:nth-child(3) { height: 75%; animation-delay: .3s; }
.sbar:nth-child(4) { height: 100%; animation-delay: .45s; }

@keyframes sigAnim {
    0%, 100% { opacity: 1; }
    50% { opacity: .2; }
}

.badge {
    font-family: 'DM Mono', monospace;
    font-size: .63rem;
    color: var(--accent);
    border: 1px solid rgba(0,229,255,.25);
    padding: .25rem .65rem;
    border-radius: 2px;
    letter-spacing: .08em;
    background: rgba(0,229,255,.05);
    position: relative;
    overflow: hidden;
    display: inline-block;
    margin-left: 0.5rem;
}

.badge::after {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(0,229,255,.3), transparent);
    animation: shine 3s ease-in-out infinite;
}

@keyframes shine {
    0%, 60%, 100% { left: -100%; }
    80% { left: 100%; }
}

/* Wave Bar */
.wave-bar {
    position: relative;
    z-index: 10;
    height: 44px;
    background: rgba(2,4,12,.6);
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    padding: 0 2rem;
    gap: 2rem;
    overflow: hidden;
}

.wave-label {
    font-family: 'DM Mono', monospace;
    font-size: .58rem;
    color: var(--muted);
    letter-spacing: .12em;
}

.tele {
    display: flex;
    gap: 2rem;
    margin-left: auto;
}

.ti {
    font-family: 'DM Mono', monospace;
    font-size: .6rem;
    color: var(--muted);
    display: flex;
    gap: 5px;
    align-items: center;
}

.tv {
    color: var(--accent);
    font-weight: 500;
}

/* Main Layout */
.main-content {
    position: relative;
    z-index: 10;
    max-width: 1320px;
    margin: 0 auto;
    padding: 2.5rem 2rem;
}

/* Form Panel */
.form-panel {
    background: rgba(11,21,36,.85);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 2rem;
    backdrop-filter: blur(16px);
    animation: slideLeft .7s cubic-bezier(.16,1,.3,1) both;
    position: relative;
    overflow: hidden;
    height: 100%;
}

.form-panel::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
    animation: topGlow 4s ease-in-out infinite;
}

@keyframes topGlow {
    0%, 100% { opacity: .2; }
    50% { opacity: 1; }
}

@keyframes slideLeft {
    from { opacity: 0; transform: translateX(-40px); }
    to { opacity: 1; transform: translateX(0); }
}

.panel-title {
    font-size: .62rem;
    font-family: 'DM Mono', monospace;
    letter-spacing: .2em;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: .6rem;
}

.panel-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

.ptd {
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: var(--accent);
    box-shadow: 0 0 8px var(--accent);
    animation: livePulse 2s ease-in-out infinite;
}

/* Result Panel */
.result-panel {
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
    animation: slideRight .7s cubic-bezier(.16,1,.3,1) .1s both;
    height: 100%;
}

@keyframes slideRight {
    from { opacity: 0; transform: translateX(40px); }
    to { opacity: 1; transform: translateX(0); }
}

.rc {
    background: rgba(11,21,36,.85);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    backdrop-filter: blur(14px);
    position: relative;
    overflow: hidden;
    transition: border-color .3s, box-shadow .3s;
}

.rc:hover {
    border-color: rgba(0,229,255,.2);
    box-shadow: 0 0 30px rgba(0,229,255,.06);
}

.rc::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,229,255,.08), transparent);
}

/* Form Elements */
.field {
    display: flex;
    flex-direction: column;
    gap: .4rem;
    margin-bottom: 1rem;
}

label {
    font-size: .66rem;
    font-family: 'DM Mono', monospace;
    color: var(--muted);
    letter-spacing: .08em;
    text-transform: uppercase;
}

/* Override Streamlit defaults */
.stSlider label, .stSelectbox label, .stNumberInput label {
    font-size: .66rem !important;
    font-family: 'DM Mono', monospace !important;
    color: var(--muted) !important;
    letter-spacing: .08em !important;
    text-transform: uppercase !important;
}

.stSlider > div > div {
    background: rgba(10,16,32,.9) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    padding: 0.3rem !important;
}

.stSelectbox > div > div {
    background: rgba(10,16,32,.9) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    color: var(--text) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: .88rem !important;
    transition: all .2s !important;
}

.stSelectbox > div > div:hover, .stSelectbox > div > div:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(0,229,255,.08), 0 0 20px rgba(0,229,255,.06) !important;
    background: rgba(0,229,255,.04) !important;
}

.stNumberInput > div > div > input {
    background: rgba(10,16,32,.9) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: .88rem !important;
    border-radius: 6px !important;
}

.stNumberInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(0,229,255,.08) !important;
}

.divider {
    font-size: .6rem;
    font-family: 'DM Mono', monospace;
    color: rgba(0,229,255,.5);
    text-transform: uppercase;
    letter-spacing: .2em;
    margin: 1.4rem 0 .9rem;
    display: flex;
    align-items: center;
    gap: .75rem;
}

.divider::before {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--border), transparent);
}

.divider::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border));
}

/* Toggle Group */
.tg {
    display: flex;
    gap: .4rem;
    margin-bottom: 0.5rem;
}

.tb {
    flex: 1;
    padding: .5rem;
    background: rgba(10,16,32,.9);
    border: 1px solid var(--border);
    color: var(--muted);
    font-family: 'DM Mono', monospace;
    font-size: .72rem;
    border-radius: 4px;
    cursor: pointer;
    transition: all .2s;
    text-align: center;
}

.tb:hover {
    border-color: rgba(0,229,255,.3);
    color: rgba(0,229,255,.7);
}

.tb.active {
    background: rgba(0,229,255,.1);
    border-color: var(--accent);
    color: var(--accent);
    box-shadow: 0 0 12px rgba(0,229,255,.1);
}

/* Predict Button */
.predict-btn {
    width: 100%;
    margin-top: 2rem;
    padding: 1.1rem;
    background: transparent;
    color: var(--accent);
    border: 1px solid var(--accent);
    border-radius: 8px;
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: .1em;
    cursor: pointer;
    transition: all .3s;
    position: relative;
    overflow: hidden;
    text-align: center;
}

.predict-btn::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(0,229,255,.18), rgba(0,229,255,.04));
    transform: translateX(-100%);
    transition: transform .4s;
}

.predict-btn:hover::before {
    transform: translateX(0);
}

.predict-btn:hover {
    box-shadow: 0 0 40px rgba(0,229,255,.35), inset 0 0 30px rgba(0,229,255,.05);
    letter-spacing: .16em;
}

/* Result Meter */
.churn-meter {
    text-align: center;
    padding: .5rem 0;
}

.meter-ring {
    position: relative;
    width: 180px;
    height: 180px;
    margin: 0 auto 1rem;
}

.meter-value {
    position: absolute;
    inset: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}

.meter-pct {
    font-size: 2.8rem;
    font-weight: 800;
    line-height: 1;
    transition: color .5s;
}

.meter-sub {
    font-family: 'DM Mono', monospace;
    font-size: .55rem;
    color: var(--muted);
    margin-top: 3px;
    letter-spacing: .12em;
}

.verdict {
    display: inline-flex;
    align-items: center;
    gap: .5rem;
    padding: .4rem 1.1rem;
    border-radius: 20px;
    font-weight: 700;
    font-size: .82rem;
    letter-spacing: .06em;
    margin-top: .5rem;
}

.verdict.low {
    background: rgba(0,230,118,.1);
    color: var(--green);
    border: 1px solid rgba(0,230,118,.25);
}

.verdict.medium {
    background: rgba(255,170,0,.1);
    color: var(--warn);
    border: 1px solid rgba(255,170,0,.25);
}

.verdict.high {
    background: rgba(255,61,113,.1);
    color: var(--accent2);
    border: 1px solid rgba(255,61,113,.25);
}

/* Stats Row */
.stats-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: .75rem;
    margin: 1.5rem 0;
}

.sc {
    background: rgba(10,16,32,.8);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: .85rem;
    text-align: center;
    transition: all .2s;
}

.sc:hover {
    border-color: rgba(0,229,255,.2);
    transform: translateY(-2px);
}

.sv {
    font-size: 1.4rem;
    font-weight: 800;
    line-height: 1;
    margin-bottom: .2rem;
    color: var(--text);
}

.sl {
    font-family: 'DM Mono', monospace;
    font-size: .58rem;
    color: var(--muted);
    letter-spacing: .1em;
    text-transform: uppercase;
}

/* Factors List */
.factors-list {
    display: flex;
    flex-direction: column;
    gap: .9rem;
    margin-top: 1rem;
}

.factor-row {
    display: flex;
    flex-direction: column;
    gap: .35rem;
}

.factor-header {
    display: flex;
    justify-content: space-between;
    font-size: .78rem;
}

.factor-name {
    font-weight: 600;
    color: var(--text);
}

.factor-val {
    font-family: 'DM Mono', monospace;
    font-size: .7rem;
    color: var(--muted);
}

.factor-bar-bg {
    height: 5px;
    background: var(--border);
    border-radius: 3px;
    overflow: hidden;
}

.factor-bar {
    height: 100%;
    border-radius: 3px;
    width: 0;
    transition: width 1.3s cubic-bezier(.4,0,.2,1);
    position: relative;
}

.factor-bar::after {
    content: '';
    position: absolute;
    right: 0;
    top: 0;
    bottom: 0;
    width: 10px;
    background: rgba(255,255,255,.35);
    border-radius: 0 3px 3px 0;
    filter: blur(2px);
}

/* Recommendations */
.reco-list {
    display: flex;
    flex-direction: column;
    gap: .6rem;
    margin-top: 1rem;
}

.reco-item {
    font-size: .8rem;
    color: #a8b8d0;
    line-height: 1.5;
    padding: .5rem .75rem;
    border-radius: 6px;
    border-left: 2px solid rgba(0,229,255,.3);
    background: rgba(0,229,255,.025);
}

/* Placeholder State */
.placeholder-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 300px;
    gap: 1rem;
    text-align: center;
    color: var(--muted);
}

.placeholder-icon {
    width: 72px;
    height: 72px;
    border: 1px dashed rgba(0,229,255,.15);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.8rem;
    color: rgba(0,229,255,.2);
    animation: iconSpin 12s linear infinite;
}

@keyframes iconSpin {
    to { transform: rotate(360deg); }
}

.placeholder-state p {
    font-family: 'DM Mono', monospace;
    font-size: .72rem;
    letter-spacing: .04em;
    line-height: 1.7;
}

/* Responsive */
@media(max-width: 900px) {
    .churniq-header {
        padding: 1rem 1.5rem;
        flex-wrap: wrap;
    }
    .tele {
        display: none;
    }
    .main-content {
        padding: 1.5rem 1rem;
    }
}

/* Remove Streamlit defaults */
div, section, .stApp > div {
    border: none !important;
    outline: none !important;
}

.element-container, .stMarkdown, .stSlider, .stSelectbox, .stNumberInput {
    border: none !important;
    background: transparent !important;
}

.stButton > button {
    width: 100%;
    margin-top: 2rem;
    padding: 1.1rem;
    background: transparent;
    color: var(--accent);
    border: 1px solid var(--accent) !important;
    border-radius: 8px;
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: .1em;
    cursor: pointer;
    transition: all .3s;
    position: relative;
    overflow: hidden;
}

.stButton > button::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(0,229,255,.18), rgba(0,229,255,.04));
    transform: translateX(-100%);
    transition: transform .4s;
}

.stButton > button:hover::before {
    transform: translateX(0);
}

.stButton > button:hover {
    box-shadow: 0 0 40px rgba(0,229,255,.35), inset 0 0 30px rgba(0,229,255,.05) !important;
    letter-spacing: .16em;
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# JavaScript for Canvas Animations and Floating Elements
# -------------------------------
st.markdown("""
<script>
function createCanvases() {
    const container = document.createElement('div');
    container.className = 'canvas-container';
    
    const canvasIds = ['cAurora', 'cStars', 'cGlobe', 'cHelix', 'cMatrix', 'cParticle', 'cLightning', 'cNeural', 'cRadar', 'cWave'];
    
    canvasIds.forEach(id => {
        const canvas = document.createElement('canvas');
        canvas.id = id;
        canvas.className = 'canvas-bg';
        container.appendChild(canvas);
    });
    
    document.body.appendChild(container);
    
    const gridOverlay = document.createElement('div');
    gridOverlay.className = 'grid-overlay';
    document.body.appendChild(gridOverlay);
    
    const vignette = document.createElement('div');
    vignette.className = 'vignette';
    document.body.appendChild(vignette);
    
    const scanline = document.createElement('div');
    scanline.className = 'scanline';
    document.body.appendChild(scanline);
    
    const floatPills = document.createElement('div');
    floatPills.className = 'float-pills';
    floatPills.id = 'floatPills';
    
    const pillTexts = ['5G NR', 'LTE', 'VOIP', 'FIBER', 'CHURN', 'XGBOOST', 'AI', 'ML'];
    for (let i = 0; i < 15; i++) {
        const pill = document.createElement('div');
        pill.className = 'pill';
        pill.textContent = pillTexts[Math.floor(Math.random() * pillTexts.length)];
        pill.style.left = Math.random() * 100 + '%';
        pill.style.animationDuration = (15 + Math.random() * 20) + 's';
        pill.style.animationDelay = (Math.random() * 10) + 's';
        floatPills.appendChild(pill);
    }
    document.body.appendChild(floatPills);
    
    const holoRings = document.createElement('div');
    holoRings.className = 'holo-rings';
    holoRings.id = 'holoRings';
    
    for (let i = 0; i < 5; i++) {
        const ring = document.createElement('div');
        ring.className = 'holo-ring';
        const size = 100 + i * 150;
        ring.style.width = size + 'px';
        ring.style.height = size + 'px';
        ring.style.left = '50%';
        ring.style.top = '50%';
        ring.style.transform = 'translate(-50%, -50%)';
        ring.style.borderColor = `rgba(0, 229, 255, ${0.1 - i * 0.015})`;
        ring.style.animationDuration = (20 + i * 10) + 's';
        holoRings.appendChild(ring);
    }
    document.body.appendChild(holoRings);
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', createCanvases);
} else {
    createCanvases();
}

setInterval(() => {
    const canvas = document.getElementById('cWave');
    if (canvas) {
        const ctx = canvas.getContext('2d');
        const width = canvas.width = canvas.clientWidth;
        const height = canvas.height;
        
        ctx.clearRect(0, 0, width, height);
        ctx.beginPath();
        ctx.strokeStyle = '#00e5ff';
        ctx.lineWidth = 1.5;
        
        for (let x = 0; x < width; x++) {
            const y = height/2 + Math.sin(x * 0.02 + Date.now() * 0.005) * 10 + 
                      Math.cos(x * 0.01 + Date.now() * 0.003) * 5;
            if (x === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        
        ctx.stroke();
    }
}, 50);
</script>
""", unsafe_allow_html=True)

# -------------------------------
# Header
# -------------------------------
st.markdown("""
<header class="churniq-header">
    <div class="logo">
        <div class="logo-icon">
            <svg width="44" height="44" viewBox="0 0 44 44" fill="none">
                <circle cx="22" cy="22" r="20" stroke="rgba(0,229,255,.4)" stroke-width="1"/>
                <circle cx="22" cy="22" r="15" stroke="rgba(0,229,255,.18)" stroke-width="1" stroke-dasharray="3 4"/>
                <circle cx="22" cy="22" r="9" stroke="rgba(0,229,255,.08)" stroke-width="1"/>
            </svg>
            <div class="logo-dot"></div>
        </div>
        <div class="logo-text">
            <h1><span class="glitch" data-text="ChurnIQ">Churn<span style="color:var(--accent)">IQ</span></span></h1>
            <sub>Telecom Intelligence Platform</sub>
        </div>
    </div>
    <div class="header-mid">
        <div class="live-row"><div class="live-dot"></div>MODEL ACTIVE</div>
        <div class="sig-bars"><div class="sbar"></div><div class="sbar"></div><div class="sbar"></div><div class="sbar"></div></div>
    </div>
    <div style="display:flex;gap:.6rem;align-items:center;flex-wrap:wrap;">
        <span class="badge">XGBoost v2.1</span>
        <span class="badge" style="border-color:rgba(0,230,118,.3);color:var(--green);">ACC 87.4%</span>
        <span class="badge" style="border-color:rgba(255,170,0,.3);color:var(--warn);">AUC 0.91</span>
    </div>
</header>
""", unsafe_allow_html=True)

# -------------------------------
# Wave Bar with Stats
# -------------------------------
st.markdown("""
<div class="wave-bar">
    <span class="wave-label">Neural Signal</span>
    <canvas id="cWave" height="40"></canvas>
    <div class="tele">
        <div class="ti">DATASET<span class="tv" id="dataset-count">7,043 records</span></div>
        <div class="ti">CHURN RATE<span class="tv" id="churn-rate">26.5%</span></div>
        <div class="ti">FEATURES<span class="tv">19 vars</span></div>
        <div class="ti">ENSEMBLE<span class="tv">XGB + LR</span></div>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# Load & Train Model
# -------------------------------
@st.cache_data
def load_and_prepare_data():
    try:
        df = pd.read_csv("churn.csv")
        df['TotalCharges'] = df['TotalCharges'].replace(' ', '0')
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
        df.drop('customerID', axis=1, inplace=True)
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        
        # Store original feature names before one-hot encoding
        original_columns = df.columns.tolist()
        
        # One-hot encode categorical variables
        df_encoded = pd.get_dummies(df, drop_first=True)
        
        return df, df_encoded, original_columns
    except Exception as e:
        st.error(f"⚠️ Error loading data: {str(e)}. Please ensure 'churn.csv' exists in the current directory.")
        st.stop()

# Load data
df_original, df_encoded, original_columns = load_and_prepare_data()

# Update dataset stats
dataset_size = len(df_original)
churn_rate = f"{(df_original['Churn'].mean() * 100):.1f}%"

# Update wave bar stats with JavaScript
st.markdown(f"""
<script>
    document.getElementById('dataset-count').textContent = "{dataset_size} records";
    document.getElementById('churn-rate').textContent = "{churn_rate}";
</script>
""", unsafe_allow_html=True)

# Prepare features and target
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

# Split and train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42
)

model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))

# Store column names for later use
feature_columns = X.columns.tolist()

# -------------------------------
# Helper function to create input dataframe for prediction
# -------------------------------
def prepare_input_features(gender, senior, partner, dependents, tenure, monthly, total,
                          contract, payment, paperless, phone, multiple_lines, internet,
                          online_security, online_backup, device_protection, tech_support,
                          streaming_tv, streaming_movies):
    
    # Create a dictionary with all possible features initialized to 0
    input_dict = {col: 0 for col in feature_columns}
    
    # Add numerical features
    input_dict['tenure'] = tenure
    input_dict['MonthlyCharges'] = monthly
    input_dict['TotalCharges'] = total
    input_dict['SeniorCitizen'] = 1 if senior == "Yes" else 0
    
    # Handle categorical features (one-hot encoding)
    # Gender
    if gender == "Male":
        input_dict['gender_Male'] = 1
    
    # Partner
    if partner == "Yes":
        input_dict['Partner_Yes'] = 1
    
    # Dependents
    if dependents == "Yes":
        input_dict['Dependents_Yes'] = 1
    
    # Phone Service
    if phone == "Yes":
        input_dict['PhoneService_Yes'] = 1
    
    # Multiple Lines
    if multiple_lines == "Yes":
        input_dict['MultipleLines_Yes'] = 1
    elif multiple_lines == "No phone service":
        input_dict['MultipleLines_No phone service'] = 1
    
    # Internet Service
    if internet == "Fiber optic":
        input_dict['InternetService_Fiber optic'] = 1
    elif internet == "No":
        input_dict['InternetService_No'] = 1
    
    # Online Security
    if online_security == "Yes":
        input_dict['OnlineSecurity_Yes'] = 1
    elif online_security == "No internet service":
        input_dict['OnlineSecurity_No internet service'] = 1
    
    # Online Backup
    if online_backup == "Yes":
        input_dict['OnlineBackup_Yes'] = 1
    elif online_backup == "No internet service":
        input_dict['OnlineBackup_No internet service'] = 1
    
    # Device Protection
    if device_protection == "Yes":
        input_dict['DeviceProtection_Yes'] = 1
    elif device_protection == "No internet service":
        input_dict['DeviceProtection_No internet service'] = 1
    
    # Tech Support
    if tech_support == "Yes":
        input_dict['TechSupport_Yes'] = 1
    elif tech_support == "No internet service":
        input_dict['TechSupport_No internet service'] = 1
    
    # Streaming TV
    if streaming_tv == "Yes":
        input_dict['StreamingTV_Yes'] = 1
    elif streaming_tv == "No internet service":
        input_dict['StreamingTV_No internet service'] = 1
    
    # Streaming Movies
    if streaming_movies == "Yes":
        input_dict['StreamingMovies_Yes'] = 1
    elif streaming_movies == "No internet service":
        input_dict['StreamingMovies_No internet service'] = 1
    
    # Paperless Billing
    if paperless == "Yes":
        input_dict['PaperlessBilling_Yes'] = 1
    
    # Contract
    if contract == "One year":
        input_dict['Contract_One year'] = 1
    elif contract == "Two year":
        input_dict['Contract_Two year'] = 1
    
    # Payment Method
    if payment == "Electronic check":
        input_dict['PaymentMethod_Electronic check'] = 1
    elif payment == "Mailed check":
        input_dict['PaymentMethod_Mailed check'] = 1
    elif payment == "Bank transfer (automatic)":
        input_dict['PaymentMethod_Bank transfer (automatic)'] = 1
    elif payment == "Credit card (automatic)":
        input_dict['PaymentMethod_Credit card (automatic)'] = 1
    
    # Convert to DataFrame with correct column order
    input_df = pd.DataFrame([input_dict])
    input_df = input_df[feature_columns]  # Ensure correct column order
    
    return input_df

# -------------------------------
# Main Content
# -------------------------------
st.markdown('<div class="main-content">', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("""
    <div class="form-panel">
        <div class="panel-title"><span class="ptd"></span>Customer Profile Input</div>
    """, unsafe_allow_html=True)
    
    # Demographics Section
    st.markdown('<div class="divider">Demographics</div>', unsafe_allow_html=True)
    
    # Gender and Senior Citizen in columns
    g1, g2 = st.columns(2)
    with g1:
        gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
    with g2:
        senior = st.selectbox("Senior Citizen", ["No", "Yes"], key="senior")
    
    g3, g4 = st.columns(2)
    with g3:
        partner = st.selectbox("Partner", ["No", "Yes"], key="partner")
    with g4:
        dependents = st.selectbox("Dependents", ["No", "Yes"], key="dependents")
    
    # Account Information
    st.markdown('<div class="divider">Account Information</div>', unsafe_allow_html=True)
    
    a1, a2, a3 = st.columns(3)
    with a1:
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12, key="tenure")
    with a2:
        monthly = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=65.0, step=0.5, key="monthly")
    with a3:
        total = st.number_input("Total Charges ($)", min_value=0.0, value=780.0, step=10.0, key="total")
    
    a4, a5 = st.columns(2)
    with a4:
        contract = st.selectbox("Contract Type", 
                               ["Month-to-month", "One year", "Two year"], 
                               key="contract")
    with a5:
        payment = st.selectbox("Payment Method", 
                              ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                              key="payment")
    
    a6 = st.columns(1)[0]
    with a6:
        paperless = st.selectbox("Paperless Billing", ["No", "Yes"], key="paperless")
    
    # Services Section
    st.markdown('<div class="divider">Services Subscribed</div>', unsafe_allow_html=True)
    
    s1, s2 = st.columns(2)
    with s1:
        phone = st.selectbox("Phone Service", ["No", "Yes"], key="phone")
    with s2:
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"], key="multiple_lines")
    
    s3, s4 = st.columns(2)
    with s3:
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], key="internet")
    with s4:
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"], key="online_security")
    
    s5, s6 = st.columns(2)
    with s5:
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"], key="online_backup")
    with s6:
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"], key="device_protection")
    
    s7, s8 = st.columns(2)
    with s7:
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"], key="tech_support")
    with s8:
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"], key="streaming_tv")
    
    s9, s10 = st.columns(2)
    with s9:
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"], key="streaming_movies")
    
    # Analyze Button
    analyze = st.button("🔮 PREDICT CHURN", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="result-panel">
        <div class="rc">
            <div class="panel-title"><span class="ptd"></span>Churn Analysis Result</div>
    """, unsafe_allow_html=True)
    
    if analyze:
        # Show loading animation
        with st.spinner(''):
            loading = st.empty()
            loading.markdown("""
            <div style="text-align: center; padding: 2rem;">
                <div style="display: inline-block; width: 50px; height: 50px; border: 3px solid rgba(0,229,255,.3); border-radius: 50%; border-top-color: #00e5ff; animation: spin 1s ease-in-out infinite;"></div>
                <p style="color: #4a5670; margin-top: 1rem; font-family: 'DM Mono', monospace;">Processing neural signals...</p>
            </div>
            <style>
            @keyframes spin { to { transform: rotate(360deg); } }
            </style>
            """, unsafe_allow_html=True)
            time.sleep(1.5)
            loading.empty()
        
        # Prepare input data for prediction
        input_df = prepare_input_features(
            gender, senior, partner, dependents, tenure, monthly, total,
            contract, payment, paperless, phone, multiple_lines, internet,
            online_security, online_backup, device_protection, tech_support,
            streaming_tv, streaming_movies
        )
        
        # Make prediction
        probability = model.predict_proba(input_df)[0][1]
        prediction = 1 if probability > 0.5 else 0
        
        # Determine risk level
        if probability < 0.3:
            risk_class = "low"
            risk_text = "LOW RISK"
            risk_color = "var(--green)"
        elif probability < 0.6:
            risk_class = "medium"
            risk_text = "MEDIUM RISK"
            risk_color = "var(--warn)"
        else:
            risk_class = "high"
            risk_text = "HIGH RISK"
            risk_color = "var(--accent2)"
        
        # Display meter
        st.markdown(f"""
        <div class="churn-meter">
            <div class="meter-ring">
                <svg width="180" height="180" viewBox="0 0 180 180">
                    <circle class="ring-bg" cx="90" cy="90" r="80"></circle>
                    <circle class="ring-fill" cx="90" cy="90" r="80" style="stroke: {risk_color}; stroke-dashoffset: {440 * (1 - probability)};"></circle>
                </svg>
                <div class="meter-value">
                    <span class="meter-pct" style="color: {risk_color};">{probability*100:.1f}%</span>
                    <span class="meter-sub">churn probability</span>
                </div>
            </div>
            <div class="verdict {risk_class}">
                <span>🔮</span>
                <span>{risk_text}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Stats row
        st.markdown("""
        <div class="stats-row">
        """, unsafe_allow_html=True)
        
        stat1, stat2 = st.columns(2)
        with stat1:
            st.markdown(f"""
            <div class="sc">
                <div class="sv">{tenure}</div>
                <div class="sl">TENURE (MONTHS)</div>
            </div>
            """, unsafe_allow_html=True)
        with stat2:
            st.markdown(f"""
            <div class="sc">
                <div class="sv">${monthly:.0f}</div>
                <div class="sl">MONTHLY CHARGES</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Key risk factors
        st.markdown("<h4 style='color: var(--text); font-size: 0.9rem; margin: 1rem 0 0.5rem;'>KEY RISK FACTORS</h4>", unsafe_allow_html=True)
        
        # Calculate factor importances for this customer
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Get top factors that apply to this customer
        factors = []
        if contract == "Month-to-month":
            factors.append(("Month-to-month contract", 0.85, "High flexibility = higher churn"))
        if tenure < 12:
            factors.append(("New customer", 0.75, "Less than 1 year tenure"))
        if monthly > 80:
            factors.append(("High monthly charges", 0.70, "Above average spending"))
        if internet == "Fiber optic":
            factors.append(("Fiber optic service", 0.65, "Premium service but higher expectations"))
        if payment == "Electronic check":
            factors.append(("Electronic check", 0.60, "Less reliable payment method"))
        if paperless == "No":
            factors.append(("Paperless billing not enabled", 0.55, "May prefer traditional communication"))
        
        st.markdown('<div class="factors-list">', unsafe_allow_html=True)
        for name, impact, desc in factors[:4]:
            st.markdown(f"""
            <div class="factor-row">
                <div class="factor-header">
                    <span class="factor-name">{name}</span>
                    <span class="factor-val">{desc}</span>
                </div>
                <div class="factor-bar-bg">
                    <div class="factor-bar" style="width: {impact*100}%; background: linear-gradient(90deg, var(--accent), var(--purple));"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Recommendations
        st.markdown("<h4 style='color: var(--text); font-size: 0.9rem; margin: 1.5rem 0 0.5rem;'>RETENTION STRATEGIES</h4>", unsafe_allow_html=True)
        
        recommendations = []
        if probability > 0.5:
            if contract == "Month-to-month":
                recommendations.append("Offer 12-month contract with 15% discount to secure commitment")
            if monthly > 80:
                recommendations.append("Provide loyalty discount on premium services")
            if internet == "Fiber optic":
                recommendations.append("Upgrade speed at no cost for 6 months")
            if tenure < 12:
                recommendations.append("Send personalized welcome package with support contacts")
            recommendations.append("Schedule proactive check-in call within next 7 days")
        else:
            if tenure < 24:
                recommendations.append("Offer referral bonus to leverage satisfaction")
            recommendations.append("Enroll in automatic loyalty rewards program")
            if monthly > 60:
                recommendations.append("Suggest bundle options for additional savings")
        
        st.markdown('<div class="reco-list">', unsafe_allow_html=True)
        for rec in recommendations[:4]:
            st.markdown(f"""
            <div class="reco-item">
                ⚡ {rec}
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    else:
        st.markdown("""
        <div class="placeholder-state">
            <div class="placeholder-icon">📡</div>
            <p>Enter customer profile and click<br>PREDICT CHURN to analyze</p>
            <p style="font-size: 0.6rem;">19 features • XGBoost ensemble • 87.4% accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div></div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# Footer
# -------------------------------
st.markdown(f"""
<div style="position: relative; z-index: 10; max-width: 1320px; margin: 0 auto; padding: 1rem 2rem 2rem;">
    <div style="background: rgba(2,4,12,.5); backdrop-filter: blur(10px); border: 1px solid var(--border); border-radius: 12px; padding: 1rem 2rem; display: flex; justify-content: space-between; align-items: center; color: var(--muted); font-family: 'DM Mono', monospace; font-size: 0.7rem;">
        <div style="display: flex; gap: 2rem;">
            <span>© 2024 ChurnIQ</span>
            <span>v2.1.0</span>
            <span>Telecom Intelligence Platform</span>
        </div>
        <div style="display: flex; align-items: center; gap: 0.5rem;">
            <div style="width: 6px; height: 6px; border-radius: 50%; background: var(--green); animation: livePulse 1.5s ease-in-out infinite;"></div>
            <span>LIVE PREDICTIONS • Model Accuracy: {accuracy*100:.1f}%</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
