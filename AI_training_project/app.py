import streamlit as st
import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="ChurnGuard Pro",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------------
# PREMIUM DARK THEME WITH RICH VISUALS
# -------------------------------
st.markdown("""
<style>
/* Import premium fonts */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

/* Global Styles */
.stApp {
    background: radial-gradient(ellipse at 30% 40%, #1a1a2e 0%, #16213e 50%, #0f0f1f 100%);
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* Animated gradient overlay */
.gradient-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: 
        radial-gradient(circle at 20% 30%, rgba(100, 80, 255, 0.15) 0%, transparent 40%),
        radial-gradient(circle at 80% 70%, rgba(255, 80, 200, 0.1) 0%, transparent 40%),
        radial-gradient(circle at 40% 80%, rgba(80, 200, 255, 0.1) 0%, transparent 40%);
    pointer-events: none;
    z-index: 0;
    animation: gradientShift 15s ease infinite;
}

@keyframes gradientShift {
    0%, 100% { opacity: 0.5; }
    50% { opacity: 1; }
}

/* Main container */
.main-container {
    position: relative;
    z-index: 1;
    max-width: 1400px;
    margin: 0 auto;
    padding: 2rem;
}

/* Premium Header */
.header {
    margin-bottom: 2.5rem;
    animation: slideDown 0.8s cubic-bezier(0.16, 1, 0.3, 1);
}

@keyframes slideDown {
    from { opacity: 0; transform: translateY(-30px); }
    to { opacity: 1; transform: translateY(0); }
}

.glass-header {
    background: rgba(20, 25, 45, 0.6);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 30px;
    padding: 2rem 2.5rem;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
}

.title-wrapper {
    display: flex;
    align-items: center;
    gap: 1.5rem;
}

.logo {
    width: 60px;
    height: 60px;
    background: linear-gradient(135deg, #4f46e5, #7c3aed, #c026d3);
    border-radius: 18px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 30px;
    box-shadow: 0 10px 20px rgba(79, 70, 229, 0.3);
    animation: float 3s ease-in-out infinite;
}

@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-5px); }
}

.title-text h1 {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #fff, #c7d2fe);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    letter-spacing: -0.5px;
}

.title-text p {
    color: #94a3b8;
    margin: 5px 0 0 0;
    font-size: 0.9rem;
    letter-spacing: 1px;
}

.header-stats {
    display: flex;
    gap: 2rem;
    margin-top: 1.5rem;
    padding-top: 1.5rem;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
}

.stat-item {
    display: flex;
    align-items: center;
    gap: 1rem;
}

.stat-label {
    color: #94a3b8;
    font-size: 0.85rem;
    font-weight: 500;
    letter-spacing: 0.5px;
}

.stat-value {
    background: rgba(79, 70, 229, 0.2);
    color: #a5b4fc;
    padding: 0.3rem 1rem;
    border-radius: 30px;
    font-size: 0.9rem;
    font-weight: 600;
    border: 1px solid rgba(79, 70, 229, 0.3);
}

/* Premium Cards */
.premium-card {
    background: rgba(20, 25, 45, 0.7);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 30px;
    padding: 2rem;
    height: 100%;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
    transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
    animation: cardAppear 0.6s cubic-bezier(0.16, 1, 0.3, 1);
}

@keyframes cardAppear {
    from { opacity: 0; transform: translateY(30px); }
    to { opacity: 1; transform: translateY(0); }
}

.premium-card:hover {
    transform: translateY(-5px);
    border-color: rgba(79, 70, 229, 0.3);
    box-shadow: 0 30px 50px rgba(0, 0, 0, 0.5);
}

.card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 2rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.card-header h3 {
    font-size: 1.2rem;
    font-weight: 600;
    color: white;
    margin: 0;
    letter-spacing: 0.5px;
}

.card-badge {
    background: rgba(79, 70, 229, 0.2);
    color: #a5b4fc;
    padding: 0.3rem 0.8rem;
    border-radius: 30px;
    font-size: 0.75rem;
    font-weight: 600;
    border: 1px solid rgba(79, 70, 229, 0.3);
}

/* Premium Inputs */
.input-group {
    margin-bottom: 1.5rem;
}

.input-label {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    color: #cbd5e1;
    font-size: 0.9rem;
    font-weight: 500;
    margin-bottom: 0.5rem;
}

.input-label span {
    color: #4f46e5;
}

/* Override Streamlit defaults */
.stSlider label, .stSelectbox label {
    color: #cbd5e1 !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
}

.stSlider > div > div {
    background: rgba(30, 35, 55, 0.8) !important;
    border-radius: 12px !important;
    padding: 0.3rem !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
}

.stSelectbox > div > div {
    background: rgba(30, 35, 55, 0.8) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 14px !important;
    color: white !important;
    transition: all 0.3s ease !important;
}

.stSelectbox > div > div:hover {
    border-color: #4f46e5 !important;
    box-shadow: 0 0 0 2px rgba(79, 70, 229, 0.2) !important;
}

/* Premium Button */
.premium-button {
    margin-top: 2rem;
}

.stButton > button {
    width: 100%;
    height: 56px;
    background: linear-gradient(135deg, #4f46e5, #7c3aed, #c026d3);
    color: white;
    border: none;
    border-radius: 18px;
    font-size: 1rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
    box-shadow: 0 10px 20px rgba(79, 70, 229, 0.3);
}

.stButton > button::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
    transition: left 0.5s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 20px 30px rgba(79, 70, 229, 0.4);
}

.stButton > button:hover::before {
    left: 100%;
}

/* Premium Result Card */
.result-premium {
    padding: 2.5rem;
    border-radius: 25px;
    text-align: center;
    margin-bottom: 1.5rem;
    animation: resultGlow 2s ease-in-out infinite;
}

@keyframes resultGlow {
    0%, 100% { box-shadow: 0 0 30px currentColor; }
    50% { box-shadow: 0 0 50px currentColor; }
}

.result-high {
    background: linear-gradient(135deg, rgba(127, 29, 29, 0.9), rgba(153, 27, 27, 0.9));
    border: 1px solid rgba(239, 68, 68, 0.3);
    color: #fecaca;
}

.result-low {
    background: linear-gradient(135deg, rgba(6, 78, 59, 0.9), rgba(4, 120, 87, 0.9));
    border: 1px solid rgba(16, 185, 129, 0.3);
    color: #bbf7d0;
}

.result-icon {
    font-size: 3.5rem;
    margin-bottom: 1rem;
    filter: drop-shadow(0 0 20px currentColor);
}

.result-title {
    font-size: 1rem;
    font-weight: 500;
    letter-spacing: 2px;
    margin-bottom: 0.5rem;
    opacity: 0.9;
}

.result-number {
    font-size: 5rem;
    font-weight: 800;
    line-height: 1;
    margin: 0.5rem 0;
    text-shadow: 0 0 30px currentColor;
}

.result-label {
    font-size: 0.9rem;
    opacity: 0.8;
}

/* Premium Progress Bar */
.stProgress > div > div {
    background: linear-gradient(90deg, #4f46e5, #7c3aed, #c026d3, #db2777) !important;
    background-size: 300% 100% !important;
    animation: gradientShift 3s ease infinite !important;
    border-radius: 100px !important;
    height: 12px !important;
    box-shadow: 0 0 20px rgba(79, 70, 229, 0.5) !important;
}

/* Premium Metrics Grid */
.metrics-premium {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin: 2rem 0;
}

.metric-premium {
    background: rgba(30, 35, 55, 0.8);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 20px;
    padding: 1.5rem 1rem;
    text-align: center;
    transition: all 0.3s ease;
    animation: metricPop 0.5s cubic-bezier(0.16, 1, 0.3, 1);
}

@keyframes metricPop {
    from { opacity: 0; transform: scale(0.9); }
    to { opacity: 1; transform: scale(1); }
}

.metric-premium:hover {
    transform: translateY(-5px);
    border-color: #4f46e5;
    box-shadow: 0 10px 20px rgba(79, 70, 229, 0.2);
}

.metric-premium .value {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #fff, #a5b4fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}

.metric-premium .label {
    color: #94a3b8;
    font-size: 0.8rem;
    font-weight: 500;
    letter-spacing: 0.5px;
}

/* Premium Insights */
.insights-premium {
    margin-top: 2rem;
}

.insight-premium {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 1rem;
    background: rgba(30, 35, 55, 0.8);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 16px;
    margin-bottom: 0.8rem;
    transition: all 0.3s ease;
    animation: insightSlide 0.5s cubic-bezier(0.16, 1, 0.3, 1);
}

@keyframes insightSlide {
    from { opacity: 0; transform: translateX(20px); }
    to { opacity: 1; transform: translateX(0); }
}

.insight-premium:hover {
    transform: translateX(5px);
    border-color: #4f46e5;
    background: rgba(40, 45, 65, 0.9);
}

.insight-icon {
    width: 40px;
    height: 40px;
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
}

.insight-content {
    flex: 1;
}

.insight-title {
    color: white;
    font-weight: 600;
    font-size: 0.9rem;
    margin-bottom: 0.2rem;
}

.insight-text {
    color: #94a3b8;
    font-size: 0.8rem;
}

/* Premium Footer */
.footer-premium {
    margin-top: 3rem;
    padding: 2rem;
    background: rgba(20, 25, 45, 0.5);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 30px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    color: #94a3b8;
    font-size: 0.9rem;
    animation: fadeIn 1s ease;
}

.footer-left {
    display: flex;
    align-items: center;
    gap: 1.5rem;
}

.footer-badge {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white;
    padding: 0.3rem 1rem;
    border-radius: 30px;
    font-size: 0.8rem;
    font-weight: 600;
}

.footer-right {
    display: flex;
    align-items: center;
    gap: 1rem;
}

.pulse-dot {
    width: 8px;
    height: 8px;
    background: #10b981;
    border-radius: 50%;
    animation: pulse 2s ease infinite;
}

@keyframes pulse {
    0%, 100% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.5); opacity: 0.5; }
}

/* Loading animation */
.loading-premium {
    text-align: center;
    padding: 3rem;
}

.loading-spinner {
    display: inline-block;
    width: 50px;
    height: 50px;
    border: 3px solid rgba(79, 70, 229, 0.3);
    border-radius: 50%;
    border-top-color: #4f46e5;
    animation: spin 1s ease-in-out infinite;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

/* Placeholder animation */
.placeholder-premium {
    text-align: center;
    padding: 3rem 2rem;
    animation: fadePulse 2s ease infinite;
}

@keyframes fadePulse {
    0%, 100% { opacity: 0.5; }
    50% { opacity: 1; }
}

.placeholder-icon {
    font-size: 5rem;
    margin-bottom: 1rem;
    filter: drop-shadow(0 0 20px #4f46e5);
}

/* No white lines - all borders are colored or transparent */
div, section, .stApp > div {
    border: none !important;
    outline: none !important;
}

/* Remove any default Streamlit borders */
.element-container, .stMarkdown, .stSlider, .stSelectbox {
    border: none !important;
}

.css-1dp5vir, .css-12oz5g7 {
    border: none !important;
    background: transparent !important;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Gradient Overlay
# -------------------------------
st.markdown('<div class="gradient-overlay"></div>', unsafe_allow_html=True)

# -------------------------------
# Main Container
# -------------------------------
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# -------------------------------
# Premium Header
# -------------------------------
st.markdown("""
<div class="header">
    <div class="glass-header">
        <div class="title-wrapper">
            <div class="logo">🛡️</div>
            <div class="title-text">
                <h1>ChurnGuard Pro</h1>
                <p>ENTERPRISE RETENTION ANALYTICS</p>
            </div>
        </div>
        <div class="header-stats">
            <div class="stat-item">
                <span class="stat-label">STATUS</span>
                <span class="stat-value">● ACTIVE</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">VERSION</span>
                <span class="stat-value">3.0.0</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">MODE</span>
                <span class="stat-value">REAL-TIME</span>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# Load & Train Model
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("churn.csv")
    df['TotalCharges'] = df['TotalCharges'].replace(' ', '0')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
    df.drop('customerID', axis=1, inplace=True)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    df = pd.get_dummies(df, drop_first=True)
    return df

try:
    with st.spinner(''):
        data = load_data()
        
        X = data.drop('Churn', axis=1)
        y = data['Churn']
        
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
        
except Exception as e:
    st.error("⚠️ Error loading data. Please ensure 'churn.csv' exists.")
    st.stop()

# -------------------------------
# Main Layout
# -------------------------------
col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.markdown("""
    <div class="premium-card">
        <div class="card-header">
            <h3>CUSTOMER PROFILE</h3>
            <span class="card-badge">INPUT</span>
        </div>
    """, unsafe_allow_html=True)
    
    # Premium inputs
    tenure = st.slider("📊 Tenure (months)", 0, 72, 12)
    monthly = st.slider("💰 Monthly Charges ($)", 0, 200, 70)
    
    contract = st.selectbox(
        "📝 Contract Type",
        ["Month-to-month", "One year", "Two year"]
    )
    
    internet = st.selectbox(
        "🌐 Internet Service",
        ["DSL", "Fiber optic", "No"]
    )
    
    payment = st.selectbox(
        "💳 Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
    )
    
    st.markdown('<div class="premium-button">', unsafe_allow_html=True)
    analyze = st.button("🚀 ANALYZE NOW", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="premium-card">
        <div class="card-header">
            <h3>ANALYSIS RESULT</h3>
            <span class="card-badge">LIVE</span>
        </div>
    """, unsafe_allow_html=True)
    
    if analyze:
        # Show loading
        with st.spinner(''):
            loading = st.empty()
            loading.markdown("""
            <div class="loading-premium">
                <div class="loading-spinner"></div>
                <p style="color: #94a3b8; margin-top: 1rem;">Processing...</p>
            </div>
            """, unsafe_allow_html=True)
            time.sleep(1.5)
            loading.empty()
        
        # Prepare input
        input_dict = {
            "tenure": tenure,
            "MonthlyCharges": monthly,
            "TotalCharges": tenure * monthly
        }
        
        input_df = pd.DataFrame([input_dict])
        
        for col in X.columns:
            if col not in input_df.columns:
                input_df[col] = 0
        
        for col in input_df.columns:
            if contract in col:
                input_df[col] = 1
            if internet in col:
                input_df[col] = 1
            if payment in col:
                input_df[col] = 1
        
        input_df = input_df[X.columns]
        
        # Predict
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        
        # Display result
        if prediction == 1:
            st.markdown(f"""
            <div class="result-premium result-high">
                <div class="result-icon">⚠️</div>
                <div class="result-title">HIGH RISK DETECTED</div>
                <div class="result-number">{probability*100:.1f}%</div>
                <div class="result-label">churn probability</div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(int(probability * 100))
        else:
            st.markdown(f"""
            <div class="result-premium result-low">
                <div class="result-icon">✅</div>
                <div class="result-title">LOW RISK DETECTED</div>
                <div class="result-number">{(1-probability)*100:.1f}%</div>
                <div class="result-label">retention probability</div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(int((1-probability) * 100))
        
        # Metrics
        st.markdown("""
        <div class="metrics-premium">
        """, unsafe_allow_html=True)
        
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""
            <div class="metric-premium">
                <div class="value">{tenure}</div>
                <div class="label">MONTHS</div>
            </div>
            """, unsafe_allow_html=True)
        
        with m2:
            st.markdown(f"""
            <div class="metric-premium">
                <div class="value">${monthly}</div>
                <div class="label">MONTHLY</div>
            </div>
            """, unsafe_allow_html=True)
        
        with m3:
            st.markdown(f"""
            <div class="metric-premium">
                <div class="value">{accuracy*100:.0f}%</div>
                <div class="label">ACCURACY</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Insights
        st.markdown("""
        <div class="insights-premium">
            <h4 style="color: white; font-size: 0.9rem; margin-bottom: 1rem;">KEY INSIGHTS</h4>
        """, unsafe_allow_html=True)
        
        insights = []
        if tenure < 12:
            insights.append(("🆕", "New Customer", "Higher risk in first year"))
        if monthly > 100:
            insights.append(("💰", "Premium Plan", "Above average spending"))
        if contract == "Month-to-month":
            insights.append(("📋", "Flexible Contract", "No commitment - higher flexibility"))
        
        for icon, title, text in insights:
            st.markdown(f"""
            <div class="insight-premium">
                <div class="insight-icon">{icon}</div>
                <div class="insight-content">
                    <div class="insight-title">{title}</div>
                    <div class="insight-text">{text}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
    else:
        st.markdown("""
        <div class="placeholder-premium">
            <div class="placeholder-icon">✨</div>
            <h3 style="color: white; margin-bottom: 0.5rem;">Ready to Analyze</h3>
            <p style="color: #94a3b8;">Enter customer details and click ANALYZE NOW</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# Premium Footer
# -------------------------------
st.markdown(f"""
<div class="footer-premium">
    <div class="footer-left">
        <span class="footer-badge">PRO</span>
        <span>© 2024 ChurnGuard</span>
        <span>•</span>
        <span>v3.0.0</span>
    </div>
    <div class="footer-right">
        <div class="pulse-dot"></div>
        <span>Live Predictions • {len(X)} customers analyzed</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)