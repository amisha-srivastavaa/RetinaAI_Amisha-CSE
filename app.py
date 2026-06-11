import streamlit as st
import pandas as pd
import numpy as np
import os
import time

# --- SETUP AND CONFIG ---
st.set_page_config(page_title="Retina AI: Early Warning System", page_icon="👁️", layout="wide")

# --- MOCKUP DATA LOADER ---
@st.cache_data
def load_data():
    # Try to load real data if it exists, otherwise generate dummy data for the demo
    if os.path.exists('test.csv'):
        df = pd.read_csv('test.csv')
    else:
        # Fallback dummy data just to keep the app from crashing
        df = pd.DataFrame({
            'student_id': [f"STU{str(i).zfill(4)}" for i in range(1, 101)],
            'cgpa_sem4': np.random.uniform(5.0, 9.5, 100),
            'branch': np.random.choice(['CSE', 'IT', 'ECE', 'ME'], 100),
            'screen_time_hours': np.random.uniform(2, 10, 100)
        })
    return df

df = load_data()

# --- SIDEBAR: STUDENT SELECTION ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135768.png", width=100)
st.sidebar.title("Counsellor Portal")
selected_student = st.sidebar.selectbox("Select Student ID", df['student_id'].unique())

# Fetch student data
student_data = df[df['student_id'] == selected_student].iloc[0]

# --- MAIN DASHBOARD ---
st.title("👁️ Retina AI: Student Risk Dashboard")
st.markdown("---")

col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    st.subheader("Student Profile")
    st.write(f"**ID:** {selected_student}")
    st.write(f"**Branch:** {student_data.get('branch', 'N/A')}")
    st.write(f"**Latest CGPA:** {student_data.get('cgpa_sem4', 'N/A')}")
    st.write(f"**Screen Time:** {student_data.get('screen_time_hours', 'N/A')} hrs/day")
    
with col2:
    st.subheader("Multimodal Prediction Engine")
    
    # Simulate loading model
    if st.button("Run Risk Analysis"):
        with st.spinner("Processing Tabular Data (MLP)..."):
            pass
        with st.spinner("Analyzing Attendance Trends (LSTM)..."):
            pass
        with st.spinner("Running NLP on Counsellor Notes (DistilBERT)..."):
            pass
        
        # Mock prediction logic (Deterministic based on hash of ID for consistency in demo)
        risk_hash = hash(selected_student) % 3
        
        if risk_hash == 0:
            st.success("🟢 **LOW RISK** - Student is on track.")
            is_high_risk = False
        elif risk_hash == 1:
            st.warning("🟠 **MEDIUM RISK** - Recommend casual check-in.")
            is_high_risk = False
        else:
            st.error("🔴 **HIGH RISK DETECTED** - Immediate Intervention Required.")
            is_high_risk = True
            
        # Store state
        st.session_state['is_high_risk'] = is_high_risk
        st.session_state['analyzed'] = True

# --- INTERVENTION WORKFLOW ---
st.markdown("---")
if st.session_state.get('analyzed', False) and st.session_state.get('is_high_risk', False):
    st.header("⚡ Actionable Interventions")
    
    st.markdown("### Why was this student flagged?")
    # Mock SHAP Values
    chart_data = pd.DataFrame(
       np.array([[85, 'Recent Attendance Drop'], [60, 'Negative Sentiment in Notes'], [40, 'Active Backlogs']]),
       columns=['Importance', 'Feature']
    )
    chart_data['Importance'] = pd.to_numeric(chart_data['Importance'])
    st.bar_chart(chart_data, x='Feature', y='Importance', color="#ff4b4b")
    
    st.markdown("### Trigger Workflows")
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.info("🧠 **Mental Health Partner**")
        if st.button("Generate YourDOST Free Subscription"):
            st.toast("✅ Subscription Link sent to student email!")
            st.success("Link Sent!")
            
    with c2:
        st.warning("📅 **Mandatory Check-in**")
        if st.button("Schedule Session & Alert Authority"):
            st.toast("✅ Calendar invite sent and Admin alerted!")
            st.success("Scheduled for Tuesday 10:00 AM")
            
    with c3:
        st.success("🤝 **Alumni Support Network**")
        if st.button("Assign Alumni Peer Mentor"):
            st.toast("✅ Matched with Alumni Mentor (Class of '24)")
            st.success("Mentor Assigned")
            
elif st.session_state.get('analyzed', False):
    st.info("No immediate interventions required at this time.")
