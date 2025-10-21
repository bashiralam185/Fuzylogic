import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# App setup
st.set_page_config(page_title="Fuzzy Personality Classifier", layout="wide")
st.title("Fuzzy Personality Classifier")
st.markdown(
    "Classify along the **Introvert–Ambivert–Extrovert** spectrum "
    "and estimate **Emotional Stability** using fuzzy logic."
)

# 1. Define universes
U10 = np.arange(0, 10.1, 0.1)
U100 = np.arange(0, 100.5, 0.5)

# 2. Inputs (Antecedents)
social = ctrl.Antecedent(U10, 'social')
talk = ctrl.Antecedent(U10, 'talk')
confidence = ctrl.Antecedent(U10, 'confidence')
energy = ctrl.Antecedent(U10, 'energy')
alone = ctrl.Antecedent(U10, 'alone')
initiate = ctrl.Antecedent(U10, 'initiate')
group = ctrl.Antecedent(U10, 'group')
public_speak = ctrl.Antecedent(U10, 'public_speak')
express = ctrl.Antecedent(U10, 'express')

# 3. Outputs (Consequents)
orientation = ctrl.Consequent(U100, 'orientation')  # Existing
stability = ctrl.Consequent(U100, 'stability')      # New

# Membership functions helper
def add_low_med_high(var):
    var['low']  = fuzz.trapmf(var.universe, [0, 0, 3, 5])
    var['med']  = fuzz.trimf(var.universe, [4, 5.5, 7])
    var['high'] = fuzz.trapmf(var.universe, [6, 8, 10, 10])

for v in [social, talk, confidence, energy, initiate, group, public_speak, alone]:
    add_low_med_high(v)

# Expressiveness custom
express['reserved']   = fuzz.trapmf(express.universe, [0, 0, 3, 5])
express['balanced']   = fuzz.trimf(express.universe, [4, 5.5, 7])
express['expressive'] = fuzz.trapmf(express.universe, [6, 8, 10, 10])

# Orientation output membership functions
orientation['introvert'] = fuzz.trimf(orientation.universe, [0, 0, 50])
orientation['ambivert']  = fuzz.trimf(orientation.universe, [30, 50, 70])
orientation['extrovert'] = fuzz.trimf(orientation.universe, [60, 100, 100])

# Emotional Stability output membership functions
stability['low']      = fuzz.trimf(stability.universe, [0, 0, 50])
stability['balanced'] = fuzz.trimf(stability.universe, [30, 50, 70])
stability['high']     = fuzz.trimf(stability.universe, [60, 100, 100])

# 4. Rules

# Personality Orientation rules
orientation_rules = [
    ctrl.Rule(social['high'] & talk['high'] & confidence['high'], orientation['extrovert']),
    ctrl.Rule(initiate['high'] & (talk['med'] | confidence['high']), orientation['extrovert']),
    ctrl.Rule(group['high'] & public_speak['high'], orientation['extrovert']),
    ctrl.Rule(energy['high'] & express['expressive'], orientation['extrovert']),
    ctrl.Rule(alone['low'] & (social['high'] | initiate['high']), orientation['extrovert']),
    ctrl.Rule(social['low'] & talk['low'] & alone['high'], orientation['introvert']),
    ctrl.Rule(group['low'] & public_speak['low'], orientation['introvert']),
    ctrl.Rule(energy['low'] & express['reserved'], orientation['introvert']),
    ctrl.Rule(initiate['low'] & confidence['low'], orientation['introvert']),
    ctrl.Rule(alone['high'] & (social['low'] | talk['low']), orientation['introvert']),
    ctrl.Rule(social['med'] & talk['med'], orientation['ambivert']),
    ctrl.Rule(confidence['med'] & alone['med'], orientation['ambivert']),
    ctrl.Rule(group['med'] & public_speak['med'], orientation['ambivert']),
    ctrl.Rule(energy['med'] & express['balanced'], orientation['ambivert']),
    ctrl.Rule(social['high'] & alone['med'], orientation['ambivert']),
    ctrl.Rule(talk['high'] & express['reserved'] & confidence['med'], orientation['ambivert']),
    ctrl.Rule(initiate['med'] & (group['med'] | public_speak['med']), orientation['ambivert']),
    ctrl.Rule(confidence['high'] & public_speak['low'] & alone['med'], orientation['ambivert'])
]

# Emotional Stability rules
stability_rules = [
    # Low Stability
    ctrl.Rule(confidence['low'] & express['expressive'], stability['low']),
    ctrl.Rule(energy['high'] & alone['high'], stability['low']),
    ctrl.Rule(confidence['low'] & energy['low'], stability['low']),
    ctrl.Rule(group['low'] & public_speak['low'], stability['low']),
    # Balanced
    ctrl.Rule(confidence['med'] & express['balanced'], stability['balanced']),
    ctrl.Rule(energy['med'] & alone['med'], stability['balanced']),
    ctrl.Rule(social['med'] & talk['med'], stability['balanced']),
    ctrl.Rule(group['med'] & public_speak['med'], stability['balanced']),
    # High
    ctrl.Rule(confidence['high'] & express['balanced'], stability['high']),
    ctrl.Rule(energy['med'] & confidence['high'], stability['high']),
    ctrl.Rule(group['high'] & public_speak['high'], stability['high']),
    ctrl.Rule(alone['low'] & confidence['high'], stability['high']),
    ctrl.Rule(social['high'] & express['balanced'], stability['high'])
]

# 5. Control Systems
orientation_system = ctrl.ControlSystem(orientation_rules)
stability_system = ctrl.ControlSystem(stability_rules)

sim_orient = ctrl.ControlSystemSimulation(orientation_system)
sim_stab = ctrl.ControlSystemSimulation(stability_system)

# 6. UI Inputs
st.sidebar.header("Provide Your Ratings (0–10)")

def slider(label, help_text):
    return st.sidebar.slider(label, 0.0, 10.0, 4.5, 0.5, help=help_text)

inp_social   = slider("Social Interaction Preference", "Enjoyment of social gatherings and being around people.")
inp_talk     = slider("Talkativeness Level", "Frequency and ease of engaging in conversation.")
inp_conf     = slider("Confidence in Social Settings", "Comfort meeting new people or being in crowds.")
inp_energy   = slider("Energy Expression", "Observable enthusiasm and liveliness in interactions.")
inp_alone    = slider("Alone Time Need", "Preference for solitude to recharge (higher → more introvert).")
inp_initiate = slider("Initiation of Conversation", "Tendency to start conversations or introduce yourself.")
inp_group    = slider("Group Participation Drive", "Desire to join and contribute in group discussions.")
inp_public   = slider("Comfort in Public Speaking", "Ease of presenting ideas to a group or audience.")
inp_express  = slider("Emotional Expressiveness", "How openly you display feelings in social settings.")

# Feed both systems
inputs = {
    'social': inp_social,
    'talk': inp_talk,
    'confidence': inp_conf,
    'energy': inp_energy,
    'alone': inp_alone,
    'initiate': inp_initiate,
    'group': inp_group,
    'public_speak': inp_public,
    'express': inp_express
}

# Feed orientation 
for k, v in inputs.items():
    sim_orient.input[k] = v

# Feed stability (only relevant inputs)
for k in ['social', 'talk', 'confidence', 'energy', 'alone', 'group', 'public_speak', 'express']:
    sim_stab.input[k] = inputs[k]

# 7. Compute
try:
    sim_orient.compute()
    sim_stab.compute()
except Exception as e:
    st.error(f"Simulation error: {e}")
    st.stop()

# 8. Results
score_orient = float(sim_orient.output['orientation'])
score_stab = float(sim_stab.output['stability'])

def interpret_orientation(score):
    if score < 40: return "Introvert"
    elif score > 60: return "Extrovert"
    else: return "Ambivert"

def interpret_stability(score):
    if score < 40: return "Low"
    elif score > 60: return "High"
    else: return "Balanced"

label_orient = interpret_orientation(score_orient)
label_stab = interpret_stability(score_stab)

# 9. Gauges
def gauge_chart(value, label, title, colors):
    return go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": " / 100"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"thickness": 0.25},
            "steps": [
                {"range": [0, 40], "color": colors[0]},
                {"range": [40, 60], "color": colors[1]},
                {"range": [60, 100], "color": colors[2]}
            ],
            "threshold": {"line": {"color": "black", "width": 3}, "value": value}
        },
        title={"text": f"{title} → <b>{label}</b>"}
    ))

gauge_orient = gauge_chart(score_orient, label_orient, "Personality Orientation",
                           ["#cfe3ff", "#e8f7e5", "#ffe1de"])
gauge_stab   = gauge_chart(score_stab, label_stab, "Emotional Stability",
                           ["#ffd9d9", "#fff8d6", "#d4f4d7"])

# 9b. Fuzzy Membership Visualization for Outputs

# Orientation MFs
y_intro = orientation['introvert'].mf
y_ambi  = orientation['ambivert'].mf
y_ext   = orientation['extrovert'].mf

mf_fig_orient = go.Figure()
mf_fig_orient.add_trace(go.Scatter(x=U100, y=y_intro, name="Introvert",
                                   fill="tozeroy", fillcolor="rgba(147,197,253,0.4)", line_color="blue"))
mf_fig_orient.add_trace(go.Scatter(x=U100, y=y_ambi, name="Ambivert",
                                   fill="tozeroy", fillcolor="rgba(144,238,144,0.4)", line_color="green"))
mf_fig_orient.add_trace(go.Scatter(x=U100, y=y_ext, name="Extrovert",
                                   fill="tozeroy", fillcolor="rgba(255,160,122,0.4)", line_color="red"))
mf_fig_orient.add_vline(x=score_orient, line_width=3, line_dash="dash", line_color="black")
mf_fig_orient.update_layout(
    title="Fuzzy Output Sets: Personality Orientation",
    xaxis_title="Orientation (0 = Introvert, 100 = Extrovert)",
    yaxis_title="Membership Degree",
    legend_title="Output Categories",
    hovermode="x unified"
)

# Stability MFs
y_low = stability['low'].mf
y_bal = stability['balanced'].mf
y_high = stability['high'].mf

mf_fig_stab = go.Figure()
mf_fig_stab.add_trace(go.Scatter(x=U100, y=y_low, name="Low Stability",
                                 fill="tozeroy", fillcolor="rgba(255,182,193,0.4)", line_color="red"))
mf_fig_stab.add_trace(go.Scatter(x=U100, y=y_bal, name="Balanced Stability",
                                 fill="tozeroy", fillcolor="rgba(255,255,153,0.4)", line_color="orange"))
mf_fig_stab.add_trace(go.Scatter(x=U100, y=y_high, name="High Stability",
                                 fill="tozeroy", fillcolor="rgba(144,238,144,0.4)", line_color="green"))
mf_fig_stab.add_vline(x=score_stab, line_width=3, line_dash="dash", line_color="black")
mf_fig_stab.update_layout(
    title="Fuzzy Output Sets: Emotional Stability",
    xaxis_title="Stability (0 = Low, 100 = High)",
    yaxis_title="Membership Degree",
    legend_title="Output Categories",
    hovermode="x unified"
)

# 10. Layout
col1, col2 = st.columns(2)
with col1:
    st.subheader("Personality Orientation")
    st.metric("Orientation Score (0–100)", f"{score_orient:.1f}")
    st.write(f"**Interpretation:** {label_orient}")
    st.plotly_chart(gauge_orient, use_container_width=True)
    st.plotly_chart(mf_fig_orient, use_container_width=True)

with col2:
    st.subheader("Emotional Stability")
    st.metric("Stability Score (0–100)", f"{score_stab:.1f}")
    st.write(f"**Interpretation:** {label_stab}")
    st.plotly_chart(gauge_stab, use_container_width=True)
    st.plotly_chart(mf_fig_stab, use_container_width=True)

# 11. Summary
st.divider()
st.markdown("#### 🧭 Combined Summary")
st.write(f"You appear to be **{label_orient}** with **{label_stab} Emotional Stability**.")
if label_orient == "Extrovert" and label_stab == "High":
    st.info("Energetic, socially confident, and emotionally steady — a natural leader.")
elif label_orient == "Introvert" and label_stab == "High":
    st.info("Calm, thoughtful, and emotionally composed — introspective yet resilient.")
elif label_stab == "Low":
    st.warning("Shows some emotional reactivity — consider balancing energy and confidence for steadier composure.")
else:
    st.success("Moderately balanced in both sociability and emotional regulation.")
