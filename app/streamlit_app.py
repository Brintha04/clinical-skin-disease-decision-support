import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Load Model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(os.path.join(BASE_DIR, "models", "skin_disease_model.pkl"))
features = joblib.load(os.path.join(BASE_DIR, "models", "features.pkl"))

# Page Config
st.set_page_config(page_title="Clinical Skin Diagnosis Assistant", layout="wide")

st.title("🩺 Clinical Skin Disease Decision Support")
st.markdown("Enter clinical findings to assist diagnosis")

# Severity Mapping
severity_map = {"None": 0, "Mild": 1, "Moderate": 2, "Severe": 3}
reverse_map = {v: k for k, v in severity_map.items()}

def severity_input(label, help_text):
    return severity_map[
        st.radio(label, ["None", "Mild", "Moderate", "Severe"],
                 horizontal=True, help=help_text)
    ]

# Input Sections
with st.expander("🔴 Primary Clinical Symptoms", expanded=True):
    erythema = severity_input("Erythema (Redness)", "Skin redness")
    scaling = severity_input("Scaling", "Flaking skin")
    itching = severity_input("Pruritus (Itching)", "Patient-reported itching")

with st.expander("🟠 Morphological Features"):
    definite_borders = severity_input("Well-defined borders", "Sharp lesion edges")
    koebner = severity_input("Koebner Phenomenon", "Lesions at trauma sites")

with st.expander("🟡 Lesion Characteristics"):
    polygonal_papules = severity_input("Polygonal Papules", "Flat lesions")
    follicular_papules = severity_input("Follicular Papules", "Hair follicle lesions")

with st.expander("🟢 Body Area Involvement"):
    oral = severity_input("Oral Mucosal Involvement", "Inside mouth lesions")
    knee_elbow = severity_input("Knee & Elbow Involvement", "Common psoriasis sites")

# Build Input
input_dict = {col: 0 for col in features}
input_dict.update({
    "erythema": erythema,
    "scaling": scaling,
    "itching": itching,
    "definite_borders": definite_borders,
    "koebner_phenomenon": koebner,
    "polygonal_papules": polygonal_papules,
    "follicular_papules": follicular_papules,
    "oral_mucosal_involvement": oral,
    "knee_and_elbow_involvement": knee_elbow
})

# Disease Mapping
disease_map = {
    1: "Psoriasis",
    2: "Seborrheic Dermatitis",
    3: "Lichen Planus",
    4: "Pityriasis Rosea",
    5: "Chronic Dermatitis",
    6: "Pityriasis Rubra Pilaris"
}

# Predict
st.markdown("---")

if st.button("🧠 Generate Clinical Prediction"):

    df = pd.DataFrame([input_dict])
    df = df.reindex(columns=features)

    probs = model.predict_proba(df)[0]
    top_indices = np.argsort(probs)[::-1][:2]

    top1_idx, top2_idx = top_indices
    top1_prob = probs[top1_idx]
    top2_prob = probs[top2_idx]

    top1_class = top1_idx + 1
    top2_class = top2_idx + 1

    disease1 = disease_map.get(top1_class, "Unknown")
    disease2 = disease_map.get(top2_class, "Unknown")

    # Patient Summary
    summary_parts = []
    for k, v in input_dict.items():
        if v > 0:
            severity = reverse_map[v].lower()
            feature_clean = k.replace("_", " ")
            summary_parts.append(f"{severity} {feature_clean}")

    patient_summary = (
        "Patient presents with " + ", ".join(summary_parts) + "."
        if summary_parts else "No significant clinical symptoms reported."
    )

    # Output UI
    col1, col2 = st.columns(2)

    with col1:
        st.success(f"🩺 Primary Diagnosis: {disease1}")
        st.progress(float(top1_prob))
        st.write(f"{round(top1_prob*100,2)}%")

    with col2:
        st.warning(f"⚠️ Differential Diagnosis: {disease2}")
        st.progress(float(top2_prob))
        st.write(f"{round(top2_prob*100,2)}%")

    # Interpretation
    if top1_prob > 0.85:
        st.markdown("**High confidence prediction. Clinical correlation recommended.**")
    elif top1_prob > 0.65:
        st.markdown("**Moderate confidence. Consider differential diagnosis.**")
    else:
        st.markdown("**Low confidence. Further evaluation required.**")

    # Patient Summary
    st.subheader("🧾 Patient Clinical Summary")
    st.write(patient_summary)

    # Download Report
    report_text = f"""
Skin Disease Prediction Report

Primary Diagnosis: {disease1} ({round(top1_prob*100,2)}%)
Differential Diagnosis: {disease2} ({round(top2_prob*100,2)}%)

Clinical Summary:
{patient_summary}
"""

    st.download_button(
        "📥 Download Report (TXT)",
        report_text,
        file_name="skin_report.txt"
    )

    st.download_button(
        "📥 Download Input Data (CSV)",
        df.to_csv(index=False),
        file_name="patient_data.csv"
    )