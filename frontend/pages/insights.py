"""
PRISM Frontend — Clinical Insights Dashboard

End-to-end patient cough analysis:
    1. Record or upload audio via microphone / file upload
    2. CNN Cough Detector detects coughs & generates embeddings
    3. Temporal Transformer predicts 30-day trajectory (or demo)
    4. RATM pipeline generates evidence-backed clinical insights

Launch:
    streamlit run frontend/pages/insights.py
"""

from __future__ import annotations

import dataclasses
import datetime
import io
import os
import random
import string
import sys

# Ensure the project root is on sys.path so that first-party packages
# (retrieval, models, backend, etc.) can be resolved when Streamlit
# launches this page directly.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402
import torch  # noqa: E402

from frontend.styles import inject_css, inject_sidebar_nav  # noqa: E402

st.set_page_config(
    page_title="PRISM - Clinical Insights",
    page_icon="🫁",
    layout="wide",
)

# Inject CSS and sidebar
inject_css()
inject_sidebar_nav("insights")

# ──────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────

TRAJECTORY_INFO = {
    0: ("Stable", "Consistent cough pattern with minimal variation"),
    1: ("Improving", "Cough frequency is decreasing over time"),
    2: ("Increasing", "Cough frequency is rising over time"),
    3: ("Abnormal", "Irregular spike pattern detected"),
}

# ──────────────────────────────────────────────────────────────────
# Cached model loading
# ──────────────────────────────────────────────────────────────────


@st.cache_resource
def load_cough_detector():
    """Load the CNN cough detector with checkpoint.
    Prefers the fine-tuned mic-robust checkpoint if available,
    falls back to the original training checkpoint.
    """
    import torch

    from models.cough_detector.model import build_model
    from models.shared.checkpoint import load_checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=1, pretrained=False, device=device)

    checkpoints = [
        os.path.join("models", "checkpoints", "cough_detector_finetuned.pt"),
        os.path.join("models", "checkpoints", "cough_detector_best.pt"),
    ]
    loaded = False
    for ckpt in checkpoints:
        if os.path.exists(ckpt):
            load_checkpoint(ckpt, model, device=device)
            loaded = True
            break

    if not loaded:
        st.sidebar.warning("No checkpoint found. Using untrained weights.")

    model.eval()
    return model, device


@st.cache_resource
def load_ratm_pipeline():
    """Load the RATM pipeline."""
    from retrieval.ratm_pipeline import RATMPipeline

    return RATMPipeline(use_retrieval=False)


@st.cache_resource
def load_disease_classifier():
    """Load and cache the disease classifier for embedding-based predictions."""
    from models.disease_classifier.classifier import (
        DISEASE_CLASSES,
        DiseaseClassifierHead,
    )

    classifier = DiseaseClassifierHead(input_dim=512, num_classes=len(DISEASE_CLASSES))
    ckpt_path = os.path.join("models", "checkpoints", "disease_classifier_v1.pt")

    if os.path.exists(ckpt_path):
        classifier.load_state_dict(
            torch.load(ckpt_path, map_location="cpu", weights_only=True)
        )

    classifier.eval()
    return classifier


def audio_to_mel(audio_bytes: bytes, sample_rate: int = 16000) -> tuple:
    import librosa
    import soundfile as sf

    with io.BytesIO(audio_bytes) as buf:
        waveform, sr = sf.read(buf, dtype="float32")

    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    if sr != sample_rate:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=sample_rate)

    peak = np.abs(waveform).max()
    if peak > 1e-6:
        waveform = waveform / peak

    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_mels=128,
        n_fft=2048,
        hop_length=512,
        fmin=20,
        fmax=8000,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)

    return log_mel, waveform, sample_rate


def _segment_waveform(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    segment_sec: float = 3.0,
    overlap_sec: float = 0.5,
) -> list[np.ndarray]:
    seg_len = int(segment_sec * sample_rate)
    hop = int((segment_sec - overlap_sec) * sample_rate)
    total = len(waveform)

    if total <= seg_len:
        padded = np.zeros(seg_len, dtype=waveform.dtype)
        padded[:total] = waveform
        return [padded]

    segments = []
    start = 0
    while start + seg_len <= total:
        segments.append(waveform[start : start + seg_len])
        start += hop

    if start < total:
        tail = np.zeros(seg_len, dtype=waveform.dtype)
        tail[: total - start] = waveform[start:]
        segments.append(tail)

    return segments


def _mel_from_segment(segment: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    import librosa

    mel = librosa.feature.melspectrogram(
        y=segment,
        sr=sample_rate,
        n_mels=128,
        n_fft=2048,
        hop_length=512,
        fmin=20,
        fmax=8000,
    )
    return librosa.power_to_db(mel, ref=np.max)


def classify_audio(
    mel_spec: np.ndarray, model, device, waveform=None, sr=16000
) -> dict:
    if waveform is not None:
        segments = _segment_waveform(waveform, sample_rate=sr)
        best_prob = -1.0
        best_embedding = None
        segment_probs = []

        for seg in segments:
            seg_mel = _mel_from_segment(seg, sr)
            seg_tensor = (
                torch.tensor(seg_mel, dtype=torch.float32)
                .unsqueeze(0)
                .unsqueeze(0)
                .to(device)
            )

            # CRITICAL FIX: Normalize exactly like training (Z-score)
            mean = seg_tensor.mean()
            std = seg_tensor.std().clamp(min=1e-6)
            seg_tensor = (seg_tensor - mean) / std

            with torch.no_grad():
                logits, embeddings = model(seg_tensor)
                prob = torch.sigmoid(logits).item()
                segment_probs.append(prob)

                if prob > best_prob:
                    best_prob = prob
                    best_embedding = embeddings.squeeze(0).cpu().numpy()

        return {
            "is_cough": best_prob > 0.5,
            "probability": best_prob,
            "embedding": best_embedding,
            "num_segments": len(segments),
            "segment_probs": segment_probs,
        }

    else:
        tensor_spec = (
            torch.tensor(mel_spec, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )

        # Normalize
        mean = tensor_spec.mean()
        std = tensor_spec.std().clamp(min=1e-6)
        tensor_spec = (tensor_spec - mean) / std

        with torch.no_grad():
            logits, embeddings = model(tensor_spec)
            prob = torch.sigmoid(logits).item()

        return {
            "is_cough": prob > 0.5,
            "probability": prob,
            "embedding": embeddings.squeeze(0).cpu().numpy(),
            "num_segments": 1,
            "segment_probs": [prob],
        }


def generate_patient_id():
    """Generate a random clinical-looking patient ID."""
    prefix = random.choice(["PRISM", "PED", "PULM"])
    nums = "".join(random.choices(string.digits, k=6))
    return f"{prefix}-{nums}"


# ──────────────────────────────────────────────────────────────────
# Page layout
# ──────────────────────────────────────────────────────────────────


def main():
    st.markdown(
        '<h1 class="prism-title">Clinical Insights</h1>', unsafe_allow_html=True
    )
    st.markdown(
        '<p class="prism-subtitle">Acoustic Analysis & Trajectory Prediction</p>',
        unsafe_allow_html=True,
    )

    # Initialize session state for patient ID
    if "patient_id" not in st.session_state:
        st.session_state.patient_id = generate_patient_id()

    # Create layout columns for Patient Info
    info_col1, info_col2, info_col3 = st.columns([2, 1, 1])
    with info_col1:
        patient_name = st.text_input(
            "Patient Name (Optional)", placeholder="e.g. John Doe", key="patient_name"
        )
    with info_col2:
        st.text_input("Patient ID", value=st.session_state.patient_id, disabled=True)
    with info_col3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("New Patient", use_container_width=True):
            st.session_state.patient_id = generate_patient_id()
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Input Tabs ──
    tab_mic, tab_upload, tab_demo = st.tabs(
        ["🎙️ Record Audio", "📁 Upload File", "🧪 Demo Mode"]
    )

    # MODE: Microphone Recording
    with tab_mic:
        st.info("Record a short cough sample directly from your microphone.", icon="🎙️")
        audio_data = st.audio_input("Record your cough", key="mic_input")
        if audio_data is not None:
            _process_audio(
                audio_data.getvalue(),
                st.session_state.patient_id,
                patient_name,
                key_suffix="mic",
            )

    # MODE: File Upload
    with tab_upload:
        uploaded = st.file_uploader(
            "Upload an audio file (WAV, OGG, FLAC, MP3)",
            type=["wav", "ogg", "flac", "mp3", "webm"],
            key="file_upload",
        )
        if uploaded is not None:
            st.audio(uploaded, format="audio/wav")
            _process_audio(
                uploaded.getvalue(),
                st.session_state.patient_id,
                patient_name,
                key_suffix="upload",
            )

    # MODE: Demo Mode
    with tab_demo:
        st.info(
            "Demo mode generates a synthetic 30-day patient record and runs the full RATM pipeline.",
            icon="🧪",
        )

        demo_trajectory = st.selectbox(
            "Simulated Trajectory Class",
            options=[0, 1, 2, 3],
            format_func=lambda x: f"{TRAJECTORY_INFO[x][0]} - {TRAJECTORY_INFO[x][1]}",
            index=2,
        )

        if st.button("Generate Demo Insight", type="primary", use_container_width=True):
            with st.spinner("Running RATM pipeline on synthetic data..."):
                try:
                    pipeline = load_ratm_pipeline()
                    insight = pipeline.generate_demo_insight(
                        trajectory_class=demo_trajectory,
                        subject_id=st.session_state.patient_id,
                    )
                    _render_clinical_report(
                        dataclasses.asdict(insight), patient_name=patient_name
                    )
                except Exception as e:
                    st.error(f"Pipeline error: {e}")


def _process_audio(
    audio_bytes: bytes, patient_id: str, patient_name: str, key_suffix: str = "audio"
):
    """Process audio: detect cough, show results, generate insight."""

    with st.spinner("Processing audio through CNN cough detector..."):
        try:
            mel_spec, waveform, sr = audio_to_mel(audio_bytes)
            model, device = load_cough_detector()
            result = classify_audio(mel_spec, model, device, waveform=waveform, sr=sr)
        except Exception as e:
            st.error(f"Audio processing error: {e}")
            return

    # Phase B: Audio Detection Results
    st.divider()
    st.markdown("#### Acoustic Analysis")

    duration = len(waveform) / sr

    col_det1, col_det2, col_det3 = st.columns(3)
    with col_det1:
        if result["is_cough"]:
            st.success("Cough Detected", icon="✅")
        else:
            st.warning("No Cough Detected", icon="❌")
    with col_det2:
        st.metric("Detection Confidence", f"{result['probability']:.1%}")
    with col_det3:
        st.metric(
            "Audio Duration", f"{duration:.1f}s  ({result['num_segments']} segments)"
        )

    # Segment probability bars
    seg_probs = result.get("segment_probs", [])
    if len(seg_probs) > 1:
        with st.expander(
            f"Per-Segment Analysis ({len(seg_probs)} segments)", expanded=True
        ):
            for i, prob in enumerate(seg_probs):
                color = "var(--severity-high)" if prob > 0.5 else "var(--prism-accent)"
                label = "cough" if prob > 0.5 else "non-cough"
                st.markdown(
                    f"""
                    <div class="prism-prob-row">
                        <div class="prism-prob-label" style="flex: 0 0 70px;">Seg {i+1}</div>
                        <div class="prism-prob-bar-container">
                            <div class="prism-prob-bar"
                                 style="width: {prob*100}%; background-color: {color};"></div>
                        </div>
                        <div class="prism-prob-val">{prob:.0%}</div>
                        <div style="flex: 0 0 90px; text-align: right; font-size: 0.8rem;
                                    color: var(--prism-text-dim);">{label}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # Phase C: Clinical Insight Generation
    st.markdown("<br>", unsafe_allow_html=True)
    st.info(
        "Trajectory analysis requires 30 days of data. "
        "We'll simulate a 30-day context matching the acoustic profile detected.",
        icon="📅",
    )

    suggested_class = 2 if (result["is_cough"] and result["probability"] > 0.8) else 0

    traj_class = st.selectbox(
        "Simulate trajectory context",
        options=[0, 1, 2, 3],
        format_func=lambda x: f"{TRAJECTORY_INFO[x][0]} - {TRAJECTORY_INFO[x][1]}",
        index=suggested_class,
        key=f"traj_select_{key_suffix}",
    )

    if st.button(
        "Generate Clinical Report",
        type="primary",
        use_container_width=True,
        key=f"gen_report_{key_suffix}",
    ):
        with st.spinner("Running RATM pipeline..."):
            try:
                pipeline = load_ratm_pipeline()
                insight = pipeline.generate_demo_insight(
                    trajectory_class=traj_class,
                    subject_id=patient_id,
                )
                insight_dict = dataclasses.asdict(insight)

                # Predict disease probabilities using the cached classifier
                try:
                    from models.disease_classifier.classifier import DISEASE_CLASSES

                    classifier = load_disease_classifier()
                    emb_tensor = torch.tensor(
                        result["embedding"], dtype=torch.float32
                    ).unsqueeze(0)

                    with torch.no_grad():
                        logits = classifier(emb_tensor)
                        probs = torch.softmax(logits, dim=1).squeeze().tolist()

                    insight_dict["disease_probabilities"] = {
                        cls: round(p, 4)
                        for cls, p in zip(DISEASE_CLASSES, probs, strict=False)
                    }
                except Exception as e:
                    st.warning(f"Could not load disease classifier: {e}")

                _render_clinical_report(insight_dict, patient_name=patient_name)
            except Exception as e:
                st.error(f"Pipeline error: {e}")


def _render_clinical_report(insight: dict, patient_name: str = ""):
    """Render the formal PRISM Clinical Report Card using native Streamlit widgets."""

    st.divider()

    patient_display = (
        f"{patient_name} ({insight['patient_id']})"
        if patient_name
        else insight["patient_id"]
    )
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M UTC")

    traj_name = str(insight["trajectory_class"])
    severity = insight["overall_severity"]

    # ── Report Header ──
    st.markdown(
        f"""
        <div style="background-color: rgba(0, 180, 216, 0.05);
                    border: 1px solid var(--prism-border);
                    border-radius: 8px; padding: 1.2rem; margin-bottom: 1.5rem;">
            <h3 style="margin: 0 0 0.5rem 0; color: var(--prism-text);
                       font-weight: 700; letter-spacing: 0.5px;">
                PRISM CLINICAL ASSESSMENT REPORT
            </h3>
            <div style="display: flex; justify-content: space-between;
                        color: var(--prism-text-muted);
                        font-family: 'JetBrains Mono', monospace; font-size: 0.9rem;">
                <span>Patient: <strong style="color: var(--prism-text);">{patient_display}</strong></span>
                <span>Generated: {timestamp}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Key Metrics ──
    col_t, col_c, col_s = st.columns(3)
    with col_t:
        st.metric("Trajectory", traj_name.upper())
    with col_c:
        st.metric("Confidence", f"{insight['trajectory_confidence']:.0%}")
    with col_s:
        st.metric("Severity", severity.upper())

    # ── Clinical Summary ──
    st.markdown("#### Clinical Summary")
    st.markdown(
        f"""
        <div style="background-color: var(--prism-bg);
                    border-left: 4px solid var(--prism-accent);
                    padding: 1.2rem; border-radius: 4px; line-height: 1.6;">
            {insight["summary"]}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")

    # ── Observations ──
    observations = insight.get("observations", [])
    if observations:
        st.markdown("#### Observations")
        for i, obs in enumerate(observations):
            sev = obs["severity"]
            cat = obs["category"].replace("_", " ").title()
            expanded = i == 0 or sev in ("high", "moderate")
            with st.expander(f"{sev.upper()}  ·  {cat}", expanded=expanded):
                st.markdown(obs["text"])

    # ── Disease Probabilities ──
    probs = insight.get("disease_probabilities")
    if probs:
        st.markdown("#### Disease Probability Assessment")
        st.caption("Based on the acoustic signature of the 512-D CNN embedding.")

        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)

        for disease, prob in sorted_probs:
            color = "var(--prism-accent)"
            if prob > 0.5:
                color = "var(--severity-high)"
            elif prob > 0.3:
                color = "var(--severity-moderate)"

            st.markdown(
                f"""
                <div class="prism-prob-row">
                    <div class="prism-prob-label">{disease}</div>
                    <div class="prism-prob-bar-container">
                        <div class="prism-prob-bar"
                             style="width: {prob*100}%; background-color: {color};"></div>
                    </div>
                    <div class="prism-prob-val">{prob:.1%}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Download Report ──
    import json

    report_json = json.dumps(insight, indent=2)
    st.download_button(
        label="⬇ Download Report (JSON)",
        data=report_json,
        file_name=f"prism_report_{insight['patient_id']}.json",
        mime="application/json",
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
