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
import io

import numpy as np
import streamlit as st
import torch

st.set_page_config(
    page_title="PRISM - Clinical Insights",
    page_icon="🫁",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────

SEVERITY_COLORS = {
    "info": "#3498db",
    "low": "#2ecc71",
    "moderate": "#f39c12",
    "high": "#e74c3c",
}

TRAJECTORY_COLORS = {
    "Stable": "#2ecc71",
    "Improving": "#3498db",
    "Increasing": "#f39c12",
    "Abnormal": "#e74c3c",
}

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
    from pathlib import Path

    from models.cough_detector.model import build_model
    from models.shared.checkpoint import load_checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(pretrained=False, device=device)

    # Try finetuned first, fall back to original
    checkpoints = [
        "models/checkpoints/cough_detector_finetuned.pt",
        "models/checkpoints/cough_detector_best.pt",
    ]
    loaded = False
    for ckpt in checkpoints:
        if Path(ckpt).exists():
            try:
                load_checkpoint(ckpt, model, device=device)
                st.sidebar.caption(f"CNN: {Path(ckpt).stem}")
                loaded = True
                break
            except Exception:
                continue

    if not loaded:
        st.sidebar.warning("CNN: no checkpoint found (random weights)")

    model.eval()
    return model, device


@st.cache_resource
def load_ratm_pipeline():
    """Load the RATM pipeline."""
    from retrieval.ratm_pipeline import RATMPipeline

    return RATMPipeline(use_retrieval=False)


def audio_to_mel(audio_bytes: bytes, sample_rate: int = 16000) -> tuple:
    """
    Convert audio bytes to a log-mel spectrogram tensor.

    Applies the same preprocessing as the training pipeline:
    - Resample to 16 kHz mono
    - Peak-normalize waveform amplitude
    - Compute log-mel spectrogram matching AudioConfig defaults

    Returns:
        (log_mel, waveform, sample_rate)
    """
    import librosa
    import soundfile as sf

    # Read audio from bytes
    with io.BytesIO(audio_bytes) as buf:
        waveform, sr = sf.read(buf, dtype="float32")

    # Convert to mono if stereo
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    # Resample if needed
    if sr != sample_rate:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=sample_rate)

    # Peak-normalize so amplitude matches training data distribution
    peak = np.abs(waveform).max()
    if peak > 1e-6:
        waveform = waveform / peak

    # Compute mel spectrogram (matching AudioConfig defaults)
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
    """
    Split a waveform into fixed-length overlapping segments,
    matching the training pipeline's segmentation strategy.

    Short recordings (< segment_sec) are zero-padded.
    """
    seg_len = int(segment_sec * sample_rate)  # 48000 samples at 16 kHz
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

    # Pad the tail
    if start < total:
        tail = np.zeros(seg_len, dtype=waveform.dtype)
        tail[: total - start] = waveform[start:]
        segments.append(tail)

    return segments


def _mel_from_segment(segment: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """Compute log-mel spectrogram from a single 3-second segment."""
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
    """
    Run the CNN cough detector on audio.

    If a waveform is provided, segments it into 3-second windows (matching
    training) and takes the segment with the highest cough probability.
    Otherwise falls back to the full mel spectrogram.

    Returns dict with: is_cough, probability, embedding (512-dim),
                       num_segments, segment_probs.
    """
    if waveform is not None:
        # Segment the audio into 3-second chunks (matching training pipeline)
        segments = _segment_waveform(waveform, sample_rate=sr)
        best_prob = -1.0
        best_embedding = None
        segment_probs = []

        for seg in segments:
            seg_mel = _mel_from_segment(seg, sr)
            mel_tensor = (
                torch.from_numpy(seg_mel).float().unsqueeze(0).unsqueeze(0).to(device)
            )  # (1, 1, 128, T)

            # CRITICAL FIX: Normalize exactly like training (Z-score)
            mean = mel_tensor.mean()
            std = mel_tensor.std().clamp(min=1e-6)
            mel_tensor = (mel_tensor - mean) / std

            with torch.no_grad():
                logits, embedding = model(mel_tensor)
                prob = torch.sigmoid(logits).item()

            segment_probs.append(round(prob, 4))
            if prob > best_prob:
                best_prob = prob
                best_embedding = embedding.cpu().numpy()[0]

        return {
            "is_cough": best_prob > 0.5,
            "probability": round(best_prob, 4),
            "embedding": best_embedding,
            "num_segments": len(segments),
            "segment_probs": segment_probs,
        }
    else:
        # Fallback: full spectrogram (variable length)
        mel_tensor = (
            torch.from_numpy(mel_spec).float().unsqueeze(0).unsqueeze(0).to(device)
        )

        # Normalize
        mean = mel_tensor.mean()
        std = mel_tensor.std().clamp(min=1e-6)
        mel_tensor = (mel_tensor - mean) / std

        with torch.no_grad():
            logits, embedding = model(mel_tensor)
            prob = torch.sigmoid(logits).item()

        return {
            "is_cough": prob > 0.5,
            "probability": round(prob, 4),
            "embedding": embedding.cpu().numpy()[0],
            "num_segments": 1,
            "segment_probs": [round(prob, 4)],
        }


def render_severity_badge(severity: str) -> str:
    """Render a colored severity badge."""
    color = SEVERITY_COLORS.get(severity, "#999")
    return (
        f'<span style="background:{color};color:white;padding:4px 12px;'
        f"border-radius:14px;font-size:0.85em;font-weight:600;"
        f'letter-spacing:0.5px;">{severity.upper()}</span>'
    )


def render_trajectory_badge(traj_class: str) -> str:
    """Render a trajectory class badge."""
    color = TRAJECTORY_COLORS.get(traj_class, "#999")
    return (
        f'<span style="background:{color};color:white;padding:4px 14px;'
        f"border-radius:14px;font-size:1.1em;font-weight:700;"
        f'letter-spacing:0.5px;">{traj_class}</span>'
    )


# ──────────────────────────────────────────────────────────────────
# Page layout
# ──────────────────────────────────────────────────────────────────


def main():
    st.title("PRISM Clinical Insights")
    st.markdown(
        "Record or upload a cough audio sample, detect coughs, "
        "predict respiratory trajectory, and generate clinical insights."
    )
    st.divider()

    # ── Sidebar ──
    with st.sidebar:
        st.header("Analysis Mode")

        mode = st.radio(
            "Select input mode",
            ["Microphone Recording", "Audio File Upload", "Demo Mode"],
            index=2,
            help="Choose how to provide audio input for analysis.",
        )

        st.divider()

        if mode == "Demo Mode":
            st.subheader("Demo Settings")
            demo_trajectory = st.selectbox(
                "Trajectory Class",
                options=[0, 1, 2, 3],
                format_func=lambda x: f"{TRAJECTORY_INFO[x][0]} - {TRAJECTORY_INFO[x][1]}",
                index=2,
            )
            demo_id = st.text_input("Patient ID", value="demo_patient_001")

        st.divider()
        st.caption(
            "PRISM analyzes cough audio using a CNN detector, "
            "predicts trajectory trends with a Temporal Transformer, "
            "and generates clinical insights via the RATM pipeline."
        )

    # ──────────────────────────────────────────────────────────────
    # MODE: Microphone Recording
    # ──────────────────────────────────────────────────────────────

    if mode == "Microphone Recording":
        st.subheader("Step 1: Record Cough Audio")
        st.info(
            "Click the microphone button below to record a cough sample. "
            "Allow microphone access when prompted by your browser.",
            icon="🎙️",
        )

        audio_data = st.audio_input("Record your cough", key="mic_input")

        if audio_data is not None:
            _process_audio(audio_data.getvalue())

    # ──────────────────────────────────────────────────────────────
    # MODE: File Upload
    # ──────────────────────────────────────────────────────────────

    elif mode == "Audio File Upload":
        st.subheader("Step 1: Upload Cough Audio")
        uploaded = st.file_uploader(
            "Upload an audio file (WAV, OGG, FLAC, MP3)",
            type=["wav", "ogg", "flac", "mp3", "webm"],
            key="file_upload",
        )

        if uploaded is not None:
            st.audio(uploaded, format="audio/wav")
            _process_audio(uploaded.getvalue())

    # ──────────────────────────────────────────────────────────────
    # MODE: Demo
    # ──────────────────────────────────────────────────────────────

    elif mode == "Demo Mode":
        st.subheader("Demo: Synthetic Patient Analysis")
        st.info(
            "Demo mode generates a synthetic 30-day patient record and "
            "runs the full RATM pipeline. No audio input required.",
            icon="🧪",
        )

        if st.button("Generate Demo Insight", type="primary", use_container_width=True):
            with st.spinner("Running RATM pipeline on synthetic data..."):
                try:
                    pipeline = load_ratm_pipeline()
                    insight = pipeline.generate_demo_insight(
                        trajectory_class=demo_trajectory,
                        subject_id=demo_id,
                    )
                    _display_insight(
                        dataclasses.asdict(insight), show_audio_section=False
                    )
                except Exception as e:
                    st.error(f"Pipeline error: {e}")


def _process_audio(audio_bytes: bytes):
    """Process audio: detect cough, show results, generate insight."""

    # Step 2: Cough Detection
    st.divider()
    st.subheader("Step 2: Cough Detection (CNN)")

    with st.spinner("Processing audio through CNN cough detector..."):
        try:
            mel_spec, waveform, sr = audio_to_mel(audio_bytes)
            model, device = load_cough_detector()
            result = classify_audio(mel_spec, model, device, waveform=waveform, sr=sr)
        except Exception as e:
            st.error(f"Audio processing error: {e}")
            return

    # Display detection results
    col_det1, col_det2, col_det3 = st.columns(3)

    with col_det1:
        if result["is_cough"]:
            st.success("Cough Detected", icon="✅")
        else:
            st.warning("No Cough Detected", icon="❌")

    with col_det2:
        st.metric("Detection Confidence", f"{result['probability']:.1%}")

    with col_det3:
        duration = len(waveform) / sr
        st.metric("Audio Duration", f"{duration:.1f}s")

    # Per-segment breakdown
    seg_probs = result.get("segment_probs", [])
    if len(seg_probs) > 1:
        with st.expander(
            f"Per-Segment Analysis ({result['num_segments']} segments × 3s)",
            expanded=True,
        ):
            import matplotlib.pyplot as plt

            fig_seg, ax_seg = plt.subplots(figsize=(10, 2.5))
            fig_seg.patch.set_facecolor("#0e1117")
            ax_seg.set_facecolor("#0e1117")

            colors = ["#e74c3c" if p > 0.5 else "#3498db" for p in seg_probs]
            ax_seg.bar(
                range(len(seg_probs)), seg_probs, color=colors, width=0.7, alpha=0.85
            )
            ax_seg.axhline(
                y=0.5,
                color="#f39c12",
                linestyle="--",
                linewidth=1.2,
                alpha=0.8,
                label="Threshold (0.5)",
            )
            ax_seg.set_xlabel("Segment", color="white", fontsize=10)
            ax_seg.set_ylabel("Cough Probability", color="white", fontsize=10)
            ax_seg.set_title(
                "Cough Probability per 3-Second Segment", color="white", fontsize=11
            )
            ax_seg.set_ylim(0, 1.05)
            ax_seg.tick_params(colors="white", labelsize=9)
            ax_seg.legend(loc="upper right", fontsize=9)
            for spine in ax_seg.spines.values():
                spine.set_color("#444")
            st.pyplot(fig_seg)
            plt.close(fig_seg)

            # Text summary
            cough_segs = sum(1 for p in seg_probs if p > 0.5)
            st.markdown(
                f"**{cough_segs}** of **{len(seg_probs)}** segments classified as cough "
                f"(max prob: **{max(seg_probs):.1%}**, avg: **{np.mean(seg_probs):.1%}**)"
            )

    # Show mel spectrogram
    with st.expander("Mel Spectrogram Visualization", expanded=False):
        import librosa.display
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 3))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")
        librosa.display.specshow(
            mel_spec,
            sr=sr,
            hop_length=512,
            x_axis="time",
            y_axis="mel",
            ax=ax,
            cmap="magma",
        )
        ax.set_title("Log-Mel Spectrogram", color="white", fontsize=12)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        st.pyplot(fig)
        plt.close(fig)

    # Show embedding info
    with st.expander("CNN Embedding (512-dim)", expanded=False):
        emb = result["embedding"]
        st.markdown(
            f"**Shape:** `{emb.shape}` | **L2 Norm:** `{np.linalg.norm(emb):.4f}`"
        )
        st.markdown(
            f"**Min:** `{emb.min():.4f}` | **Max:** `{emb.max():.4f}` | **Mean:** `{emb.mean():.4f}`"
        )

        # Mini histogram
        import matplotlib.pyplot as plt

        fig2, ax2 = plt.subplots(figsize=(8, 2))
        fig2.patch.set_facecolor("#0e1117")
        ax2.set_facecolor("#0e1117")
        ax2.bar(range(len(emb)), emb, color="#3498db", width=1.0, alpha=0.7)
        ax2.set_title("Embedding Vector Components", color="white", fontsize=10)
        ax2.tick_params(colors="white", labelsize=8)
        for spine in ax2.spines.values():
            spine.set_color("#333")
        st.pyplot(fig2)
        plt.close(fig2)

    # Step 3: Trajectory Prediction + Clinical Insight
    st.divider()
    st.subheader("Step 3: Trajectory & Clinical Insight")
    st.info(
        "Since real-time trajectory analysis requires 30 days of data, "
        "we generate a synthetic 30-day context that matches the cough "
        "characteristics detected in your audio sample.",
        icon="📅",
    )

    # Determine which trajectory class to simulate based on detection
    if result["is_cough"] and result["probability"] > 0.8:
        suggested_class = 2  # Increasing (more concerning)
    elif result["is_cough"]:
        suggested_class = 0  # Stable
    else:
        suggested_class = 0  # Stable for non-cough

    traj_class = st.selectbox(
        "Simulate trajectory context",
        options=[0, 1, 2, 3],
        format_func=lambda x: f"{TRAJECTORY_INFO[x][0]} - {TRAJECTORY_INFO[x][1]}",
        index=suggested_class,
        key="traj_select_audio",
    )

    if st.button("Generate Clinical Insight", type="primary", use_container_width=True):
        with st.spinner("Running RATM pipeline..."):
            try:
                pipeline = load_ratm_pipeline()
                insight = pipeline.generate_demo_insight(
                    trajectory_class=traj_class,
                    subject_id="audio_patient_001",
                )
                _display_insight(
                    dataclasses.asdict(insight),
                    show_audio_section=True,
                    cough_result=result,
                )
            except Exception as e:
                st.error(f"Pipeline error: {e}")


def _display_insight(
    insight: dict,
    show_audio_section: bool = False,
    cough_result: dict | None = None,
):
    """Display a clinical insight with all observations and metadata."""

    st.divider()

    # ── Header Metrics ──
    st.subheader("Clinical Assessment")

    col1, col2, col3, col4 = st.columns(4)

    traj_class = insight["trajectory_class"]

    with col1:
        st.metric("Patient", insight["patient_id"])
    with col2:
        st.markdown(
            f"**Trajectory**<br>{render_trajectory_badge(traj_class)}",
            unsafe_allow_html=True,
        )
    with col3:
        st.metric("Confidence", f"{insight['trajectory_confidence']:.0%}")
    with col4:
        st.markdown(
            f"**Severity**<br>{render_severity_badge(insight['overall_severity'])}",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Summary Narrative ──
    traj_color = TRAJECTORY_COLORS.get(traj_class, "#999")
    st.markdown("### Clinical Summary")
    st.markdown(
        f'<div style="background:linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);'
        f"border-left:4px solid {traj_color};padding:20px;border-radius:8px;"
        f'line-height:1.7;font-size:1.05em;box-shadow:0 2px 8px rgba(0,0,0,0.3);">'
        f'{insight["summary"]}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("")  # spacing

    # ── Observations ──
    observations = insight.get("observations", [])
    if observations:
        st.markdown("### Detailed Observations")
        for i, obs in enumerate(observations):
            sev = obs["severity"]
            sev_color = SEVERITY_COLORS.get(sev, "#999")
            cat = obs["category"].replace("_", " ").title()

            with st.expander(
                f"{cat}  |  {sev.upper()}",
                expanded=(i == 0 or sev in ("high", "moderate")),
            ):
                st.markdown(
                    f'<div style="border-left:3px solid {sev_color};'
                    f"padding:10px 15px;background:rgba(0,0,0,0.15);"
                    f'border-radius:0 6px 6px 0;">'
                    f'{obs["text"]}</div>',
                    unsafe_allow_html=True,
                )

    # ── Audio Detection Summary (if from audio input) ──
    if show_audio_section and cough_result:
        st.markdown("")
        st.markdown("### Audio Analysis Summary")
        col_a, col_b = st.columns(2)
        with col_a:
            status = "Cough Detected" if cough_result["is_cough"] else "No Cough"
            st.metric("Detection Result", status)
        with col_b:
            st.metric("CNN Confidence", f"{cough_result['probability']:.1%}")

    # ── Generation Metadata ──
    with st.expander("Generation Metadata", expanded=False):
        meta_col1, meta_col2 = st.columns(2)
        with meta_col1:
            st.markdown(f"**Generated at:** `{insight.get('generated_at', 'N/A')}`")
            st.markdown(f"**Method:** `{insight.get('generation_method', 'N/A')}`")
        with meta_col2:
            st.markdown(f"**Similar cases:** `{insight.get('similar_cases_count', 0)}`")
            st.markdown(f"**Alerts detected:** `{insight.get('alerts_count', 0)}`")
        st.markdown("**Templates used:**")
        templates = insight.get("templates_used", [])
        if templates:
            st.code(", ".join(templates))

    # ── Raw JSON ──
    with st.expander("Raw Insight JSON", expanded=False):
        st.json(insight)


if __name__ == "__main__":
    main()
