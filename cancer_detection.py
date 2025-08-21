"""
⚠️ DISCLAIMER
This app is for EDUCATIONAL/DEMO purposes only. It is NOT a medical device and must not be used for diagnosis or treatment decisions.

How to run (from PyCharm terminal or any terminal in the project folder):
    python -m streamlit run app.py

If you prefer a Run Configuration in PyCharm, set Parameters to: -m streamlit run app.py

Optional extras:
- For image models, you can upload a PyTorch .pt/.pth file that matches the chosen architecture.
- SHAP explanations are optional (install with: pip install shap). The app will gracefully skip if SHAP is not available.

Tested with Python 3.10+ on Windows/macOS/Linux.
"""

import io
import os
import json
import time
import base64
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report, auc
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibrationDisplay

# Optional imports: SHAP & Torch (image models)
try:
    import shap  # type: ignore
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False

try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    from torchvision.models import resnet18, resnet34, resnet50
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

st.set_page_config(page_title="Cancer Detection Demo (Streamlit)", layout="wide")

# ================================
# Utility helpers
# ================================

def _download_button_bytes(content: bytes, filename: str, label: str):
    b64 = base64.b64encode(content).decode()
    href = f'<a download="{filename}" href="data:file/octet-stream;base64,{b64}">{label}</a>'
    st.markdown(href, unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def get_breast_cancer_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')  # 0 = malignant, 1 = benign (per sklearn dataset)
    return X, y

@st.cache_resource(show_spinner=False)
def build_pipeline(model_name: str, params: Dict[str, Any]):
    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=params.get("max_iter", 200), C=params.get("C", 1.0), solver="lbfgs")
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", model)
        ])
    elif model_name == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=params.get("n_estimators", 300),
            max_depth=params.get("max_depth", None),
            min_samples_split=params.get("min_samples_split", 2),
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        )
        pipe = Pipeline([ ("clf", model) ])
    else:
        raise ValueError("Unsupported model")
    return pipe

@st.cache_data(show_spinner=False)
def kfold_cv_predict(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, n_splits: int = 5):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", n_jobs=None)[:, 1]
    return y_proba


def plot_roc_pr(y_true: np.ndarray, y_proba: np.ndarray):
    col1, col2, col3 = st.columns(3)
    with col1:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        fig = plt.figure()
        plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc_score(y_true, y_proba):.3f}")
        plt.plot([0,1],[0,1],'--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        st.pyplot(fig, clear_figure=True)
    with col2:
        prec, rec, _ = precision_recall_curve(y_true, y_proba)
        ap = average_precision_score(y_true, y_proba)
        fig = plt.figure()
        plt.plot(rec, prec, label=f"AP = {ap:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend(loc="lower left")
        st.pyplot(fig, clear_figure=True)
    with col3:
        # Calibration curve using a simple train/test split to avoid data leakage in display
        X, y = get_breast_cancer_dataset()
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
        # Refit pipeline on train
        # Since we don't have access to the pipe here, build a reasonable default for display
        default_pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=200))])
        default_pipe.fit(X_tr, y_tr)
        prob_pos = default_pipe.predict_proba(X_te)[:, 1]
        fig = plt.figure()
        disp = CalibrationDisplay.from_predictions(y_te, prob_pos, n_bins=10)
        plt.title("Calibration Curve (example)")
        st.pyplot(fig, clear_figure=True)


def show_confusion_and_report(y_true: np.ndarray, y_pred: np.ndarray):
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(values_format='d')
    plt.title("Confusion Matrix")
    st.pyplot(fig, clear_figure=True)
    st.text("Classification Report:\n" + classification_report(y_true, y_pred, digits=3))


# ================================
# Image utilities (optional, Torch)
# ================================
class _Identity(nn.Module):
    def forward(self, x):
        return x

@st.cache_resource(show_spinner=False)
def get_torch_model(arch: str, num_classes: int = 2, ckpt_bytes: bytes | None = None):
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch/torchvision not installed. Install with: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
    if arch == "resnet18":
        model = resnet18(weights=None)
    elif arch == "resnet34":
        model = resnet34(weights=None)
    elif arch == "resnet50":
        model = resnet50(weights=None)
    else:
        raise ValueError("Unsupported architecture")
    # Replace final FC for binary classification
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)
    if ckpt_bytes is not None:
        buffer = io.BytesIO(ckpt_bytes)
        try:
            state = torch.load(buffer, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                model.load_state_dict({k.replace("model.", ""): v for k, v in state["state_dict"].items()}, strict=False)
            else:
                model.load_state_dict(state, strict=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load weights: {e}")
    model.eval()
    return model

IMG_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]) if _HAS_TORCH else None


def grad_cam_heatmap(model: nn.Module, img_tensor: "torch.Tensor", target_layer_name: str = "layer4"):
    # Basic Grad-CAM for ResNet target layer
    feats = []
    grads = []

    def fwd_hook(_m, _i, o):
        feats.append(o.detach())
    def bwd_hook(_m, _gi, go):
        grads.append(go[0].detach())

    target_layer = dict(model.named_modules()).get(target_layer_name)
    if target_layer is None:
        raise RuntimeError(f"Layer {target_layer_name} not found in model")

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    try:
        img_tensor = img_tensor.requires_grad_(True)
        out = model(img_tensor)
        # Take the logit for class 1 (assumed malignant) for explanation
        score = out[:, 1].sum()
        model.zero_grad()
        score.backward()
        A = feats[0]  # [B, C, H, W]
        G = grads[0]  # [B, C, H, W]
        weights = G.mean(dim=(2, 3), keepdim=True)
        cam = (weights * A).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = cam[0, 0].cpu().numpy()
        return cam
    finally:
        h1.remove()
        h2.remove()


def overlay_heatmap_on_image(np_img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    # Expect np_img in HxWx3 range [0,1], heatmap HxW in [0,1]
    h, w, _ = np_img.shape
    hm = (heatmap * 255).astype(np.uint8)
    # Use matplotlib to create a colored heatmap
    fig = plt.figure()
    plt.imshow(np_img)
    plt.imshow(hm, cmap='jet', alpha=alpha)
    plt.axis('off')
    fig.canvas.draw()
    overlay = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    overlay = overlay.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return overlay

# ================================
# Sidebar
# ================================
st.sidebar.title("⚙️ Controls")
st.sidebar.info("This app is for demo/education only. Not for clinical use.")
section = st.sidebar.radio(
    "Go to section",
    ["Overview", "Tabular (Breast Cancer)", "Image (Custom Model)", "Batch Inference"],
    index=0
)

# ================================
# Overview
# ================================
if section == "Overview":
    st.title("🧪 Cancer Detection — Demo & Educational App")
    st.markdown(
        """
        This Streamlit app showcases **two workflows**:

        1. **Tabular classification** using the built-in *Breast Cancer Wisconsin* dataset from scikit-learn (binary classification).
        2. **Image inference (optional)** using a **custom PyTorch CNN** you upload (e.g., a ResNet trained on your medical images). Includes a simple Grad-CAM visualization.

        **Key features**
        - Interactive model selection and hyperparameters
        - Cross-validated metrics (ROC AUC, PR AUC), confusion matrix, classification report
        - Probability calibration curve example
        - Export predictions as CSV
        - Optional SHAP explanations (if `shap` is installed)
        - Optional Grad-CAM for CNN image models (if `torch` is installed and a compatible model is uploaded)
        """
    )

    st.warning("This app is not a medical device and is not intended for diagnosis or clinical decision-making.")

# ================================
# Tabular (Breast Cancer)
# ================================
if section == "Tabular (Breast Cancer)":
    st.title("📊 Tabular Cancer Classification (Wisconsin Dataset)")
    X, y = get_breast_cancer_dataset()

    with st.expander("Peek at the data"):
        st.write("Shape:", X.shape)
        st.dataframe(pd.concat([X.head(20), y.head(20)], axis=1))

    st.subheader("Model & Hyperparameters")
    model_choice = st.selectbox("Choose model", ["Logistic Regression", "Random Forest"], index=1)

    with st.container():
        if model_choice == "Logistic Regression":
            C = st.slider("Inverse regularization (C)", 0.01, 5.0, 1.0, 0.01)
            max_iter = st.slider("Max iterations", 100, 1000, 300, 50)
            params = {"C": C, "max_iter": max_iter}
        else:
            n_estimators = st.slider("n_estimators", 50, 1000, 400, 50)
            max_depth = st.slider("max_depth (None=auto)", 0, 40, 0, 1)
            max_depth = None if max_depth == 0 else max_depth
            min_samples_split = st.slider("min_samples_split", 2, 10, 2, 1)
            params = {"n_estimators": n_estimators, "max_depth": max_depth, "min_samples_split": min_samples_split}

    n_splits = st.slider("Cross-validation folds", 3, 10, 5, 1)

    if st.button("Run CV & Evaluate", type="primary"):
        with st.spinner("Training and evaluating..."):
            pipe = build_pipeline(model_choice, params)
            y_proba = kfold_cv_predict(pipe, X, y, n_splits=n_splits)
            auc_roc = roc_auc_score(y, y_proba)
            ap = average_precision_score(y, y_proba)
            st.success(f"ROC AUC: {auc_roc:.4f} | Average Precision (PR AUC): {ap:.4f}")

            # Choose threshold
            thr = st.slider("Decision threshold (default 0.5)", 0.0, 1.0, 0.5, 0.01)
            y_pred = (y_proba >= thr).astype(int)

            plot_roc_pr(y.values, y_proba)
            show_confusion_and_report(y.values, y_pred)

            # Export predictions
            out = pd.DataFrame({"y_true": y, "y_proba": y_proba, "y_pred": y_pred})
            st.download_button(
                label="Download predictions CSV",
                data=out.to_csv(index=False).encode(),
                file_name="breast_cancer_predictions.csv",
                mime="text/csv",
            )

    st.subheader("Explainability (optional)")
    if _HAS_SHAP:
        st.caption("Using KernelExplainer/LR or TreeExplainer/RF depending on model.")
        if st.button("Compute SHAP on a sample"):
            pipe = build_pipeline(model_choice, params)
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
            pipe.fit(X_tr, y_tr)
            model = pipe.named_steps.get("clf")

            if model_choice == "Random Forest":
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_te)
                fig = plt.figure()
                shap.summary_plot(shap_values[1], X_te, show=False)
                st.pyplot(fig, clear_figure=True)
            else:
                # KernelExplainer (can be slow) — take a small sample
                back = shap.sample(X_tr, 100, random_state=42)
                explainer = shap.KernelExplainer(lambda data: pipe.predict_proba(data)[:,1], back)
                sample = shap.sample(X_te, 100, random_state=7)
                shap_values = explainer.shap_values(sample, nsamples=100)
                fig = plt.figure()
                shap.summary_plot(shap_values, sample, show=False)
                st.pyplot(fig, clear_figure=True)
    else:
        st.info("Install SHAP to enable explainability: pip install shap")

# ================================
# Image (Custom Model)
# ================================
if section == "Image (Custom Model)":
    st.title("🖼️ Image Inference with Custom CNN (Optional)")
    st.warning("This section expects you to upload a compatible PyTorch model. Without it, predictions are placeholders.")

    arch = st.selectbox("Architecture", ["resnet18", "resnet34", "resnet50"], index=0)
    uploaded_weights = st.file_uploader("Upload PyTorch weights (.pt/.pth) matching the chosen arch (optional)", type=["pt", "pth"], accept_multiple_files=False)

    model = None
    if _HAS_TORCH:
        try:
            weights_bytes = uploaded_weights.read() if uploaded_weights else None
            model = get_torch_model(arch, num_classes=2, ckpt_bytes=weights_bytes)
            st.success("Model ready.")
        except Exception as e:
            st.error(f"Model load error: {e}")
    else:
        st.info("PyTorch not installed. Install with: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")

    img_files = st.file_uploader("Upload image(s)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    if img_files and model is not None and _HAS_TORCH:
        for img_file in img_files:
            st.markdown("---")
            st.write(f"**{img_file.name}**")
            from PIL import Image
            pil = Image.open(img_file).convert("RGB")
            st.image(pil, caption="Input", use_column_width=True)

            with torch.no_grad():
                t = IMG_TRANSFORM(pil).unsqueeze(0)
                logits = model(t)
                prob = torch.softmax(logits, dim=1)[0].cpu().numpy()
                benign_p = float(prob[0])
                malignant_p = float(prob[1])
                st.write({"benign": round(benign_p, 4), "malignant": round(malignant_p, 4)})

            # Grad-CAM
            try:
                heat = grad_cam_heatmap(model, t, target_layer_name="layer4")
                # Convert PIL to numpy in [0,1]
                arr = np.asarray(pil).astype(np.float32)/255.0
                overlay = overlay_heatmap_on_image(arr, heat, alpha=0.45)
                st.image(overlay, caption="Grad-CAM overlay", use_column_width=True)
            except Exception as e:
                st.info(f"Grad-CAM not available: {e}")

    elif img_files and (model is None or not _HAS_TORCH):
        st.info("Upload a compatible model and ensure PyTorch is installed to run inference.")

    st.caption("Note: Class labels are assumed as index 0 = benign, 1 = malignant. Adjust your training accordingly.")

# ================================
# Batch Inference
# ================================
if section == "Batch Inference":
    st.title("📦 Batch Inference (Tabular)")
    st.write("Upload a CSV with the **same columns** as the Wisconsin dataset. We'll run the selected model and return predictions.")

    X_ref, y_ref = get_breast_cancer_dataset()
    example_csv = pd.concat([X_ref.head(20)], axis=1)
    st.download_button(
        label="Download example CSV (features only)",
        data=example_csv.to_csv(index=False).encode(),
        file_name="example_features.csv",
        mime="text/csv",
    )

    up = st.file_uploader("Upload CSV", type=["csv"])

    model_choice = st.selectbox("Model", ["Logistic Regression", "Random Forest"], index=1, key="batch_model")
    if model_choice == "Logistic Regression":
        params = {"C": 1.0, "max_iter": 300}
    else:
        params = {"n_estimators": 400, "max_depth": None, "min_samples_split": 2}

    if up is not None:
        df = pd.read_csv(up)
        missing = set(X_ref.columns) - set(df.columns)
        if missing:
            st.error(f"Missing columns: {sorted(missing)[:10]} ...")
        else:
            if st.button("Run batch predictions", type="primary"):
                pipe = build_pipeline(model_choice, params)
                # Fit on full reference dataset for best use of data (demo purpose)
                pipe.fit(X_ref, y_ref)
                proba = pipe.predict_proba(df)[:, 1]
                pred = (proba >= 0.5).astype(int)
                out = df.copy()
                out["pred_proba_malignant"] = proba
                out["pred_label"] = pred
                st.success("Predictions ready.")
                st.dataframe(out.head(50))
                st.download_button(
                    label="Download predictions CSV",
                    data=out.to_csv(index=False).encode(),
                    file_name="batch_predictions.csv",
                    mime="text/csv",
                )

# ================================
# Footer
# ================================
st.markdown("""
---
**Notes**
- Dataset license: Scikit-learn toy dataset (Breast Cancer Wisconsin) for demonstration.
- For real projects: use institution-approved datasets, rigorous validation, and comply with clinical regulations.
""")
