import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, auc,
)

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Telecom Churn Predictor",
    page_icon="📡",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────
# Constants — must match the notebook exactly
# ─────────────────────────────────────────────────────────────
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
NUMERICAL_FEATURES = [
    "tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen",
    "NewCustomer", "Tenure_MonthlyCharges_Interaction",
]

MODEL_PATH    = "rf_churn_model.pkl"
SCALER_PATH   = "scaler.pkl"
ENCODERS_PATH = "encoders.pkl"
COLS_PATH     = "feature_columns.pkl"


# ─────────────────────────────────────────────────────────────
# Preprocessing helper (mirrors the notebook)
# ─────────────────────────────────────────────────────────────
def preprocess_raw(df: pd.DataFrame):
    """Full pipeline used during training. Returns X, y, scaler, encoders, columns."""
    df = df.copy()

    # Fix TotalCharges dtype
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])

    # Encode target
    df["Churn"] = df["Churn"].replace({"Yes": 1, "No": 0})
    y = df["Churn"].astype(int)

    X = df.drop(columns=["Churn", "customerID"], errors="ignore")

    # Feature engineering
    X["NewCustomer"] = (X["tenure"] <= 2).astype(int)
    X["Tenure_MonthlyCharges_Interaction"] = X["tenure"] * X["MonthlyCharges"]

    # Label encode categoricals
    label_encoders = {}
    for col in CATEGORICAL_FEATURES:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le

    # Scale numericals
    scaler = StandardScaler()
    X[NUMERICAL_FEATURES] = scaler.fit_transform(X[NUMERICAL_FEATURES])

    return X, y, scaler, label_encoders, X.columns.tolist()


def preprocess_single(row: dict, scaler, label_encoders, feature_columns):
    """Preprocess a single input dict for inference."""
    df = pd.DataFrame([row])

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])

    # Feature engineering
    df["NewCustomer"] = (df["tenure"] <= 2).astype(int)
    df["Tenure_MonthlyCharges_Interaction"] = df["tenure"] * df["MonthlyCharges"]

    # Encode categoricals
    for col in CATEGORICAL_FEATURES:
        if col in df.columns and col in label_encoders:
            le = label_encoders[col]
            val = str(df[col].iloc[0])
            if val not in le.classes_:
                val = le.classes_[0]
            df[col] = le.transform([val])

    # Scale numericals
    df[NUMERICAL_FEATURES] = scaler.transform(df[NUMERICAL_FEATURES])

    # Align columns
    for c in feature_columns:
        if c not in df.columns:
            df[c] = 0
    df = df[feature_columns]

    return df


# ─────────────────────────────────────────────────────────────
# Load saved artifacts
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    if all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, ENCODERS_PATH, COLS_PATH]):
        model    = joblib.load(MODEL_PATH)
        scaler   = joblib.load(SCALER_PATH)
        encoders = joblib.load(ENCODERS_PATH)
        cols     = joblib.load(COLS_PATH)
        return model, scaler, encoders, cols
    return None, None, None, None


# ─────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center; color:#0F52BA;'>📡 Telecom Customer Churn Prediction</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center; color:gray;'>Group 7 — Binary Classification on Imbalanced Telecom Data</p>",
    unsafe_allow_html=True,
)
st.divider()

# ─────────────────────────────────────────────────────────────
# Sidebar — Train model from CSV
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Model Training")
    st.markdown("Upload `Telecom_churn.csv` to train and save the model.")
    uploaded_csv = st.file_uploader("Upload CSV", type="csv", key="train_csv")

    if uploaded_csv:
        with st.spinner("Training Random Forest…"):
            raw = pd.read_csv(uploaded_csv)
            X, y, scaler, label_encoders, feature_cols = preprocess_raw(raw)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            rf = RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
            rf.fit(X_train, y_train)

            y_pred  = rf.predict(X_test)
            y_proba = rf.predict_proba(X_test)[:, 1]
            roc     = roc_auc_score(y_test, y_proba)

            joblib.dump(rf,             MODEL_PATH)
            joblib.dump(scaler,         SCALER_PATH)
            joblib.dump(label_encoders, ENCODERS_PATH)
            joblib.dump(feature_cols,   COLS_PATH)

            # Clear cache so new model is loaded
            load_artifacts.clear()

        st.success(f"✅ Model trained!  ROC AUC: **{roc:.4f}**")

        with st.expander("Classification Report"):
            report = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose().round(2))

        with st.expander("ROC Curve"):
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc:.2f}")
            ax.plot([0, 1], [0, 1], "k--")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

    st.divider()
    st.markdown("**Artifact status**")
    for label, path in [
        ("Model",    MODEL_PATH),
        ("Scaler",   SCALER_PATH),
        ("Encoders", ENCODERS_PATH),
        ("Columns",  COLS_PATH),
    ]:
        icon = "✅" if os.path.exists(path) else "❌"
        st.markdown(f"{icon} `{label}`")


# ─────────────────────────────────────────────────────────────
# Load artifacts
# ─────────────────────────────────────────────────────────────
model, scaler, label_encoders, feature_cols = load_artifacts()

# ─────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Single Prediction", "📂 Batch Prediction", "📊 Feature Importance"])

# ══════════════════════════════════════════════
# TAB 1 — Single Prediction
# ══════════════════════════════════════════════
with tab1:
    st.subheader("Predict Churn for One Customer")

    if model is None:
        st.info("Train the model first by uploading the CSV in the sidebar.")
    else:
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Demographics**")
            gender         = st.selectbox("Gender", ["Male", "Female"])
            senior         = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
            partner        = st.selectbox("Partner", ["Yes", "No"])
            dependents     = st.selectbox("Dependents", ["Yes", "No"])
            tenure         = st.slider("Tenure (months)", 0, 72, 12)

        with c2:
            st.markdown("**Services**")
            phone_service  = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
            internet       = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_sec     = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            online_bkp     = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            device_prot    = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            tech_support   = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            streaming_tv   = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            streaming_mov  = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

        with c3:
            st.markdown("**Billing**")
            contract       = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless      = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment        = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)",
            ])
            monthly_charge = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, step=0.5)
            total_charges  = st.number_input("Total Charges ($)", 0.0, 10000.0, 500.0, step=10.0)

        if st.button("🚀 Predict", use_container_width=True, type="primary"):
            row = {
                "gender": gender, "SeniorCitizen": senior,
                "Partner": partner, "Dependents": dependents,
                "tenure": tenure, "PhoneService": phone_service,
                "MultipleLines": multiple_lines, "InternetService": internet,
                "OnlineSecurity": online_sec, "OnlineBackup": online_bkp,
                "DeviceProtection": device_prot, "TechSupport": tech_support,
                "StreamingTV": streaming_tv, "StreamingMovies": streaming_mov,
                "Contract": contract, "PaperlessBilling": paperless,
                "PaymentMethod": payment, "MonthlyCharges": monthly_charge,
                "TotalCharges": total_charges,
            }

            X_in   = preprocess_single(row, scaler, label_encoders, feature_cols)
            pred   = model.predict(X_in)[0]
            proba  = model.predict_proba(X_in)[0]

            st.divider()
            res_col1, res_col2 = st.columns(2)

            with res_col1:
                if pred == 1:
                    st.error("⚠️ **This customer is likely to CHURN**")
                else:
                    st.success("✅ **This customer is likely to STAY**")

                st.metric("Churn Probability",  f"{proba[1]:.1%}")
                st.metric("Retention Probability", f"{proba[0]:.1%}")

            with res_col2:
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.bar(["Stay", "Churn"], proba, color=["#2ecc71", "#e74c3c"])
                ax.set_ylim(0, 1)
                ax.set_ylabel("Probability")
                ax.set_title("Prediction Confidence")
                for i, v in enumerate(proba):
                    ax.text(i, v + 0.02, f"{v:.1%}", ha="center", fontweight="bold")
                st.pyplot(fig)
                plt.close(fig)

            # Risk level
            st.divider()
            churn_prob = proba[1]
            if churn_prob >= 0.7:
                st.error(f"🔴 **High Risk** ({churn_prob:.1%}) — Immediate retention action recommended.")
            elif churn_prob >= 0.4:
                st.warning(f"🟡 **Medium Risk** ({churn_prob:.1%}) — Monitor and offer incentives.")
            else:
                st.success(f"🟢 **Low Risk** ({churn_prob:.1%}) — Customer appears loyal.")


# ══════════════════════════════════════════════
# TAB 2 — Batch Prediction
# ══════════════════════════════════════════════
with tab2:
    st.subheader("Batch Prediction from CSV")
    st.markdown("Upload a CSV with the same columns as `Telecom_churn.csv` (without the `Churn` column).")

    batch_file = st.file_uploader("Upload batch CSV", type="csv", key="batch")

    if batch_file:
        if model is None:
            st.warning("Train the model first.")
        else:
            batch_df = pd.read_csv(batch_file)
            results  = []

            for _, row in batch_df.iterrows():
                row_dict = row.to_dict()
                try:
                    X_in  = preprocess_single(row_dict, scaler, label_encoders, feature_cols)
                    pred  = model.predict(X_in)[0]
                    prob  = model.predict_proba(X_in)[0][1]
                    results.append({"Prediction": "Churn" if pred == 1 else "Stay",
                                    "Churn Probability": f"{prob:.1%}"})
                except Exception:
                    results.append({"Prediction": "Error", "Churn Probability": "N/A"})

            out_df = batch_df.copy()
            out_df["Predicted Churn"] = [r["Prediction"] for r in results]
            out_df["Churn Probability"] = [r["Churn Probability"] for r in results]

            st.dataframe(out_df.head(100), use_container_width=True)

            # Summary
            total   = len(out_df)
            churned = (out_df["Predicted Churn"] == "Churn").sum()
            st.info(f"**{churned}** out of **{total}** customers predicted to churn ({churned/total:.1%})")

            csv_bytes = out_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Predictions CSV",
                csv_bytes, "churn_predictions.csv", "text/csv",
                use_container_width=True,
            )


# ══════════════════════════════════════════════
# TAB 3 — Feature Importance
# ══════════════════════════════════════════════
with tab3:
    st.subheader("Random Forest Feature Importance")

    if model is None:
        st.info("Train the model first to see feature importance.")
    else:
        importances = model.feature_importances_
        feat_df = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": importances,
        }).sort_values("Importance", ascending=False).head(15)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x="Importance", y="Feature", data=feat_df, palette="viridis", ax=ax)
        ax.set_title("Top 15 Feature Importances (Random Forest)", fontsize=14)
        ax.set_xlabel("Importance Score")
        ax.set_ylabel("")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.dataframe(feat_df.reset_index(drop=True), use_container_width=True)
