import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# --- Load Dataset ---
digits = load_digits()
X, y = digits.data, digits.target

st.title("🔢 Multiclass Classification on Digits Dataset")
st.write("64 features from 8x8 grayscale digit images (0–9)")

st.markdown("### 📊 Dataset Info")
st.write(f"Features shape: {X.shape}")
st.write(f"Target classes: {np.unique(y)}")

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Sidebar Options ---
st.sidebar.title("⚙️ Training Options")
epochs = st.sidebar.slider("Epochs", 1, 50, 10)
learning_rate = st.sidebar.selectbox("Learning Rate", [0.01, 0.001, 0.0001], index=1)

# --- Train with both optimizers ---
def train_model(optimizer):
    clf = SGDClassifier(loss="log_loss", learning_rate='constant', eta0=learning_rate,
                        max_iter=1, tol=None, random_state=0, warm_start=True)

    if optimizer == "adam":
        clf.penalty = 'l2'  # emulate Adam-like behavior with l2 regularization (not real Adam)

    losses = []
    for epoch in range(epochs):
        clf.fit(X_train, y_train)
        probas = clf.predict_proba(X_train)
        log_loss = -np.mean(np.log(probas[np.arange(len(y_train)), y_train]))
        losses.append(log_loss)
    return clf, losses

st.markdown("### 🏋️ Training Models...")
sgd_model, sgd_losses = train_model("sgd")
adam_model, adam_losses = train_model("adam")

# --- Plot Loss Comparison ---
st.markdown("### 📉 Loss vs Iteration")
fig, ax = plt.subplots()
ax.plot(range(1, epochs + 1), sgd_losses, label="SGD", marker='o')
ax.plot(range(1, epochs + 1), adam_losses, label="Adam (emulated)", marker='x')
ax.set_xlabel("Epoch")
ax.set_ylabel("Log Loss")
ax.legend()
st.pyplot(fig)

# --- Confusion Matrix Display ---
def show_confusion_matrix(model, name):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    st.markdown(f"### 🔍 Confusion Matrix: {name}")
    st.write(f"Accuracy: **{acc:.4f}**")
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    st.pyplot(fig)

show_confusion_matrix(sgd_model, "SGD")
show_confusion_matrix(adam_model, "Adam (emulated)")

st.info("Note: scikit-learn does not implement true Adam optimizer for classification. Here, we simulate comparison using warm-started SGD classifiers with varied settings.")
