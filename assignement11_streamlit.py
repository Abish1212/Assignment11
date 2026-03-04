
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Naive Bayes Classifier — Upload CSV")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of dataset:")
    st.dataframe(df.head())

    target = st.selectbox("Select target column", options=df.columns)
    test_size = st.slider("Test size (fraction)", 0.05, 0.5, 0.2, 0.05)
    model_name = st.selectbox("Model", ["Gaussian", "Multinomial", "Bernoulli"])
    auto_bin_target = st.checkbox(
        "If target is continuous numeric, convert it to classes", value=True
    )
    n_bins = st.slider("Number of target bins", 2, 10, 4)

    if st.button("Run"):
        X = df.drop(target, axis=1)
        y = df[target]

        if y.isna().any():
            st.error("Target column has missing values. Please clean data and try again.")
            st.stop()

        if pd.api.types.is_numeric_dtype(y):
            unique_ratio = y.nunique(dropna=True) / len(y)
            looks_continuous = y.nunique(dropna=True) > 20 and unique_ratio > 0.1

            if looks_continuous:
                if auto_bin_target:
                    y = pd.qcut(y, q=n_bins, duplicates="drop").astype(str)
                    st.info(
                        f"Converted continuous target to {y.nunique()} class bins using quantiles."
                    )
                else:
                    st.error(
                        "Selected target appears continuous numeric. "
                        "Naive Bayes classifier needs class labels. "
                        "Enable binning or choose a categorical target column."
                    )
                    st.stop()

        X = pd.get_dummies(X)
        X = X.fillna(0)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        if model_name == "Multinomial" and (X_train < 0).any().any():
            st.error(
                "Multinomial model requires non-negative features. "
                "Choose Gaussian/Bernoulli or preprocess features."
            )
            st.stop()

        if model_name == "Gaussian":
            model = GaussianNB()
        elif model_name == "Multinomial":
            model = MultinomialNB()
        else:
            model = BernoulliNB()

        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_acc = round(accuracy_score(y_train, train_pred), 3)
        test_acc = round(accuracy_score(y_test, test_pred), 3)

        st.metric("Train accuracy", f"{train_acc}")
        st.metric("Test accuracy", f"{test_acc}")

        cm = confusion_matrix(y_test, test_pred)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

        st.success("Done")

else:
    st.info("Upload a CSV to get started")
