# TITANIC DATASET - DATA CLEANING, ML MODELS & NEURAL NETWORK

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score



# 1. LOAD DATA


df = pd.read_csv("titanic.csv")



# 2. CLEAN DATA


# Remove Cabin because it contains many missing values.
df = df.drop(columns=["Cabin"])

# Fill missing Age with the median.
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill missing Embarked with the most common value.
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Create FamilySize.
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

# Create IsAlone.
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

# Extract passenger title from Name.
df["Title"] = df["Name"].str.extract(r",\s*([^.]*)\.", expand=False)

# Keep common titles and group uncommon ones as Rare.
df["Title"] = df["Title"].replace(
    ["Lady", "Countess", "Capt", "Col", "Don", "Dr",
     "Major", "Rev", "Sir", "Jonkheer", "Dona"],
    "Rare"
)

df["Title"] = df["Title"].replace({
    "Mlle": "Miss",
    "Ms": "Miss",
    "Mme": "Mrs"
})



# 3. SELECT FEATURES AND TARGET


X = df[
    [
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked",
        "FamilySize",
        "IsAlone",
        "Title"
    ]
]

y = df["Survived"]



# 4. TRAIN / TEST SPLIT


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)



# 5. PREPROCESSING


numeric_features = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "FamilySize",
    "IsAlone"
]

categorical_features = [
    "Sex",
    "Embarked",
    "Title"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            numeric_features
        ),
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical_features
        )
    ]
)


# 6. MODEL 1 - LOGISTIC REGRESSION


logistic_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=2000))
])

logistic_model.fit(X_train, y_train)

y_pred_logistic = logistic_model.predict(X_test)

logistic_accuracy = accuracy_score(
    y_test,
    y_pred_logistic
)

print("Logistic Regression Accuracy:",
      round(logistic_accuracy * 100, 2), "%")



# 7. MODEL 2 - RANDOM FOREST


random_forest = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=500,
        max_depth=7,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        class_weight="balanced"
    ))
])

random_forest.fit(X_train, y_train)

y_pred_rf = random_forest.predict(X_test)

rf_accuracy = accuracy_score(
    y_test,
    y_pred_rf
)

print("Random Forest Accuracy:",
      round(rf_accuracy * 100, 2), "%")



# 8. MODEL 3 - SVM


svm_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", SVC(
        C=1.5,
        kernel="rbf",
        gamma="scale"
    ))
])

svm_model.fit(X_train, y_train)

y_pred_svm = svm_model.predict(X_test)

svm_accuracy = accuracy_score(
    y_test,
    y_pred_svm
)

print("SVM Accuracy:",
      round(svm_accuracy * 100, 2), "%")



# 9. NEURAL NETWORK


neural_network = Pipeline([
    ("preprocessor", preprocessor),
    ("model", MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        alpha=0.0005,
        learning_rate_init=0.001,
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,
        random_state=42
    ))
])

neural_network.fit(X_train, y_train)

y_pred_nn = neural_network.predict(X_test)

nn_accuracy = accuracy_score(
    y_test,
    y_pred_nn
)

print("Neural Network Accuracy:",
      round(nn_accuracy * 100, 2), "%")



# 10. FINAL ACCURACY COMPARISON


results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "SVM",
        "Neural Network"
    ],
    "Accuracy (%)": [
        round(logistic_accuracy * 100, 2),
        round(rf_accuracy * 100, 2),
        round(svm_accuracy * 100, 2),
        round(nn_accuracy * 100, 2)
    ]
})

print("\nFINAL ACCURACY RESULTS")
print(results)

print("\nBest Model:",
      results.loc[results["Accuracy (%)"].idxmax(), "Model"])

print("Best Accuracy:",
      results["Accuracy (%)"].max(), "%")
