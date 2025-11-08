import numpy as np
import pandas as pd
from scipy.stats import mstats

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import class_weight, resample
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

import joblib


# clase adaboost simple
class SimpleAdaBoost:
    def __init__(self, n_estimators=50, base_estimator=None):
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator or DecisionTreeClassifier(max_depth=1)
        self.alphas_, self.estimators_, self.errors_, self.sample_weights_ = [], [], [], []

    def fit(self, X, y):
        y = np.where(y == 0, -1, 1)
        cw = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
        w = np.where(y == -1, cw[0], cw[1]); w /= np.sum(w)

        for _ in range(self.n_estimators):
            stump = DecisionTreeClassifier(max_depth=self.base_estimator.max_depth)
            stump.fit(X, y, sample_weight=w)
            y_pred = stump.predict(X)

            err = np.clip(np.sum(w * (y_pred != y)) / np.sum(w), 1e-10, 1 - 1e-10)
            alpha = 0.5 * np.log((1 - err) / err)

            w *= np.exp(-alpha * y * y_pred); w /= np.sum(w)

            self.estimators_.append(stump); self.alphas_.append(alpha)
            self.errors_.append(err); self.sample_weights_.append(w.copy())
        return self

    def predict(self, X):
        pred_sum = np.sum([a * e.predict(X) for a, e in zip(self.alphas_, self.estimators_)], axis=0)
        return np.where(np.sign(pred_sum) == -1, 0, 1)

    def get_params(self):
        return {
            "model": "AdaBoostClassifier",
            "n_estimators": self.n_estimators,
            "base_estimator": str(self.base_estimator)
        }

# tenemos que considerar que los datos que nos lleguen serán sucios,
# así que la función debe implementar un preprocesamiento
def preprocess_titanic(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # imputación informada
    df["age"] = df["age"].fillna(df.groupby("pclass")["age"].transform("median"))
    df["fare"] = df["fare"].fillna(df.groupby("pclass")["fare"].transform("median"))
    df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

    # codificación básica
    df["sex"] = df["sex"].map({"male": 0, "female": 1}).astype(int)
    df = pd.get_dummies(df, columns=["embarked"], drop_first=True)

    # winsorización para outliers
    df["fare"] = mstats.winsorize(df["fare"], limits=[0.01, 0.01])
    df["age"] = mstats.winsorize(df["age"], limits=[0.01, 0.01])

    # nuevas features
    df["is_child"] = (df["age"] < 12).astype(int)
    df["is_senior"] = (df["age"] >= 60).astype(int)
    df["family_size"] = df["sibsp"] + df["parch"] + 1
    df["is_alone"] = (df["family_size"] == 1).astype(int)
    df["fare_per_person"] = df["fare"] / df["family_size"]
    df["sex_class"] = df["sex"].astype(str) + "_" + df["pclass"].astype(str)
    df = pd.get_dummies(df, columns=["sex_class"], drop_first=True)

    # escalar numéricas
    scaler = StandardScaler()
    num_cols = ["age", "fare", "sibsp", "parch", "pclass", "family_size", "fare_per_person"]
    df[num_cols] = scaler.fit_transform(df[num_cols])

    df["survived"] = df["survived"].astype(int)
    df = df.drop_duplicates()

    # balanceo
    df_major = df[df["survived"] == 0]
    df_minor = df[df["survived"] == 1]
    df_bal = pd.concat([df_major, resample(df_minor, replace=True, n_samples=len(df_major), random_state=42)])
    return df_bal


# construcción del dataset con polinomios y split
def build_dataset(csv_path: str = "train_set.csv"):
    df = pd.read_csv(csv_path)
    df_clean = preprocess_titanic(df)

    num_cols = ["age", "fare", "sibsp", "parch", "family_size", "fare_per_person"]
    poly = PolynomialFeatures(degree=2, include_bias=False)
    df_poly = pd.DataFrame(poly.fit_transform(df_clean[num_cols]),
                           columns=poly.get_feature_names_out(num_cols),
                           index=df_clean.index)

    df_enhanced = pd.concat([df_clean.drop(columns=num_cols), df_poly], axis=1)

    X = df_enhanced.drop("survived", axis=1).values
    y = df_enhanced["survived"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


# entrenamiento y guardado del modelo
def train_and_save(csv_path: str = "train_set.csv", model_path: str = "modelo.pkl", n_estimators: int = 150):
    X_train, X_test, y_train, y_test = build_dataset(csv_path)

    simple_ada = SimpleAdaBoost(n_estimators=n_estimators, base_estimator=DecisionTreeClassifier(max_depth=2))
    simple_ada.fit(X_train, y_train)

    joblib.dump(simple_ada, model_path)

    y_pred = simple_ada.predict(X_test)
    print("métricas del adaboost personalizado")
    print("accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, digits=3))

    return simple_ada


# ejecutar entrenamiento 
if __name__ == "__main__":
    train_and_save()
