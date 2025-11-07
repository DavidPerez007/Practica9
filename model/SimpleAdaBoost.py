import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from sklearn.tree import DecisionTreeClassifier

# ==== 1. TU CLASE ====
class SimpleAdaBoost:
    def __init__(self, n_estimators=50, base_estimator=None):
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator or DecisionTreeClassifier(max_depth=1)
        self.alphas_ = []
        self.estimators_ = []
        self.errors_ = []
        self.sample_weights_ = []  # opcional, para guardar la evolución

    def fit(self, X, y):
        # Asegurar y en {-1, 1}
        y = np.where(y == 0, -1, 1)
        n_samples = X.shape[0]
        w = np.ones(n_samples) / n_samples  # pesos iniciales

        for t in range(self.n_estimators):
            stump = DecisionTreeClassifier(max_depth=1)
            stump.fit(X, y, sample_weight=w)
            y_pred = stump.predict(X)

            err = np.sum(w * (y_pred != y)) / np.sum(w)
            err = np.clip(err, 1e-10, 1 - 1e-10)

            alpha = 0.5 * np.log((1 - err) / err)

            w = w * np.exp(-alpha * y * y_pred)
            w = w / np.sum(w)

            self.estimators_.append(stump)
            self.alphas_.append(alpha)
            self.errors_.append(err)
            self.sample_weights_.append(w.copy())

        # guardamos también y transformado por si quieres usarlo luego
        self._y_internal = y
        return self

    def predict(self, X):
        clf_preds = [
            alpha * est.predict(X) for alpha, est in zip(self.alphas_, self.estimators_)
        ]
        y_pred_sum = np.sum(clf_preds, axis=0)
        y_pred = np.sign(y_pred_sum)
        return np.where(y_pred == -1, 0, 1)
    
    def get_params(self, deep=True):
        """
        Devuelve los hiperparámetros del modelo, al estilo scikit-learn.
        """
        return {
            "n_estimators": self.n_estimators,
            "base_estimator": repr(self.base_estimator)
        }

# ==== 2. FUNCIÓN DE LIMPIEZA RÁPIDA ====
def quick_clean_titanic(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Imputar numéricos
    for col in ['age', 'fare']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Imputar categóricas
    for col in ['sex', 'embarked']:
        if col in df.columns and df[col].isna().any():
            moda = df[col].mode(dropna=True)
            if not moda.empty:
                df[col] = df[col].fillna(moda.iloc[0])

    df = df.drop_duplicates()

    if 'survived' in df.columns:
        df['survived'] = df['survived'].astype(int)

    return df

# ==== 3. CARGAR, LIMPIAR Y CODIFICAR ====
train_df = pd.read_csv('train_set.csv')
train_clean = quick_clean_titanic(train_df)

# Codificar sex y embarked
if 'sex' in train_clean.columns:
    train_clean['sex'] = train_clean['sex'].map({'male': 0, 'female': 1}).astype(int)

if 'embarked' in train_clean.columns:
    train_clean = pd.get_dummies(train_clean, columns=['embarked'], drop_first=True)

# Separar X / y
X = train_clean.drop('survived', axis=1).values
y = train_clean['survived'].values

# ==== 4. TRAIN / TEST Y ENTRENAMIENTO ====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

ada = SimpleAdaBoost(n_estimators=50)
ada.fit(X_train, y_train)

y_pred = ada.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))