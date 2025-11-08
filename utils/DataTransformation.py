import pandas as pd
from scipy.stats import mstats
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

def preprocess_for_prediction(df: pd.DataFrame, scaler: StandardScaler, poly: PolynomialFeatures, feature_columns: list) -> pd.DataFrame:
    """
    Convierte un DataFrame de entrada con las 7 features crudas en todas las features que espera el modelo.
    Asegura que estén todas las columnas y en el orden correcto.
    
    Parámetros:
    - df: DataFrame con las columnas 'pclass','sex','age','sibsp','parch','fare','embarked'
    - scaler: StandardScaler ya entrenado
    - poly: PolynomialFeatures ya entrenado
    - feature_columns: lista con todas las columnas finales que el modelo espera
    
    Retorna:
    - df listo para pasar al modelo
    """
    df = df.copy()
    
    # Imputaciones
    df['age'] = df['age'].fillna(df.groupby('pclass')['age'].transform('median'))
    df['fare'] = df['fare'].fillna(df.groupby('pclass')['fare'].transform('median'))
    df['embarked'] = df['embarked'].fillna('S')
    
    # Codificación categórica
    df['sex'] = df['sex'].map({'male': 0, 'female': 1}).astype(int)
    df = pd.get_dummies(df, columns=['embarked'], drop_first=True)
    
    # Winsorización
    df['fare'] = mstats.winsorize(df['fare'], limits=[0.01, 0.01])
    df['age'] = mstats.winsorize(df['age'], limits=[0.01, 0.01])
    
    # Nuevas características
    df['family_size'] = df['sibsp'] + df['parch'] + 1
    df['is_alone'] = (df['family_size'] == 1).astype(int)
    df['fare_per_person'] = df['fare'] / df['family_size']
    df['is_child'] = (df['age'] < 12).astype(int)
    df['is_senior'] = (df['age'] >= 60).astype(int)
    df['sex_class'] = df['sex'].astype(str) + "_" + df['pclass'].astype(str)
    df = pd.get_dummies(df, columns=['sex_class'], drop_first=True)
    
    # Escalado numérico
    num_cols = ['age','fare','sibsp','parch','pclass','family_size','fare_per_person']
    df[num_cols] = scaler.transform(df[num_cols])
    
    # Features polinómicas
    num_cols = ['age','fare','sibsp','parch','family_size','fare_per_person']
    df_poly = pd.DataFrame(poly.transform(df[num_cols]), columns=poly.get_feature_names_out(num_cols))
    
    # Reemplazar num_cols originales por las polinómicas
    df = pd.concat([df.drop(columns=num_cols), df_poly], axis=1)
    
    # ----- Asegurar que todas las columnas que el modelo espera estén presentes -----
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0  # agregar columna faltante
    df = df[feature_columns]  # reordenar columnas exactamente como las entrenadas
    
    return df
