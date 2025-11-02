import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer

# ml = machine learning; dt = data treatment

def ml_separate_features_and_target(data: pd.DataFrame, target_column: str):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y

def ml_get_data_string(data: pd.DataFrame, target_column: str):
    cat_features = data.select_dtypes(include=['object', 'category']).columns.to_list()
    cat_features = [col for col in cat_features if col != target_column]
    return cat_features

def ml_get_data_numeric(data: pd.DataFrame):
    num_features = data.select_dtypes(include=['number']).columns.to_list()
    return num_features

def ml_preprocess_data(numeric_features: list=[], categorical_features: list = []):
    if numeric_features == [] and categorical_features != []:
        cat_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", cat_transformer, categorical_features)
            ])
    elif categorical_features == [] and numeric_features != []:
        num_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("minmaxscaler", MinMaxScaler())
        ])
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_transformer, numeric_features)
            ])
    elif numeric_features == [] and categorical_features == []:
        preprocessor = None
    else:
        num_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("minmaxscaler", MinMaxScaler())
        ])

        cat_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_transformer, numeric_features),
                ("cat", cat_transformer, categorical_features)
            ])
    
    return preprocessor



def dt_get_full_conection():
    import os
    from dotenv import load_dotenv

    load_dotenv()

    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise ValueError("⚠️ MONGO_URI não foi definida no ambiente.")

    return mongo_uri  


def dt_get_data():
    from pymongo import MongoClient
    import pandas as pd

    mongo_uri = dt_get_full_conection()

    client = MongoClient(mongo_uri)
    db = client["dbInterEco"]
    collection = db["formulario"]

    data = pd.DataFrame(list(collection.find({}, {"_id": 0})))

    if data.empty:
        print("⚠️ AVISO: A coleção 'formulario' está vazia ou não existe.")
    
    return data


def save_model(model, filename: str, folder: str = "best_model"):
    import joblib
    import os
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    joblib.dump(model, path)
    print(f"✅ Modelo salvo em: {path}")
