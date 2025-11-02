import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer

#ml = machine learning; dt = data treatment

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
    if numeric_features == []:
            cat_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ])
            preprocessor = ColumnTransformer(
            transformers=[
            ("cat", cat_transformer, categorical_features)
             ])
    elif categorical_features == []:
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
    import psycopg2
    from dotenv import load_dotenv

    load_dotenv()
    mongo_uri = os.getenv("MONGO_URI")

    try:
        db1_host = os.getenv("DB1_HOST")
        db1_name = os.getenv("DB1_NAME")
        db1_user = os.getenv("DB1_USER")
        db1_pass = os.getenv("DB1_PASS")
        db1_port = os.getenv("DB1_PORT")

        conn_1 = psycopg2.connect(
            host=db1_host,
            database=db1_name,
            user=db1_user,
            password=db1_pass,
            port=db1_port
        )

    except Exception as e:
        print("\n⚠️ PostgreSQL OFFLINE — seguindo apenas com MongoDB")
        print("Erro:", e)
        conn_1 = None

    return mongo_uri, conn_1


def dt_get_data():
    from pymongo import MongoClient
    import pandas as pd
    import numpy as np
    import psycopg2
    import json

    mongo_uri, conn_1 = dt_get_full_conection()

    client = MongoClient(mongo_uri)
    db = client["dbInterEco"]
    collection = db["formulario"]

    mongo_df = pd.DataFrame(list(collection.find({}, {"_id": 0, "numero_cracha": 1, "classificacao_emissao": 1})))

    # tentar conectar no SQL
    try:
        sql_query = """
        SELECT
            f.numero_cracha,
            c.nivel_cargo,
            l.estado AS estado_residencia,
            l.cidade AS cidade_residencia,
            ce.nome AS nome_categoria
        FROM funcionario AS f
        LEFT JOIN cargo AS c ON f.id_cargo = c.id
        LEFT JOIN setor AS s ON c.id_setor = s.id
        LEFT JOIN empresa AS e ON s.id_empresa = e.id
        LEFT JOIN categoria_empresa AS ce ON e.id_categoria = ce.id
        LEFT JOIN localizacao AS l ON f.id_localizacao = l.id
        """

        merged_sql = pd.read_sql(sql_query, conn_1)

    except Exception as e:
        print("\n⚠️ AVISO IMPORTANTE: Banco SQL está offline")
        print("Rodando SOMENTE com dados do MongoDB\n")
        merged_sql = pd.DataFrame(columns=["numero_cracha","nivel_cargo","estado_residencia","cidade_residencia","nome_categoria"])

    # juntar
    data = pd.merge(mongo_df, merged_sql, on="numero_cracha", how="left")

    # remover coluna da chave
    if "numero_cracha" in data.columns:
        data.drop(columns=["numero_cracha"], inplace=True)

    return data


def save_model(model, filename: str, folder: str = "best_model"):
    import joblib
    import os
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    joblib.dump(model, path)
    print(f"✅ Modelo salvo em: {path}")
