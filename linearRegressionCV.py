import pandas as pd
def logisticRegressionCV(data:pd.DataFrame):
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score
    import functions as fn

    # separa dataset
    X, y = fn.ml_separate_features_and_target(data, 'classificacao_emissao')
    y_encoded = LabelEncoder().fit_transform(y)

    # identifica colunas corretas
    num_cols = fn.ml_get_data_numeric(X)
    cat_cols = fn.ml_get_data_string(X, 'classificacao_emissao')

    # pega só colunas válidas
    features = num_cols + cat_cols

    # preprocess dinâmico
    preprocessor = fn.ml_preprocess_data(num_cols, cat_cols)

    # pipeline
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegressionCV(
            Cs=[3,5, 10, 20,30],
            max_iter=6000,
            solver='saga',
            penalty='elasticnet',
            class_weight='balanced',
            cv=10,
            random_state=42,
            n_jobs=-1,
            multi_class='multinomial',
            l1_ratios=[0.1, 0.5, 0.9]
        ))
    ])

    # 🛠️ AQUI estava o erro: X tinha TODAS as colunas não processadas
    X_train, X_test, y_train, y_test = train_test_split(
        X[features],      # <--- só coluna válida
        y_encoded,
        test_size=0.2,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return model, classification_report(y_test, y_pred, output_dict=True), accuracy
