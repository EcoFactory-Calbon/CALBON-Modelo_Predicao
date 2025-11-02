def logisticRegressionCV(data):
    import functions as fn
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score, classification_report

    X, y = fn.ml_separate_features_and_target(data, 'classificacao_emissao')
    y_encoded = LabelEncoder().fit_transform(y)

    df_num = fn.ml_get_data_numeric(X)
    df_cat = fn.ml_get_data_string(X, 'classificacao_emissao')

    num_cols = df_num.columns.tolist()
    cat_cols = df_cat.columns.tolist()

    preprocessor = fn.ml_preprocess_data(num_cols, cat_cols)

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegressionCV(
            Cs=[3,5,10,20,30],
            max_iter=6000,
            solver='saga',
            penalty='elasticnet',
            class_weight='balanced',
            cv=10,
            random_state=42,
            n_jobs=-1,
            verbose=1,
            multi_class='multinomial',
            l1_ratios=[0.1,0.5,0.9]
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X[num_cols + cat_cols], y_encoded,
        test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, classification_report(y_test, y_pred, output_dict=True), accuracy
