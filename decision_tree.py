def decision_tree(data):
    import functions as fn
    from sklearn.pipeline import Pipeline
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report, accuracy_score

    X, y = fn.ml_separate_features_and_target(data, 'classificacao_emissao')
    y_encoded = LabelEncoder().fit_transform(y)

    num_cols = fn.ml_get_data_numeric(X)      # lista
    cat_cols = fn.ml_get_data_string(X, 'classificacao_emissao')  # lista

    # 🔥🔥🔥 >>> CORREÇÃO OBRIGATÓRIA <<< 🔥🔥🔥
    if len(num_cols) == 0 and len(cat_cols) == 0:
        raise ValueError("ERROR: Nenhuma coluna numérica ou categórica encontrada. As listas estão vazias.")

    if len(num_cols) == 0:
        print("Aviso: Nenhuma coluna numérica encontrada. Continuando apenas com categóricas.")

    if len(cat_cols) == 0:
        print("Aviso: Nenhuma coluna categórica encontrada. Continuando apenas com numéricas.")
    # 🔥🔥🔥 >>> FIM DA CORREÇÃO <<< 🔥🔥🔥

    df_num = X[num_cols]
    df_cat = X[cat_cols]

    preprocessor = fn.ml_preprocess_data(num_cols, cat_cols)

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('tree', DecisionTreeClassifier(max_depth=3, random_state=0))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X[num_cols + cat_cols], y_encoded,
        test_size=0.2, random_state=42
    )

    param_grid = {
        'tree__max_depth': [3,5,10,15,20,30,None],
        'tree__min_samples_split': [3,5,10,15,20,30],
        'tree__min_samples_leaf': [3,5,10,15,20,30],
        'tree__criterion': ['gini','entropy','log_loss'],
        'tree__min_weight_fraction_leaf': [0.0,0.1,0.15,0.2,0.3,0.5],
        'tree__max_features': [None],
        'tree__random_state': [42],
        'tree__max_leaf_nodes': [None,10,20,30,50]
    }

    grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return best_model, classification_report(y_test, y_pred, output_dict=True), accuracy
