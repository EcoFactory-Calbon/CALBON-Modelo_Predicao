def knn(data):
    import functions as fn
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report, accuracy_score

    X, y = fn.ml_separate_features_and_target(data, 'classificacao_emissao')
    y_encoded = LabelEncoder().fit_transform(y)

    num_cols = fn.ml_get_data_numeric(X)
    cat_cols = fn.ml_get_data_string(X, 'classificacao_emissao')

    preprocessor = fn.ml_preprocess_data(num_cols, cat_cols)

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', KNeighborsClassifier())
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X[num_cols + cat_cols], y_encoded,
        test_size=0.3, random_state=42
    )

    model.fit(X_train, y_train)

    param_grid = {
        'classifier__n_neighbors': [3, 5, 7, 9],
        'classifier__weights': ['uniform', 'distance'],
        'classifier__metric': ['euclidean', 'manhattan','minkowski'],
        'classifier__p': [1, 2, 3],
        'classifier__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'classifier__leaf_size': [20, 30, 40, 50],
        'classifier__n_jobs': [-1]
    }

    grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return best_model, classification_report(y_test, y_pred, output_dict=True), accuracy
