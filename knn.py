import pandas as pd
def knn(data: pd.DataFrame):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import LabelEncoder
    import functions as fn

    X, y = fn.ml_separate_features_and_target(data, 'classificacao_emissao')
    y_encoded = LabelEncoder().fit_transform(y)
    df_num_columns = fn.ml_get_data_numeric(X)
    df_cat_columns = fn.ml_get_data_string(X, 'classificacao_emissao')
    preprocessor = fn.ml_preprocess_data(df_num_columns, df_cat_columns)
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", KNeighborsClassifier())
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    param_grid = {
        'classifier__n_neighbors': [3, 5, 7, 9],
        'classifier__weights': ['uniform', 'distance'],
        'classifier__metric': ['euclidean', 'manhattan','minkowski'], #botei todas as métricas pra ele testar de td
        'classifier__p': [1, 2, 3], #tenho q ver se isso aq vai coisar o modelo
        'classifier__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'classifier__leaf_size': [20, 30, 40, 50], #O padrão para 10k é 30, mas com buscamos o melhor modelo, visei testar outros valores, seguindo o padrão de 5 elementos usado na DecisionTree
        'classifier__n_jobs': [-1] #Usar todos os núcleos disponíveis -> São 10k de dados
    }

    grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return best_model, classification_report(y_test, y_pred, output_dict=True), accuracy
