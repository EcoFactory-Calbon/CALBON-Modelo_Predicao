import pandas as pd
def decision_tree(data: pd.DataFrame):
    from sklearn.pipeline import Pipeline
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report
    from sklearn.metrics import accuracy_score
    import functions as fn

    # === Carregar variáveis de ambiente ===

    X, y = fn.ml_separate_features_and_target(data, 'classificacao_emissao')
    y_encoded = LabelEncoder().fit_transform(y)
    df_num_columns = fn.ml_get_data_numeric(X)
    df_cat_columns = fn.ml_get_data_string(X, 'classificacao_emissao')
    preprocessor = fn.ml_preprocess_data(df_num_columns, df_cat_columns)
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('tree', DecisionTreeClassifier(max_depth=3, random_state=0))
    ])
    # dividindo em conjunto de treino e test
    X_train, X_test, y_train, y_test = train_test_split(X,y_encoded,
                                                        test_size=0.2, 
                                                        random_state=42)
        
    param_grid = {
        'tree__max_depth': [3, 5, 10, 15, 20, 30, None],       # Profundidades - Fazer range em tds
        # removed None from min_samples_split to avoid invalid type errors
        'tree__min_samples_split': [3, 5, 10, 15, 20, 30],     # Min. para dividir nó
        'tree__min_samples_leaf': [3, 5, 10, 15, 20, 30],      # Min. em folha
        'tree__criterion': ['gini', 'entropy', 'log_loss'],    # Critério de divisão
        'tree__min_weight_fraction_leaf': [0.0, 0.1, 0.15, 0.2, 0.3, 0.5], # Fração mínima de peso na folha
        'tree__max_features': [None],                          # Número máximo de features consideradas para divisão
        'tree__random_state': [42],                            # Semente para reprodutibilidade
        'tree__max_leaf_nodes': [None, 10, 20, 30, 50],        # Número máximo de nós folha
        #min_impurity_decrease, class_weight e ccp_alpha não foram adicionados visando o objetivo de que a árvore tenha a liberdade de se aprofundar e se ajustar aos dados existentes, uma vez que eles serão inseridos em um banco de dados real e dinâmico, onde novos dados serão constantemente adicionados.
    }


    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5, #Isso aq é cross validation, 5 para balancear entre tempo e performance
        scoring='accuracy'
    )

    grid.fit(X_train, y_train)
    #Usando pipeline, ele já tratou todos os dados usados adiante
    best_model = grid.best_estimator_
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return best_model, classification_report(y_test, y_pred, output_dict=True), accuracy
