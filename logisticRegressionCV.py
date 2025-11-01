import pandas as pd
def logisticRegressionCV(data:pd.DataFrame):
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score
    import functions as fn

    X, y = fn.ml_separate_features_and_target(data, 'classificacao_emissao')
    y_encoded = LabelEncoder().fit_transform(y)
    df_num_columns = fn.ml_get_data_numeric(X)
    df_cat_columns = fn.ml_get_data_string(X, 'classificacao_emissao')
    preprocessor = fn.ml_preprocess_data(df_num_columns, df_cat_columns)
    if preprocessor is not None:
        model = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegressionCV(
                Cs=[3,5, 10, 20,30],
                max_iter=6000, #como eu coloquei saga e elasticnet, o ideal é aumentar o max_iter - MAS AUMENTAR MUITO MESMO -> eles demoram pacas
                solver='saga', # 'saga' suporta penalty='elasticnet' e multi_class='multinomial'
                penalty='elasticnet', #vou deixar assim por enquanto pq vou mudar do dataset, o elasticnet combina l1 e l2 - l1 encolhe os coeficientes de forma suave e o l2 força alguns coeficientes a zero; a diferença está no calculo feito e o elasticnet tenta balancear os dois
                class_weight='balanced', #defini o peso balanceado conforme a distribuicao das classes
                cv=10, #deixei 10 para mais robustez
                random_state=42, #Num padrão para reprodutibilidade - não influencia mt no resultado
                n_jobs=-1, # Usar todos os núcleos disponíveis para acelerar o treinamento
                verbose=1, # Para ver o progresso do treinamento
                multi_class='multinomial', #Ele treina todas as classes de uma vez então para 10k de dados é melhor do que os outros
                l1_ratios=[0.1, 0.5, 0.9] #Mistura l1 e l2 na regularização
            ))
        ])
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return model, classification_report(y_test, y_pred, output_dict=True), accuracy