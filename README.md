<h1 align="center">🌱 Treinamento de Modelos 🌍</h1>

---

## 📖 O que o script faz?

O Fluxo do script é separado em partes entre modelos, orquestrador e notebooks. Foram escolhidos 3 modelos preditivos **com base nos conhecimentos estatísticos do time de dados do segundo ano**, criamos um arquivo `.py` para cada um dos modelos e também o arquivo `choose_a_model.py`, que executa e compara cada modelo, para assim, salvar somente o modelo com melhor desempenho na pasta `best_model`. Os notebooks servem como referência de fluxo em cada modelo, além de serem a primeira versão de cada um. O target de cada modelo é **`nivel_emissao`**, a fim de validar a pegada de carbono do usuário baseado nos dados fornecidos.

> 💡 **Nota:** Reforçamos que os notebooks presentes neste diretório servem **única** e **exclusivamente** como apoio visual. O funcionamento correto de cada modelo está presente em cada arquivo `.py` com o mesmo nome do modelo utilizado

---

## 🎲 Sobre os dados utilizados

Os dados vêm de duas fontes diferentes: `PostgresSQL` e `MongoDB`. 

## 🏦 PostgresSQL

| Coluna | Descrição |
|:-------|:-----------|
| **`numero_cracha`** | ID do funcionário, vem da tabela funcionário |
| **`nivel_cargo`** | Cargo alto, médio ou baixo é o `nivel_cargo`, vem da tabela cargo|
| **`cidade`** | Cidade de residência do funcionário, vem da tabela localização |
| **`estado`** | Estado de residência do funcionário, vem da tabela localização |
| **`categoria`** | Tipo de categoria da empresa que o funcionário trabalha (Alimentício, Energia etc), vem da tabela categoria_empresa |


## 🏦 MongoDB

| Coluna | Descrição |
|:-------|:-----------|
| **`numero_cracha`** | ID do funcionário, vem da collection formulario para relacionar com o numero_cracha do sql |
| **`nivel_emissao`** | Emissão alta, média ou baixa, vem da collection formulario |


> 💡**Nota** A divisão de treino e teste foram separados em *80%* de treino e *20%* de teste, como padrão de divisão. Cada banco tem no mínimo 10k de dados.

---

## ✅ Modelos escolhidos e parâmetros utilizados


Cada modelo foi feito usando pipelines fornecidas pela biblioteca do `scikit-learn`, cada um deles recebe um **preprocessador** com a seguinte estrutura:
- Recebe as colunas numéricas e categóricas
- Valida a existência para cada tipo de coluna (pode existir somente colunas categóricas etc)
- Aplica a padronização necessário para cada tipo de dado
#### Por que escolhemos cada padronização? 
| Tipo | Descrição |
|:-------|:-----------|
| **`SimpleImputer(strategy="mean")`** | O `SimpleImputer` substitui valores ausentes pela média da coluna, a média mantém a distribuição dos dados e evita enviesar o modelo com substituições arbitrárias. |
| **`MinMaxScaler()`** | O `MinMaxScaler()` transforma todos os valores numéricos para o intervalo `[0, 1]`, Diferente do StandardScaler, o MinMaxScaler preserva a forma original da distribuição e é mais versátil para dados com limites conhecidos ou que serão usados em algoritmos baseados em distância. Também é útil para modelos sensíveis a magnitude (como o KNN) |
| **`SimpleImputer(strategy="most_frequent")`** | O `SimpleImputer(strategy="most_frequent")` substitui valores ausentes pela categoria mais frequente, evita perda de dados e mantém coerência sem criar novas classes artificiais. Essa abordagem funciona bem em qualquer tipo de modelo  |
| **`OneHotEncoder(handle_unknown="ignore")`** | O `OneHotEncoder(handle_unknown="ignore")` converte variáveis categóricas em variáveis binárias (dummies), cria uma representação numérica compatível com qualquer modelo de ML. O parâmetro `handle_unknown="ignore"` evita erros quando aparecem categorias inéditas no conjunto de teste, garantindo generalização segura. |

O `ColumnTransformer` aplica os pipelines adequados (numérico e categórico) em colunas diferentes de forma simultânea.

> 💡**Nota:** Todos os métodos foram feitos com o objetivo de funcionar em todos/muitos modelos diferentes de ML, tornando os métodos universais e reutilizáveis. 


```bash
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

```

## 🪞 Modelos 

<details>
<summary> 🌲 Decision Tree </summary>

| Parâmetro | Descrição |
|:-------|:-----------|
| **`'tree__max_depth': [3, 5, 10, 15, 20, 30, None]`** | Define a profundidade máxima da árvore, ou seja, quantos níveis de divisão podem existir, a faixa `3 a 30` cobre desde árvores rasas (gerais, com pouca complexidade) até árvores profundas (mais ajustadas aos dados). Além disso, `None` foi incluído para permitir que a árvore cresça sem limite de profundidade, testando o caso extremo. |
| **`'tree__min_samples_split': [3, 5, 10, 15, 20, 30]`** | Indica o mínimo de amostras necessárias para dividir um nó, valores de 3 a 30 seguem a mesma escala para garantir consistência, além disso, quanto maior o valor, mais amostras são exigidas para criar novos nós, o que reduz overfitting. |
| **`tree__min_samples_leaf': [3, 5, 10, 15, 20, 30]`** | Define o mínimo de amostras que cada nó folha deve conter, segue a mesma faixa de valores de 3 a 30 para manter coerência na escala de controle de complexidade. para manter coerência na escala de controle de complexidade. |
| **`'tree__criterion': ['gini', 'entropy', 'log_loss']`** | Foram usados os 3 critérios disponíveis segundo a documentação do **`scikit-learn`**. `gini` = padrão, eficiente e simples; `entropy` = considera a impureza de forma mais detalhada; `log_loss` = mais sensível a probabilidades previstas. |
| **`'tree__min_weight_fraction_leaf': [0.0, 0.1, 0.15, 0.2, 0.3, 0.5]`** | Determina a fração mínima do peso total das amostras necessária em cada folha, testa de 0 (sem restrição) até 0.5 (folhas muito grandes), cobrindo escalas pequenas e médias.  |
| **`'tree__max_features': [None]`** | Indica quantas features são consideradas para cada divisão, mantido como None para usar todas as variáveis disponíveis.  |
| **`'tree__random_state': [42]`** | Define a semente aleatória para reprodutibilidade, o valor fixo 42 é padrão e facilita reproduzir resultados. |
| **`'tree__max_leaf_nodes': [None, 10, 20, 30, 50]`** | Limita o número máximo de folhas, valores 10 a 50 seguem a escala crescente (Números maiores considerando que são 10k de dados) e `None` para dar liberdade de crescimento livre. |

```bash
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


```

</details>

<details>
<summary> 🟢 KNN </summary>

| Parâmetro | Descrição |
|:-------|:-----------|
| **`'classifier__n_neighbors': [3, 5, 7, 9]`** | Número de vizinhos considerados para classificar uma amostra, valores ímpares de 3 a 9 foram escolhidos para evitar empates e manter a coerência com a escala usada faixa 3 a 30. Quanto maior o número, mais “suave” e generalizado o modelo e quanto menor, mais sensível aos ruídos. |
| **`'classifier__weights': ['uniform', 'distance']`** | Define como cada vizinho contribui na decisão. `uniform`: todos os vizinhos têm peso igual; `distance`: vizinhos mais próximos têm mais peso.  |
| **`'classifier__metric': ['euclidean', 'manhattan','minkowski']`** | Define a métrica de distância usada para calcular a proximidade entre pontos. `euclidean`: distância padrão (reta entre dois pontos); `manhattan`: soma das distâncias absolutas, útil em dados com muitas features independentes; `minkowski`: generaliza o `euclidean` e `manhattan` (controlada pelo parâmetro p). |
| **`'classifier__p': [1, 2, 3]`** | Define o expoente da distância de Minkowski. `p=1` = Manhattan; `p=2` = Euclidiana, `p=3` = distância cúbica (mais sensível a grandes diferenças). |
| **`'classifier__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']`** | Define o algoritmo usado para encontrar os vizinhos mais próximos. `auto`: o sklearn escolhe automaticamente o método mais eficiente; `ball_tree` e `kd_tree`: métodos baseados em estruturas de árvore; `brute`: faz busca direta (mais lento, mas garante exatidão).   |
| **`'classifier__n_jobs': [-1]`** | Define quantos núcleos do processador usar, -1 = usa todos os núcleos disponíveis.  |


```bash
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


```
</details>

<details>
<summary> 📈 LogisticRegressionCV </summary>

| Parâmetro | Descrição |
|:-------|:-----------|
| **`Cs=[3,5, 10, 20,30]`** | Define os valores do hiperparâmetro de regularização C (inverso da força de regularização), q2uanto menor o número, maior é a regularização. |
| **`max_iter=6000`** | Define o número máximo de iterações para o algoritmo convergir, o solver saga é mais custoso e, com elasticnet, exige mais iterações e 6000 garante convergência mesmo em bases grandes (como 10k linhas). |
| **`solver='saga'`** | Define o algoritmo usado para otimização, `saga` é o único solver que suporta penalty='elasticnet' e multi_class='multinomial', é escalável e eficiente para grandes datasets. |
| **`penalty='elasticnet'`** | Define o tipo de regularização aplicada. O elasticnet combina `L1` e `L2`. `L1` =  zera alguns coeficientes, seleção de features; `L2` =  encolhe coeficientes suavemente, estabilidade. |
| **`class_weight='balanced'`** | Compensa desbalanceamentos de classes, isso evita que classes majoritárias dominem o modelo, melhorando o recall das classes minoritárias. |
| **`cv=10`** | Número de divisões na validação cruzada interna, o valor 10 é um padrão robusto, oferecendo boa estimativa de generalização sem exagerar no tempo de treino. Isso reduz a variância nas métricas de validação e melhora a confiança nos hiperparâmetros escolhidos. |
| **`random_state=42`** | Semente fixa para reprodutibilidade, garante que os mesmos resultados possam ser reproduzidos em execuções futuras. |
| **`n_jobs=-1`** | Utiliza todos os núcleos de CPU disponíveis, isso é essencial para acelerar o LogisticRegressionCV, que realiza múltiplos treinos paralelamente. Além disso, otimiza o tempo de execução, especialmente com cv=10 e Cs múltiplos. |
| **`verbose=1`** | Exibe o progresso do treinamento durante a execução, útil para monitorar o tempo de convergência e desempenho durante o ajuste com grandes bases. |
| **`multi_class='multinomial'`** | Define a estratégia para problemas multiclasse, `multinomial` treina todas as classes simultaneamente, ao contrário de `ovr` (one-vs-rest). Isso fornece previsões mais consistentes quando há múltiplas classes e interdependência entre elas, ideal para datasets com várias categorias de emissão. |
| **`l1_ratios=[0.1, 0.5, 0.9]`** | Define a proporção entre L1 e L2 na penalização elasticnet, permite testar diferentes graus de regularização combinada, ajustando o modelo à complexidade dos dados. Quanto maior o número, mais agressiva é a regularização na seleção de variáveis. |


> 💡 **Nota:** o LogisticRegressionCV é o LogisticRegression mais otimizado, como o C crítico para mostrar a regularização é importante, o scikit-learn também disponibiliza uma versão já com Cross Validation do Logistic Regression. Resumindo, é o Logistic Regression com GridSearchCV implementado.


```bash
import pandas as pd
def LogisticRegressionCV(data:pd.DataFrame):
    from sklearn.Logistic_model import LogisticRegressionCV
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


```

</details>

 > 💡 Nota: Para abrir a explicação, clique na seta na esquerda. Ela contém o funcionamente e explicação final de cada ponto do código e também da saída.
---

## Funções utilizadas pelos modelos

| Função | Descrição |
|:-------|:-----------|
| **`ml_separate_features_and_target(data: pd.DataFrame, target_column: str)`** | Separa o target das demais colunas. |
| **`ml_get_data_string(data: pd.DataFrame, target_column: str)`** | Retorna uma lista com os nomes das colunas categóricas (tipo texto ou categoria), excluindo a coluna-alvo. |
| **`ml_get_data_numeric(data: pd.DataFrame)`** | Retorna uma lista com os nomes das colunas numéricas do DataFrame. |
| **`ml_preprocess_data(numeric_features: list=[], categorical_features: list = [])`** | Cria o pré-processador de dados (pipeline do scikit-learn) com base no tipo de variável. *(Explicação do pré-processamento acima)* |
| **`dt_get_full_conection()`** | faz a conexão do `PostgresSQL` e no `MongoDB` |
| **`dt_get_data()`** | Junta dados das duas fontes, `MongoDB` e `PostgreSQL` |
| **`save_model(model, filename: str, folder: str = "best_model")`** | Salva o modelo treinado em um arquivo .joblib |

>💡 **Nota:** Funções com o prefixo `ml` são funções para os modelos e funções com o prefixo `dt` são funções para tratamento e processamento de dados.



---

## 🤖 Passo a Passo de como funciona o orquestrador - `choose_a_model.py`

- Importar bibliotecas necessárias:
  
| Import | Descrição |
|:-------|:-----------|
| **`collections`** | defaultdict é usado para criar um dicionário que inicializa automaticamente valores padrão (no caso, inteiros iniciando em 0), facilita a contagem de "vitórias" dos modelos, sem precisar checar se a chave existe.  |
| **`Modelos`** | Importação de cada arquivo `.py` contendo os modelos criados pelo time Calbon |
| **`functions`** | Funções de `functions.py`, *(especificadas acima)* |

```bash

from collections import defaultdict
from decision_tree import decision_tree
from LogisticRegressionCV import LogisticRegressionCV
import functions as fn
from knn import knn

```

##

- Carregamento dos dados
  
```bash
data = fn.dt_get_data()
```


##
- Treinamento dos modelos e pegar as métricas, a acurácia também vem para casos de desempate 
  
```bash
tree_model, tree_report, tree_accuracy = decision_tree(data)
logreg_model, logreg_report, logreg_accuracy = logisticRegressionCV(data)
knn_model, knn_report, knn_accuracy = knn(data)

```

##
- Organiza os resultados

```bash
results = {
    "Decision Tree": {"model": tree_model, "report": tree_report, "accuracy": tree_accuracy},
    "Logistic Regression CV": {"model": logreg_model, "report": logreg_report, "accuracy": logreg_accuracy},
    "KNN": {"model": knn_model, "report": knn_report, "accuracy": knn_accuracy}
}

```
##
- Preparação das métricas, aqui define quais métricas serão usadas na comparação entre os modelos.
```bash
metric_fields = ["precision", "recall", "f1-score"]
```
##
- Junta todas as classes (ou rótulos) existentes nos relatórios do `report`
```bash
metric_fields = ["precision", "recall", "f1-score"]
```
##
- Armazena quantas vezes cada modelo teve a melhor métrica.
```bash
  wins = defaultdict(int)
```
##
- Compara cada modelo em cada métrica (precision, recall e f1-score) para cada classe.
```bash
all_keys = set()
for data in results.values():
    rpt = data["report"]
    if isinstance(rpt, str):
        raise RuntimeError("classification reports must be dicts. Use output_dict=True when calling classification_report.")
    all_keys.update(rpt.keys())

  for key in all_keys:
    for field in metric_fields:
        values = {}
        for name, data in results.items():
            rpt = data["report"]
            try:
                val = rpt[key][field]
            except Exception:
                continue
            try:
                values[name] = float(val)
            except Exception:
                continue
        if not values:
            continue

        max_val = max(values.values())
        for name, v in values.items():
            if v == max_val:
                wins[name] += 1
```
##
- Comparação por acurácia
```bash
acc_values = {}
for name, data in results.items():
    try:
        acc_values[name] = float(data["accuracy"])
    except Exception:
        continue

if acc_values:
    rounded = {n: round(v, 6) for n, v in acc_values.items()}
    max_acc = max(rounded.values())
    for n, v in rounded.items():
        if v == max_acc:
            wins[n] += 1
```
##
- Escolhe o melhor modelo, printa o resumo final e salva
```bash
best_name = max(results.keys(), key=lambda n: (wins.get(n, 0), results[n].get("accuracy", 0)))
best_model = results[best_name]["model"]

print("wins per model:", dict(wins))
print("chosen model:", best_name)

for name, data in results.items():
    if name == best_name:
        fn.save_model(data["model"], "best_model.pkl")
```

---

<h3 align="center">✨ Desenvolvido para CALBON - Treinamento de Modelo de Predição 🌿</h3>
