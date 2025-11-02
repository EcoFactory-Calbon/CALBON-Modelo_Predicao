from collections import defaultdict
from decision_tree import decision_tree
from logisticRegressionCV import logisticRegressionCV
from knn import knn
import functions as fn
import subprocess
import os
import numpy as np


def main():
    print("=== INICIANDO SELEÇÃO DE MODELO ===")
    
    data = fn.dt_get_data()
    
    print(f"Dataset carregado: {data.shape}")

    print("\n=== DIAGNÓSTICO DO DATASET ===")
    print(f"Shape do dataset: {data.shape}")
    print(f"Colunas: {list(data.columns)}")

    target_column = "classificacao_emissao"
    if target_column not in data.columns:
        print(f"❌ ERRO: Coluna target '{target_column}' não encontrada!")
        print(f"Colunas disponíveis: {list(data.columns)}")
        # Tentar encontrar coluna target alternativa
        possible_targets = [col for col in data.columns if 'classificacao' in col.lower() or 'emissao' in col.lower()]
        if possible_targets:
            print(f"Possíveis colunas target: {possible_targets}")
        exit(1)

    X = data.drop(columns=[target_column])
    y = data[target_column]

    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Valores únicos no target: {y.unique()}")

    print("\n=== TIPOS DE DADOS ===")
    print(X.dtypes)

    numeric_features = X.select_dtypes(include=['number']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    print(f"Features numéricas ({len(numeric_features)}): {numeric_features}")
    print(f"Features categóricas ({len(categorical_features)}): {categorical_features}")

    if not numeric_features and not categorical_features:
        print("❌ ERRO: Nenhuma feature encontrada!")
        exit(1)

    if not numeric_features:
        print("⚠️  Nenhuma feature numérica encontrada. Criando feature dummy...")
        data['dummy_numeric_feature'] = 1
        print("✅ Feature numérica dummy criada")

    print("\n=== VALORES NULOS ===")
    print(data.isnull().sum())

    # Limpeza de dados não escalares
    print("\n=== VERIFICANDO DADOS NÃO ESCALARES ===")
    for col in data.columns:
        data[col] = data[col].apply(
            lambda x: x if isinstance(x, (int, float, str)) else str(x)
        )

    type_counts = data.applymap(lambda x: type(x).__name__).nunique()

    print("=== TIPOS DETECTADOS POR COLUNA ===")
    for c, t in type_counts.items():
        print(f"{c}: {t}")

    print("\n=== PRIMEIRAS 5 LINHAS COM TIPOS ===")
    print(data.head(5).applymap(lambda x: type(x).__name__))

    def is_scalar(x):
        return isinstance(x, (str, int, float, bool, type(None), np.generic))

    mask = data.applymap(lambda x: not is_scalar(x)).any(axis=1)
    print(f"\n=== LINHAS COM CÉLULAS NÃO-ESCALARES: {mask.sum()} ===")
    if mask.sum() > 0:
        print(data[mask].head(10))
    else:
        print("✅ Nenhuma célula não-escalar encontrada")

    print("\n" + "="*50)
    print("TREINANDO MODELOS...")
    print("="*50)

    try:
        tree_model, tree_report, tree_accuracy = decision_tree(data)
        print("✅ Decision Tree concluído")
    except Exception as e:
        print(f"❌ Erro no Decision Tree: {e}")
        tree_model, tree_report, tree_accuracy = None, {}, 0.0

    try:
        logreg_model, logreg_report, logreg_accuracy = logisticRegressionCV(data)
        print("✅ Logistic Regression CV concluído")
    except Exception as e:
        print(f"❌ Erro no Logistic Regression: {e}")
        logreg_model, logreg_report, logreg_accuracy = None, {}, 0.0

    try:
        knn_model, knn_report, knn_accuracy = knn(data)
        print("✅ KNN concluído")
    except Exception as e:
        print(f"❌ Erro no KNN: {e}")
        knn_model, knn_report, knn_accuracy = None, {}, 0.0

    # Coletar resultados
    results = {}
    if tree_model is not None:
        results["Decision Tree"] = {"model": tree_model, "report": tree_report, "accuracy": tree_accuracy}
    if logreg_model is not None:
        results["Logistic Regression CV"] = {"model": logreg_model, "report": logreg_report, "accuracy": logreg_accuracy}
    if knn_model is not None:
        results["KNN"] = {"model": knn_model, "report": knn_report, "accuracy": knn_accuracy}

    if not results:
        print("❌ Nenhum modelo foi treinado com sucesso!")
        exit(1)

    print(f"\n=== MODELOS TREINADOS COM SUCESSO: {len(results)} ===")

    metric_fields = ["precision", "recall", "f1-score"]
    wins = defaultdict(int)

    for name, data_item in results.items():
        rpt = data_item["report"]
        data_item["valid_classes"] = [
            k for k in rpt.keys()
            if isinstance(rpt[k], dict) and all(m in rpt[k] for m in metric_fields)
        ]
        print(f"{name}: {len(data_item['valid_classes'])} classes válidas")

    for field in metric_fields:
        for cls in set().union(*[v["valid_classes"] for v in results.values()]):
            values = {}
            for name, data_item in results.items():
                rpt = data_item["report"]
                if cls in data_item["valid_classes"]:
                    try:
                        values[name] = float(rpt[cls][field])
                    except:
                        pass

            if not values:
                continue

            max_val = max(values.values())
            for name, v in values.items():
                if v == max_val:
                    wins[name] += 1

    acc_values = {name: float(d["accuracy"]) for name, d in results.items()}
    max_acc = max(acc_values.values())
    for n, v in acc_values.items():
        if v == max_acc:
            wins[n] += 1

    print(f"\n=== VITÓRIAS POR MODELO ===")
    for name, win_count in wins.items():
        print(f"{name}: {win_count} vitórias (acuracia: {acc_values[name]:.4f})")

    best_name = max(results.keys(), key=lambda n: (wins[n], results[n]["accuracy"]))
    best_model = results[best_name]["model"]
    best_accuracy = results[best_name]["accuracy"]

    print(f"\n🏆 MELHOR MODELO: {best_name}")
    print(f"📊 Acurácia: {best_accuracy:.4f}")
    print(f"🎯 Vitórias: {wins[best_name]}")

    fn.save_model(best_model, "best_model.pkl", folder="best_model")

    try:
        repo_path = os.path.abspath(".")
        model_path = os.path.join(repo_path, "best_model/best_model.pkl")

        subprocess.run(["git", "add", model_path], cwd=repo_path, check=True)
        subprocess.run(["git", "commit", "-m", f"Atualização automática do best_model: {best_name} (acc: {best_accuracy:.4f})"], 
                      cwd=repo_path, check=True)
        subprocess.run(["git", "push"], cwd=repo_path, check=True)

        print("🚀 Modelo salvo e commit enviado ao Git!")
        
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Aviso: Não foi possível fazer commit automático: {e}")
        print("O modelo foi salvo localmente, mas não foi commitado.")
    except Exception as e:
        print(f"⚠️  Aviso: Erro inesperado no Git: {e}")

    print("\n✅ PROCESSO CONCLUÍDO!")


if __name__ == "__main__":
    main()