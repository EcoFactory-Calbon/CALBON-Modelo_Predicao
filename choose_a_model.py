from collections import defaultdict
from decision_tree import decision_tree
from logisticRegressionCV import logisticRegressionCV
from knn import knn
import functions as fn
import subprocess
import os
import numpy as np
import joblib


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

    # SALVAR MODELO - GARANTINDO QUE SALVA NA PASTA DO GIT
    print("\n=== SALVANDO MELHOR MODELO ===")
    
    # Encontrar a raiz do repositório Git
    try:
        repo_root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True
        ).stdout.strip()
        print(f"📁 Raiz do repositório Git: {repo_root}")
    except:
        repo_root = os.path.abspath(".")
        print(f"⚠️  Não consegui encontrar a raiz do Git, usando diretório atual: {repo_root}")
    
    # Definir caminhos ABSOLUTOS dentro do repositório Git
    model_folder = os.path.join(repo_root, "best_model")
    model_filename = "best_model.pkl"
    model_path = os.path.join(model_folder, model_filename)
    
    print(f"📁 Pasta do modelo: {model_folder}")
    print(f"📄 Arquivo: {model_filename}")
    print(f"📍 Caminho completo: {model_path}")
    
    # Garantir que a pasta existe
    os.makedirs(model_folder, exist_ok=True)
    print("✅ Pasta best_model criada/verificada")
    
    # VERIFICAR se a pasta best_model está no Git
    git_check = subprocess.run(
        ["git", "ls-files", "best_model/"],
        cwd=repo_root,
        capture_output=True,
        text=True
    )
    
    if git_check.returncode == 0 and git_check.stdout.strip():
        print("✅ Pasta best_model já está rastreada pelo Git")
    else:
        print("⚠️  Pasta best_model não está no Git (será adicionada)")
    
    # Salvar o modelo DIRETAMENTE no caminho absoluto
    try:
        print("💾 Salvando modelo com joblib...")
        joblib.dump(best_model, model_path)
        print("✅ Modelo salvo com joblib diretamente no repositório Git")
    except Exception as e:
        print(f"❌ Erro ao salvar modelo: {e}")
        exit(1)
    
    # VERIFICAÇÃO CRÍTICA - o arquivo foi salvo?
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path)
        print(f"✅ VERIFICAÇÃO: Modelo salvo com SUCESSO!")
        print(f"   📍 Local: {model_path}")
        print(f"   💾 Tamanho: {file_size} bytes")
        print(f"   🏆 Modelo: {best_name}")
        print(f"   📊 Acurácia: {best_accuracy:.4f}")
    else:
        print(f"❌ ERRO CRÍTICO: Modelo não foi salvo em {model_path}")
        exit(1)

    # OPERAÇÕES GIT - GARANTIR QUE O MODELO SEJA COMMITADO
    print("\n=== ENVIANDO PARA GIT ===")
    
    try:
        # Verificar status antes
        print("📋 Status do Git antes das alterações:")
        status_before = subprocess.run(
            ["git", "status", "--short"],
            cwd=repo_root,
            capture_output=True,
            text=True
        )
        print(status_before.stdout if status_before.stdout else "   (nenhuma alteração)")
        
        # Adicionar a pasta best_model INTEIRA ao Git
        print("\n➕ Adicionando pasta best_model ao Git...")
        add_result = subprocess.run(
            ["git", "add", "best_model/"],
            cwd=repo_root,
            capture_output=True,
            text=True
        )
        
        if add_result.returncode == 0:
            print("✅ Pasta best_model adicionada ao staging area")
        else:
            print(f"⚠️  Erro ao adicionar pasta: {add_result.stderr}")
            # Tentar adicionar apenas o arquivo
            print("🔄 Tentando adicionar apenas o arquivo do modelo...")
            subprocess.run(["git", "add", model_path], cwd=repo_root, check=True)
            print("✅ Arquivo do modelo adicionado")
        
        # Verificar status após add
        print("\n📋 Status após git add:")
        status_after = subprocess.run(
            ["git", "status", "--short"],
            cwd=repo_root,
            capture_output=True,
            text=True
        )
        print(status_after.stdout if status_after.stdout else "   (nenhuma alteração)")
        
        # Fazer commit
        commit_message = f"🤖 Auto-update: best_model {best_name} (accuracy: {best_accuracy:.4f})"
        print(f"\n💾 Fazendo commit: {commit_message}")
        
        commit_result = subprocess.run(
            ["git", "commit", "-m", commit_message],
            cwd=repo_root,
            capture_output=True,
            text=True
        )
        
        if commit_result.returncode == 0:
            print("✅ Commit realizado com sucesso!")
            
            # Fazer push
            print("\n🚀 Enviando para repositório remoto...")
            push_result = subprocess.run(
                ["git", "push"],
                cwd=repo_root,
                capture_output=True,
                text=True
            )
            
            if push_result.returncode == 0:
                print("✅ Push realizado com sucesso!")
                print("🎉 Modelo salvo e enviado para o Git!")
            else:
                print(f"⚠️  Push falhou: {push_result.stderr}")
                print("💡 O modelo foi salvo localmente no repositório Git.")
        
        else:
            print(f"⚠️  Commit falhou: {commit_result.stderr}")
            print("💡 Possível motivo: nada para commitar (arquivo já estava commitado)")
            print("💡 O modelo foi salvo localmente no repositório Git.")
            
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Erro no processo Git: {e}")
        print("💡 O modelo foi salvo localmente no repositório Git.")
    except FileNotFoundError:
        print("⚠️  Git não encontrado no sistema")
        print("💡 O modelo foi salvo localmente.")
    except Exception as e:
        print(f"⚠️  Erro inesperado no Git: {e}")
        print("💡 O modelo foi salvo localmente no repositório Git.")

    # VERIFICAÇÃO FINAL
    print("\n" + "="*60)
    print("🎯 VERIFICAÇÃO FINAL")
    print("="*60)
    print(f"🏆 Melhor modelo: {best_name}")
    print(f"📊 Acurácia: {best_accuracy:.4f}")
    print(f"📍 Local do modelo: {model_path}")
    print(f"💾 Tamanho do arquivo: {os.path.getsize(model_path)} bytes")
    
    # Verificar se está na pasta do Git
    if model_path.startswith(repo_root):
        print("✅ LOCALIZAÇÃO: Modelo salvo DENTRO do repositório Git")
    else:
        print("❌ LOCALIZAÇÃO: Modelo salvo FORA do repositório Git")
    
    if os.path.exists(model_path):
        print("✅ STATUS: Modelo salvo com SUCESSO!")
    else:
        print("❌ STATUS: Falha ao salvar o modelo!")
    
    print("="*60)
    print("✅ PROCESSO CONCLUÍDO!")


if __name__ == "__main__":
    main()