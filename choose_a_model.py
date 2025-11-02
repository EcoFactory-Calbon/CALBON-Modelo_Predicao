from collections import defaultdict
from decision_tree import decision_tree
from logisticRegressionCV import logisticRegressionCV
from knn import knn
import functions as fn
import subprocess
import os
import numpy as np


data = fn.dt_get_data()


for col in data.columns:
    data[col] = data[col].apply(
        lambda x: x if isinstance(x, (int, float, str)) else str(x)
    )

# detectar quantos tipos diferentes aparecem em cada coluna
type_counts = data.applymap(lambda x: type(x).__name__).nunique()

print("=== tipos detectados por coluna (ex.: {'str': 10, 'list': 2}) ===")
for c, t in type_counts.items():
    print(c, t)

# mostrar primeiras linhas com tipos (para inspeção)
print("\n=== primeiras 10 linhas com tipos de cada coluna ===")
print(data.head(10).applymap(lambda x: type(x).__name__))

# linhas que possuem valores não escalares em QUALQUER coluna
def is_scalar(x):
    return isinstance(x, (str, int, float, bool, type(None), np.generic))

mask = data.applymap(lambda x: not is_scalar(x)).any(axis=1)
print("\n=== índices com alguma célula não-escalar (mostrando até 20) ===")
print(data[mask].head(20))
print("Total de linhas com pelo menos uma célula não-escalar:", mask.sum())


tree_model, tree_report, tree_accuracy = decision_tree(data)
logreg_model, logreg_report, logreg_accuracy = logisticRegressionCV(data)
knn_model, knn_report, knn_accuracy = knn(data)

results = {
    "Decision Tree": {"model": tree_model, "report": tree_report, "accuracy": tree_accuracy},
    "Logistic Regression CV": {"model": logreg_model, "report": logreg_report, "accuracy": logreg_accuracy},
    "KNN": {"model": knn_model, "report": knn_report, "accuracy": knn_accuracy}
}

metric_fields = ["precision", "recall", "f1-score"]

wins = defaultdict(int)

for name, data_item in results.items():
    rpt = data_item["report"]
    data_item["valid_classes"] = [
        k for k in rpt.keys()
        if isinstance(rpt[k], dict) and all(m in rpt[k] for m in metric_fields)
    ]

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

best_name = max(results.keys(), key=lambda n: (wins[n], results[n]["accuracy"]))
best_model = results[best_name]["model"]

print("wins per model:", dict(wins))
print("chosen model:", best_name)


fn.save_model(best_model, "best_model.pkl", folder="best_model")


repo_path = os.path.abspath(".")
model_path = os.path.join(repo_path, "best_model/best_model.pkl")

subprocess.run(["git", "add", model_path], cwd=repo_path)
subprocess.run(["git", "commit", "-m", f"Atualização automática do best_model: {best_name}"], cwd=repo_path)
subprocess.run(["git", "push"], cwd=repo_path)

print("🚀 Modelo salvo e commit enviado ao Git!")
