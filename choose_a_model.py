from collections import defaultdict
from decision_tree import decision_tree
from logisticRegressionCV import logisticRegressionCV
from knn import knn
import functions as fn
import subprocess
import os

data = fn.dt_get_data()

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

    # Filtrar apenas classes reais (evitar 'accuracy', 'macro avg', etc)
    valid_classes = [
        k for k in rpt.keys()
        if isinstance(rpt[k], dict) and all(m in rpt[k] for m in metric_fields)
    ]

    data_item["valid_classes"] = valid_classes

# Comparação entre modelos
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

# comparar accuracy
acc_values = {name: float(d["accuracy"]) for name, d in results.items()}
max_acc = max(acc_values.values())
for n, v in acc_values.items():
    if v == max_acc:
        wins[n] += 1

# escolher melhor modelo
best_name = max(results.keys(), key=lambda n: (wins[n], results[n]["accuracy"]))
best_model = results[best_name]["model"]

print("wins per model:", dict(wins))
print("chosen model:", best_name)

# salvar
fn.save_model(best_model, "best_model.pkl", folder="best_model")


# -----------------------------------------------
# ---------  🔥 FAZER COMMIT AUTOMÁTICO ----------
# -----------------------------------------------

repo_path = os.path.abspath(".")
model_path = os.path.join(repo_path, "best_model/best_model.pkl")

subprocess.run(["git", "add", model_path], cwd=repo_path)
subprocess.run(["git", "commit", "-m", f"Atualização automática do best_model: {best_name}"], cwd=repo_path)
subprocess.run(["git", "push"], cwd=repo_path)

print("🚀 Modelo salvo e commit enviado ao Git!")
