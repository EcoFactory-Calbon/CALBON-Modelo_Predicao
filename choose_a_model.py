from collections import defaultdict
from decision_tree import decision_tree
from logisticRegressionCV import logisticRegressionCV
import functions as fn
from knn import knn

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

all_keys = set()
for data in results.values():
    rpt = data["report"]
    if isinstance(rpt, str):
        raise RuntimeError("classification reports must be dicts. Use output_dict=True when calling classification_report.")
    all_keys.update(rpt.keys())

wins = defaultdict(int)

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

# Escolhe o melhor modelo (maior número de vitórias, desempate por accuracy)
best_name = max(results.keys(), key=lambda n: (wins.get(n, 0), results[n].get("accuracy", 0)))
best_model = results[best_name]["model"]

# Resumo final
print("wins per model:", dict(wins))
print("chosen model:", best_name)

# salva o modelo escolhido
for name, data in results.items():
    if name == best_name:
        fn.save_model(data["model"], "best_model.pkl")