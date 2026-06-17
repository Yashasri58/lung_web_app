from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json

app = Flask(__name__)

# ── DATA ──────────────────────────────────────────────
data = pd.read_csv("cancer.csv")
le = LabelEncoder()
for col in data.columns:
    data[col] = le.fit_transform(data[col])

X = data.drop("LUNG_CANCER", axis=1)
y = data["LUNG_CANCER"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── MODELS ────────────────────────────────────────────
models = {
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "SVM":                 SVC(probability=True, random_state=42),
    "KNN":                 KNeighborsClassifier(),
    "Decision Tree":       DecisionTreeClassifier(random_state=42),
}

model_results = {}
for name, m in models.items():
    m.fit(X_train, y_train)
    preds = m.predict(X_test)
    cv = cross_val_score(m, X, y, cv=5, scoring='accuracy')
    model_results[name] = {
        "accuracy":  round(accuracy_score(y_test, preds) * 100, 2),
        "precision": round(precision_score(y_test, preds, zero_division=0) * 100, 2),
        "recall":    round(recall_score(y_test, preds, zero_division=0) * 100, 2),
        "f1":        round(f1_score(y_test, preds, zero_division=0) * 100, 2),
        "cv_mean":   round(cv.mean() * 100, 2),
        "cv_std":    round(cv.std() * 100, 2),
    }

# Best model for prediction
best_model_name = max(model_results, key=lambda k: model_results[k]["accuracy"])
model = models[best_model_name]

# ── CHART: MODEL COMPARISON ────────────────────────────
names   = list(model_results.keys())
metrics = ["accuracy", "precision", "recall", "f1"]
colors  = ['#2f81f7', '#3fb950', '#f0883e', '#d2a8ff']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('#161b22')

# Grouped bar chart
x = np.arange(len(names))
width = 0.2
for i, (metric, color) in enumerate(zip(metrics, colors)):
    vals = [model_results[n][metric] for n in names]
    axes[0].bar(x + i * width, vals, width, label=metric.capitalize(), color=color, alpha=0.88)

axes[0].set_facecolor('#1c2230')
axes[0].set_xticks(x + width * 1.5)
axes[0].set_xticklabels(names, rotation=20, ha='right', fontsize=8, color='#e6edf3')
axes[0].set_ylabel('Score (%)', color='#8b949e', fontsize=9)
axes[0].set_ylim(0, 115)
axes[0].tick_params(colors='#8b949e', labelsize=8)
axes[0].spines[:].set_color('#30363d')
axes[0].legend(fontsize=8, facecolor='#1c2230', edgecolor='#30363d', labelcolor='#e6edf3')
axes[0].set_title('Model Comparison — All Metrics', color='#e6edf3', fontsize=10, pad=12)

# CV accuracy line chart
cv_means = [model_results[n]["cv_mean"] for n in names]
cv_stds  = [model_results[n]["cv_std"]  for n in names]
axes[1].plot(names, cv_means, color='#2f81f7', marker='o', linewidth=2, markersize=7)
axes[1].fill_between(names,
    [m - s for m, s in zip(cv_means, cv_stds)],
    [m + s for m, s in zip(cv_means, cv_stds)],
    alpha=0.15, color='#2f81f7')
for i, (n, v) in enumerate(zip(names, cv_means)):
    axes[1].annotate(f'{v:.1f}%', (n, v), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=8, color='#e6edf3')
axes[1].set_facecolor('#1c2230')
axes[1].set_xticks(range(len(names)))
axes[1].set_xticklabels(names, rotation=20, ha='right', fontsize=8, color='#e6edf3')
axes[1].set_ylabel('CV Accuracy (%)', color='#8b949e', fontsize=9)
axes[1].set_ylim(50, 110)
axes[1].tick_params(colors='#8b949e', labelsize=8)
axes[1].spines[:].set_color('#30363d')
axes[1].set_title('5-Fold Cross-Validation Accuracy', color='#e6edf3', fontsize=10, pad=12)

plt.tight_layout(pad=2)
plt.savefig("static/model_comparison.png", dpi=130, bbox_inches='tight', facecolor='#161b22')
plt.close()

# ── CHART: SOCIAL GRAPH ───────────────────────────────
corr = data.corr()
G = nx.Graph()
for feature in corr.columns:
    G.add_node(feature)
for i in range(len(corr.columns)):
    for j in range(i + 1, len(corr.columns)):
        if abs(corr.iloc[i, j]) > 0.3:
            G.add_edge(corr.columns[i], corr.columns[j], weight=abs(corr.iloc[i, j]))

fig, ax = plt.subplots(figsize=(16, 11))
fig.patch.set_facecolor('#1c2230')
ax.set_facecolor('#1c2230')
pos = nx.spring_layout(G, seed=42, k=1.8)
weights = [d['weight'] * 3 for _, _, d in G.edges(data=True)]
nx.draw_networkx_nodes(G, pos, ax=ax, node_size=3200, node_color='#2f81f7', alpha=0.9)
nx.draw_networkx_edges(G, pos, ax=ax, width=weights, edge_color='#58a6ff', alpha=0.5)
nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_color='#e6edf3', font_weight='bold')
ax.axis('off')
plt.tight_layout()
plt.savefig("static/graph.png", dpi=130, bbox_inches='tight', facecolor='#1c2230')
plt.close()

# ── CHART: FEATURE IMPORTANCE ─────────────────────────
rf = models["Random Forest"]
importance = rf.feature_importances_
sorted_idx = np.argsort(importance)
cols = X.columns[sorted_idx]
vals = importance[sorted_idx]
bar_colors = ['#2f81f7' if v >= np.median(vals) else '#58a6ff' for v in vals]

fig, ax = plt.subplots(figsize=(16, 9))
fig.patch.set_facecolor('#1c2230')
ax.set_facecolor('#1c2230')
bars = ax.barh(cols, vals, color=bar_colors, edgecolor='none', height=0.6)
ax.set_xlabel('Importance Score', color='#8b949e', fontsize=9)
ax.tick_params(colors='#e6edf3', labelsize=8)
ax.spines[:].set_color('#30363d')
for bar, val in zip(bars, vals):
    ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
            f'{val:.3f}', va='center', ha='left', color='#8b949e', fontsize=7.5)
plt.tight_layout()
plt.savefig("static/importance.png", dpi=130, bbox_inches='tight', facecolor='#1c2230')
plt.close()

# ── ROUTES ────────────────────────────────────────────
@app.route('/')
def home():
    return render_template("index.html",
        model_results=model_results,
        best_model=best_model_name,
        model_results_json=json.dumps(model_results)
    )

@app.route('/predict', methods=['POST'])
def predict():
    input_df = pd.DataFrame([[
        int(request.form['GENDER']),
        int(request.form['AGE']),
        int(request.form['SMOKING']),
        int(request.form['YELLOW_FINGERS']),
        int(request.form['ANXIETY']),
        int(request.form['PEER_PRESSURE']),
        int(request.form['CHRONIC_DISEASE']),
        int(request.form['FATIGUE']),
        int(request.form['ALLERGY']),
        int(request.form['WHEEZING']),
        int(request.form['ALCOHOL_CONSUMING']),
        int(request.form['COUGHING']),
        int(request.form['SHORTNESS_OF_BREATH']),
        int(request.form['SWALLOWING_DIFFICULTY']),
        int(request.form['CHEST_PAIN'])
    ]], columns=X.columns)

    # Collect prediction + probability from every model
    per_model = {}
    probs = []
    for name, m in models.items():
        prob = round(float(m.predict_proba(input_df)[0][1]) * 100, 1)
        pred = int(m.predict(input_df)[0])
        per_model[name] = {"prob": prob, "pred": pred}
        probs.append(prob)

    # Combined probability = weighted average by model accuracy
    weights_acc = [model_results[n]["accuracy"] for n in models]
    combined_prob = round(
        sum(p * w for p, w in zip(probs, weights_acc)) / sum(weights_acc), 1
    )
    high_risk = combined_prob >= 50

    return render_template("index.html",
        prediction_text="high" if high_risk else "low",
        combined_prob=combined_prob,
        per_model=per_model,
        per_model_json=json.dumps(per_model),
        model_results=model_results,
        best_model=best_model_name,
        model_results_json=json.dumps(model_results)
    )

if __name__ == "__main__":
    app.run(debug=True, port=8080)
