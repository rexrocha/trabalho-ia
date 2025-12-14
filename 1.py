import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Carregar dados
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

X = train.drop(["id", "target"], axis=1)
y = train["target"]
X_test = test.drop("id", axis=1)
ids_test = test["id"]

# Codificar rótulos
encoder = LabelEncoder()
y_enc = encoder.fit_transform(y)

# Pipeline simples
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("model", XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss"
    ))
])

# Validação cruzada
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X, y_enc, cv=cv, scoring="accuracy", n_jobs=-1)
print(f"Acurácia média: {scores.mean():.4f}")

# Treinar com todos os dados
pipeline.fit(X, y_enc)

# Gerar previsões
preds = pipeline.predict(X_test)
preds = encoder.inverse_transform(preds)

# Salvar submissão
sub = pd.DataFrame({"id": ids_test, "target": preds})
sub.to_csv("submission.csv", index=False)
print("submission.csv gerado.")