"""Treina e avalia um classificador simples de risco com base em frases sintomáticas."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA = BASE_DIR / "data" / "risk_classification.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Arquivo CSV com colunas 'frase' e 'situacao'")
    parser.add_argument("--export-model", type=Path, help="Arquivo .joblib para salvar o pipeline treinado")
    parser.add_argument("--report", type=Path, help="Arquivo JSON para salvar métricas detalhadas")
    parser.add_argument("--test-size", type=float, default=0.3, help="Proporção de dados para teste (padrão: 0.3)")
    parser.add_argument("--random-state", type=int, default=42, help="Semente aleatória para reprodutibilidade")
    return parser.parse_args()


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


def export_metrics(path: Path, report: dict, conf_matrix) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "classification_report": report,
        "confusion_matrix": conf_matrix.tolist(),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise SystemExit(f"Arquivo de dados não encontrado: {args.data}")

    df = pd.read_csv(args.data)
    if {"frase", "situacao"} - set(df.columns):
        raise SystemExit("O CSV deve conter as colunas 'frase' e 'situacao'.")

    X_train, X_test, y_train, y_test = train_test_split(
        df["frase"],
        df["situacao"],
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=df["situacao"],
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = pipeline.score(X_test, y_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"Acurácia: {accuracy:.2f}")
    print("\nRelatório de classificação:")
    for label, metrics in report.items():
        if label in {"accuracy", "macro avg", "weighted avg"}:
            continue
        precision = metrics["precision"]
        recall = metrics["recall"]
        f1 = metrics["f1-score"]
        support = metrics["support"]
        print(f"  Classe '{label}': precisão={precision:.2f}, revocação={recall:.2f}, F1={f1:.2f}, suporte={support}")

    print("\nMatriz de confusão:")
    print(conf_matrix)

    if args.export_model:
        args.export_model.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, args.export_model)
        print(f"\nModelo salvo em {args.export_model}")

    if args.report:
        export_metrics(args.report, report, conf_matrix)
        print(f"Relatório detalhado salvo em {args.report}")


if __name__ == "__main__":
    main()
