"""Assistente baseado em regras para sugerir diagnósticos a partir de relatos sintomáticos."""
from __future__ import annotations

import argparse
import csv
import json
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_REPORTS = BASE_DIR / "data" / "relatos_pacientes.txt"
DEFAULT_MAPPING = BASE_DIR / "data" / "mapa_sintomas_doencas.csv"
DEFAULT_SEVERITY = "moderado"
SEVERITY_ORDER = {
    "baixo": 0,
    "moderado": 1,
    "alto": 2,
    "critico": 3,
    "crítico": 3,
}
SEVERITY_DISPLAY_ORDER = ["crítico", "alto", "moderado", "baixo"]


def normalize(text: str) -> str:
    """Lowercase text and strip accents to help with fuzzy matching."""
    normalized = unicodedata.normalize("NFD", text.lower())
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn")


def normalize_severity(value: str | None) -> str:
    if not value:
        return DEFAULT_SEVERITY
    cleaned = normalize(value).strip()
    return {
        "critico": "crítico",
        "critic": "crítico",
        "alto": "alto",
        "moderado": "moderado",
        "baixo": "baixo",
    }.get(cleaned, DEFAULT_SEVERITY)


@dataclass
class SymptomRule:
    disease: str
    severity: str
    symptoms: List[str]

    def matches(self, normalized_report: str) -> List[str]:
        hits: List[str] = []
        for raw_symptom in self.symptoms:
            normalized_symptom = normalize(raw_symptom)
            if normalized_symptom and normalized_symptom in normalized_report:
                hits.append(raw_symptom)
        return hits

    @property
    def severity_rank(self) -> int:
        return SEVERITY_ORDER.get(self.severity, SEVERITY_ORDER[normalize(DEFAULT_SEVERITY)])


def load_symptom_rules(path: Path) -> List[SymptomRule]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        rules: List[SymptomRule] = []
        for raw_row in reader:
            if not raw_row:
                continue
            disease = (raw_row.get("associated_disease") or "").strip()
            if not disease:
                continue
            collected: List[str] = []
            for column in ("symptom_1", "symptom_2"):
                cell = (raw_row.get(column) or "").strip()
                if not cell:
                    continue
                for item in cell.split(";"):
                    symptom = item.strip()
                    if symptom:
                        collected.append(symptom)
            if collected:
                severity = normalize_severity(raw_row.get("severity_level"))
                rules.append(SymptomRule(disease=disease, severity=severity, symptoms=collected))
    return rules


def load_reports(path: Path) -> List[str]:
    with path.open(encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def diagnose(reports: Iterable[str], rules: Iterable[SymptomRule]) -> List[dict]:
    results: List[dict] = []
    for index, report in enumerate(reports, start=1):
        normalized_report = normalize(report)
        matched_symptoms = set()
        matches = {}
        highest_rank = -1
        highest_label = None

        for rule in rules:
            hits = rule.matches(normalized_report)
            if hits:
                matched_symptoms.update(hits)
                key = (rule.disease, rule.severity)
                matches.setdefault(key, set()).update(hits)
                if rule.severity_rank > highest_rank:
                    highest_rank = rule.severity_rank
                    highest_label = rule.severity

        matched_rules = [
            {
                "disease": disease,
                "severity": severity,
                "matched_symptoms": sorted(symptoms),
            }
            for (disease, severity), symptoms in matches.items()
        ]
        matched_rules.sort(
            key=lambda item: (
                -SEVERITY_ORDER.get(item["severity"], 0),
                item["disease"],
            )
        )

        results.append(
            {
                "report_id": index,
                "report": report,
                "matched_symptoms": sorted(matched_symptoms),
                "diagnosis_matches": matched_rules,
                "max_severity": highest_label,
            }
        )
    return results


def summarize_results(results: List[dict]) -> dict:
    total = len(results)
    with_symptoms = sum(1 for item in results if item["matched_symptoms"])
    with_diagnosis = sum(1 for item in results if item["diagnosis_matches"])
    severity_counter = Counter(item["max_severity"] for item in results if item["max_severity"])
    return {
        "reports": total,
        "reports_with_symptoms": with_symptoms,
        "reports_with_diagnosis": with_diagnosis,
        "severity_distribution": dict(severity_counter),
    }


def export_results(results: List[dict], destination: Path, fmt: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fmt = fmt.lower()
    if fmt == "json":
        destination.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    elif fmt == "csv":
        fieldnames = [
            "report_id",
            "report",
            "matched_symptoms",
            "diagnosis_matches",
            "max_severity",
        ]
        with destination.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for item in results:
                writer.writerow(
                    {
                        "report_id": item["report_id"],
                        "report": item["report"],
                        "matched_symptoms": "; ".join(item["matched_symptoms"]),
                        "diagnosis_matches": "; ".join(
                            f"{match['disease']} (gravidade: {match['severity']})" for match in item["diagnosis_matches"]
                        ),
                        "max_severity": item["max_severity"] or "nao_classificado",
                    }
                )
    else:
        raise ValueError(f"Formato de exportação não suportado: {fmt}")


def infer_export_format(path: Path, explicit_format: str | None) -> str:
    if explicit_format:
        return explicit_format.lower()
    suffix = path.suffix.lower().lstrip(".")
    if suffix in {"json", "csv"}:
        return suffix
    raise ValueError("Não foi possível inferir o formato de exportação; use --format {json,csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=DEFAULT_REPORTS, help="Arquivo com frases de pacientes")
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING, help="Mapa de sintomas e doenças em CSV")
    parser.add_argument("--export", type=Path, help="Arquivo para salvar os diagnósticos estruturados (JSON ou CSV)")
    parser.add_argument(
        "--format",
        choices=("json", "csv"),
        help="Formato do arquivo exportado. Quando omitido, é deduzido pela extensão.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rules = load_symptom_rules(args.mapping)
    reports = load_reports(args.reports)

    if not rules:
        raise SystemExit(f"Nenhuma regra foi carregada a partir de {args.mapping}")
    if not reports:
        raise SystemExit(f"Nenhum relato foi encontrado em {args.reports}")

    results = diagnose(reports, rules)
    for item in results:
        print(f"Relato {item['report_id']}: {item['report']}")
        if item["matched_symptoms"]:
            print("  Sintomas reconhecidos:")
            for symptom in item["matched_symptoms"]:
                print(f"    - {symptom}")
        else:
            print("  Sintomas reconhecidos: nenhum mapeado")

        if item["diagnosis_matches"]:
            print("  Diagnósticos sugeridos:")
            for match in item["diagnosis_matches"]:
                print(f"    - {match['disease']} (gravidade: {match['severity']})")
            if item["max_severity"]:
                print(f"  Gravidade máxima indicada: {item['max_severity']}")
        else:
            print("  Diagnósticos sugeridos: revisão manual necessária")
        print()

    summary = summarize_results(results)
    print("Resumo geral:")
    print(f"  Relatos analisados: {summary['reports']}")
    print(f"  Relatos com sintomas reconhecidos: {summary['reports_with_symptoms']}")
    print(f"  Relatos com diagnóstico sugerido: {summary['reports_with_diagnosis']}")
    if summary["severity_distribution"]:
        print("  Distribuição de gravidade sugerida:")
        for severity in SEVERITY_DISPLAY_ORDER:
            count = summary["severity_distribution"].get(severity)
            if count:
                print(f"    - {severity}: {count}")
    else:
        print("  Distribuição de gravidade sugerida: nenhum caso classificado")

    if args.export:
        export_format = infer_export_format(args.export, args.format)
        export_results(results, args.export, export_format)
        print(f"\nResultados exportados para {args.export} ({export_format.upper()}).")


if __name__ == "__main__":
    main()
