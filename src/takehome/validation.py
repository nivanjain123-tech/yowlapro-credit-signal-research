"""Single source of truth for take-home submission structure checks."""
from __future__ import annotations

import ast
import csv
import io
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import nbformat

Status = Literal["pass", "warn", "fail"]

EXPERIMENT_LOG_COLUMNS = [
    "exp_id",
    "logged_at",
    "hypothesis_id",
    "test_type",
    "split",
    "n",
    "metric",
    "estimate",
    "ci_low",
    "ci_high",
    "p_value",
    "decision",
    "kill_reason",
    "minutes_spent",
    "notes",
]

REPORT_HEADINGS = [
    "Question",
    "What I tested and why",
    "What survived",
    "What I killed and why",
    "How I checked I was not fooling myself",
    "Limitations",
    "How to reproduce",
]

NOTEBOOK_HEADINGS = [
    "Setup and data loading",
    "Observations",
    "Pre-registered hypotheses",
    "Cheap first-pass tests",
    "Holdout tests",
    "Ways I might be fooling myself",
    "Conclusions",
]

KILL_REASONS = {
    "no_signal",
    "concentration",
    "leakage",
    "small_sample",
    "survivorship",
    "multiple_testing",
    "other",
}

TEST_TYPES = {"cheap", "full", "robustness"}
SPLITS = {"discovery", "holdout"}
DECISIONS = {"keep", "kill", "inconclusive"}

ALLOWED_EXTRA_PACKAGES = frozenset(
    {
        "scikit-learn",
        "sklearn",
        "seaborn",
        "plotly",
        "altair",
        "polars",
        "xarray",
        "lightgbm",
        "xgboost",
    }
)

RESULTS_HYPOTHESIS_REQUIRED = frozenset(
    {
        "hypothesis_id",
        "feature",
        "estimate",
        "ci",
        "p_value",
        "n",
        "decision",
        "kill_reason",
    }
)

MAX_EMBEDDED_IMAGE_BYTES = 1024 * 1024


@dataclass
class CheckResult:
    path: str
    status: Status
    message: str
    code: str = ""
    hint: str = ""


@dataclass
class ValidationReport:
    results: list[CheckResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "results": [
                {
                    "path": r.path,
                    "status": r.status,
                    "message": r.message,
                    "code": r.code,
                    "hint": r.hint,
                }
                for r in self.results
            ],
            "ok": self.ok,
        }

    @property
    def ok(self) -> bool:
        return all(r.status != "fail" for r in self.results)


def _heading_present(md: str, heading: str) -> bool:
    pattern = re.compile(
        rf"^#+\s*{re.escape(heading)}\s*$",
        re.IGNORECASE | re.MULTILINE,
    )
    return bool(pattern.search(md))


def _notebook_headings(nb_path: Path) -> list[str]:
    nb = nbformat.read(nb_path, as_version=4)
    headings: list[str] = []
    for cell in nb.cells:
        if cell.cell_type != "markdown":
            continue
        for line in cell.source.splitlines():
            m = re.match(r"^#+\s+(.+?)\s*$", line.strip())
            if m:
                headings.append(m.group(1))
    return headings


def _notebook_section_text(nb_path: Path, section_heading: str) -> str:
    nb = nbformat.read(nb_path, as_version=4)
    collecting = False
    chunks: list[str] = []
    target = section_heading.lower()
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            for line in cell.source.splitlines():
                m = re.match(r"^#+\s+(.+?)\s*$", line.strip())
                if m and target in m.group(1).lower():
                    collecting = True
                    continue
                if collecting and re.match(r"^#+\s+", line.strip()):
                    return "\n".join(chunks)
        if collecting:
            if cell.cell_type == "markdown":
                chunks.append(cell.source)
            elif cell.cell_type == "code":
                chunks.append(cell.source)
    return "\n".join(chunks)


def _count_embedded_images(md: str) -> int:
    total = 0
    for match in re.finditer(
        r"!\[[^\]]*\]\((data:image/[^;]+;base64,([A-Za-z0-9+/=]+))\)",
        md,
        re.IGNORECASE,
    ):
        total += len(match.group(2))
    return total


def _parse_requirements(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "==" not in line:
            continue
        name, _, version = line.partition("==")
        out[name.strip().lower()] = version.strip()
    return out


def _validate_requirements(
    report: ValidationReport,
    req_path: Path,
    baseline: dict[str, str] | None,
) -> None:
    if not req_path.is_file():
        report.results.append(
            CheckResult(
                "requirements.txt",
                "fail",
                "Missing requirements.txt.",
                code="MISSING_REQUIREMENTS",
                hint="Include the starter kit requirements.txt with pinned versions.",
            )
        )
        return
    text = req_path.read_text(encoding="utf-8")
    parsed = _parse_requirements(text)
    if not parsed:
        report.results.append(
            CheckResult(
                "requirements.txt",
                "fail",
                "No pinned packages found.",
                code="UNPINNED_REQUIREMENT",
                hint="Pin every dependency with package==version.",
            )
        )
        return
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "==" not in stripped:
            report.results.append(
                CheckResult(
                    "requirements.txt",
                    "fail",
                    f"Unpinned requirement: {stripped}",
                    code="UNPINNED_REQUIREMENT",
                    hint="Use package==version for every line.",
                )
            )
            return
    if baseline:
        for pkg, version in parsed.items():
            base_version = baseline.get(pkg)
            if base_version is None:
                normalized = pkg.replace("_", "-")
                if (
                    normalized not in ALLOWED_EXTRA_PACKAGES
                    and pkg not in ALLOWED_EXTRA_PACKAGES
                ):
                    report.results.append(
                        CheckResult(
                            "requirements.txt",
                            "fail",
                            f"Package not on allow-list: {pkg}",
                            code="DISALLOWED_REQUIREMENT",
                            hint="Remove extra packages or pick from the documented allow-list.",
                        )
                    )
                    return
            elif base_version != version:
                report.results.append(
                    CheckResult(
                        "requirements.txt",
                        "fail",
                        f"Changed pinned version for {pkg}.",
                        code="BASELINE_REQUIREMENT_CHANGED",
                        hint="Keep starter-kit package versions unless adding allow-listed extras.",
                    )
                )
                return


def _validate_sql_file(report: ValidationReport, path: Path, rel: str) -> None:
    if not path.is_file() or not path.read_text(encoding="utf-8").strip():
        report.results.append(
            CheckResult(
                rel,
                "fail",
                "SQL warm-up file missing or empty.",
                code="MISSING_SQL",
                hint=f"Add a single SQL statement to {rel}.",
            )
        )
        return
    text = path.read_text(encoding="utf-8").strip()
    if ";" in text.rstrip(";"):
        report.results.append(
            CheckResult(
                rel,
                "fail",
                "SQL file must contain a single statement.",
                code="SQL_MULTIPLE_STATEMENTS",
                hint="Use one statement per warm-up file.",
            )
        )


def _validate_results_json(
    report: ValidationReport,
    results_path: Path,
    confirmatory_ids: frozenset[str] | None,
    confirmatory_predictors: dict[str, str] | None = None,
) -> None:
    if not results_path.is_file():
        report.results.append(
            CheckResult(
                "submit/results.json",
                "fail",
                "Missing results.json.",
                code="MISSING_RESULTS",
                hint="Run the notebook final cell to write submit/results.json.",
            )
        )
        return
    try:
        payload = json.loads(results_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        report.results.append(
            CheckResult(
                "submit/results.json",
                "fail",
                f"Invalid JSON: {exc}",
                code="INVALID_RESULTS_JSON",
                hint="Fix JSON syntax in submit/results.json.",
            )
        )
        return
    if not isinstance(payload, dict):
        report.results.append(
            CheckResult(
                "submit/results.json",
                "fail",
                "results.json must be an object.",
                code="INVALID_RESULTS_JSON",
                hint="Use the schema from the starter kit.",
            )
        )
        return
    data_checks = payload.get("data_checks")
    if not isinstance(data_checks, dict):
        report.results.append(
            CheckResult(
                "submit/results.json",
                "fail",
                "Missing data_checks object.",
                code="INVALID_RESULTS_SCHEMA",
                hint="Include data_checks with dedup and sentinel fields.",
            )
        )
        return
    hypotheses = payload.get("hypotheses")
    if not isinstance(hypotheses, list):
        report.results.append(
            CheckResult(
                "submit/results.json",
                "fail",
                "Missing hypotheses array.",
                code="INVALID_RESULTS_SCHEMA",
                hint="Include one object per confirmatory hypothesis.",
            )
        )
        return
    seen: set[str] = set()
    for entry in hypotheses:
        if not isinstance(entry, dict):
            report.results.append(
                CheckResult(
                    "submit/results.json",
                    "fail",
                    "Each hypothesis entry must be an object.",
                    code="INVALID_RESULTS_SCHEMA",
                    hint="Follow the results.json template.",
                )
            )
            return
        missing = RESULTS_HYPOTHESIS_REQUIRED - set(entry.keys())
        if missing:
            report.results.append(
                CheckResult(
                    "submit/results.json",
                    "fail",
                    f"Missing fields: {', '.join(sorted(missing))}",
                    code="INVALID_RESULTS_SCHEMA",
                    hint="Each hypothesis needs id, estimate, ci, decision, and related fields.",
                )
            )
            return
        hid = str(entry["hypothesis_id"])
        seen.add(hid)
        if confirmatory_predictors is not None:
            locked_feature = confirmatory_predictors.get(hid)
            if locked_feature is not None and str(entry["feature"]) != locked_feature:
                report.results.append(
                    CheckResult(
                        "submit/results.json",
                        "fail",
                        f"{hid} tests {entry['feature']}, but you locked {locked_feature}.",
                        code="RESULTS_FEATURE_MISMATCH",
                        hint="Report each confirmatory hypothesis on the predictor you pre-registered.",
                    )
                )
                return
            if locked_feature is None and confirmatory_ids is not None:
                report.results.append(
                    CheckResult(
                        "submit/results.json",
                        "warn",
                        f"{hid} is not a locked confirmatory hypothesis.",
                        code="RESULTS_UNLOCKED_HYPOTHESIS",
                        hint="Only locked confirmatory hypotheses count as confirmatory; label others as exploratory.",
                    )
                )
        if entry.get("decision") not in DECISIONS:
            report.results.append(
                CheckResult(
                    "submit/results.json",
                    "fail",
                    f"Invalid decision for {hid}.",
                    code="INVALID_RESULTS_DECISION",
                    hint="Use keep, kill, or inconclusive.",
                )
            )
            return
    if confirmatory_ids is not None:
        missing_ids = confirmatory_ids - seen
        if missing_ids:
            report.results.append(
                CheckResult(
                    "submit/results.json",
                    "fail",
                    f"Missing hypothesis ids: {', '.join(sorted(missing_ids))}",
                    code="MISSING_HYPOTHESIS_IN_RESULTS",
                    hint="Include one results.json entry per locked confirmatory hypothesis.",
                )
            )


def _as_naive_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def _validate_experiment_log(
    report: ValidationReport,
    log_path: Path,
    confirmatory_ids: frozenset[str] | None,
    hypotheses_locked_at: datetime | None = None,
) -> None:
    if not log_path.is_file():
        report.results.append(
            CheckResult(
                "submit/experiment_log.csv",
                "fail",
                "Missing experiment log.",
                code="MISSING_EXPERIMENT_LOG",
                hint="Log every experiment in submit/experiment_log.csv.",
            )
        )
        return
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        report.results.append(
            CheckResult(
                "submit/experiment_log.csv",
                "fail",
                "Empty experiment log.",
                code="MISSING_EXPERIMENT_LOG",
                hint="Add at least one row per confirmatory hypothesis.",
            )
        )
        return
    header = [c.strip() for c in lines[0].split(",")]
    if header != EXPERIMENT_LOG_COLUMNS:
        report.results.append(
            CheckResult(
                "submit/experiment_log.csv",
                "fail",
                "Header does not match required schema.",
                code="INVALID_LOG_HEADER",
                hint="Use the exact column order from the starter kit.",
            )
        )
        return
    reader = csv.DictReader(io.StringIO("\n".join(lines)))
    rows = list(reader)
    confirmatory_rows = 0
    for row in rows:
        try:
            logged_at = datetime.fromisoformat(row["logged_at"].replace("Z", "+00:00"))
        except ValueError:
            report.results.append(
                CheckResult(
                    "submit/experiment_log.csv",
                    "fail",
                    f"Invalid logged_at: {row.get('logged_at')}",
                    code="INVALID_LOG_TIMESTAMP",
                    hint="Use UTC ISO-8601 timestamps.",
                )
            )
            return
        if row["test_type"] not in TEST_TYPES:
            report.results.append(
                CheckResult(
                    "submit/experiment_log.csv",
                    "fail",
                    f"Invalid test_type: {row.get('test_type')}",
                    code="INVALID_LOG_ENUM",
                    hint="Use cheap, full, or robustness.",
                )
            )
            return
        if (
            hypotheses_locked_at is not None
            and row["split"] == "holdout"
            and _as_naive_utc(logged_at) < hypotheses_locked_at
        ):
            report.results.append(
                CheckResult(
                    "submit/experiment_log.csv",
                    "fail",
                    f"Holdout experiment {row.get('exp_id')} is logged before hypotheses were locked.",
                    code="HOLDOUT_LOGGED_BEFORE_LOCK",
                    hint="Holdout tests must be logged after you lock your pre-registration.",
                )
            )
            return
        if row["split"] not in SPLITS:
            report.results.append(
                CheckResult(
                    "submit/experiment_log.csv",
                    "fail",
                    f"Invalid split: {row.get('split')}",
                    code="INVALID_LOG_ENUM",
                    hint="Use discovery or holdout.",
                )
            )
            return
        if row["decision"] not in DECISIONS:
            report.results.append(
                CheckResult(
                    "submit/experiment_log.csv",
                    "fail",
                    f"Invalid decision: {row.get('decision')}",
                    code="INVALID_LOG_ENUM",
                    hint="Use keep, kill, or inconclusive.",
                )
            )
            return
        if row["kill_reason"] and row["kill_reason"] not in KILL_REASONS:
            report.results.append(
                CheckResult(
                    "submit/experiment_log.csv",
                    "fail",
                    f"Invalid kill_reason: {row.get('kill_reason')}",
                    code="INVALID_LOG_ENUM",
                    hint="Use the controlled kill_reason vocabulary.",
                )
            )
            return
        hid = row["hypothesis_id"]
        if confirmatory_ids and hid in confirmatory_ids:
            confirmatory_rows += 1
    if confirmatory_ids and confirmatory_rows < len(confirmatory_ids):
        report.results.append(
            CheckResult(
                "submit/experiment_log.csv",
                "warn",
                "Fewer log rows than confirmatory hypotheses.",
                code="LOG_ROW_COUNT_WARN",
                hint="Log at least one row per confirmatory hypothesis.",
            )
        )


def _validate_ai_usage(report: ValidationReport, ai_path: Path) -> None:
    if not ai_path.is_file() or not ai_path.read_text(encoding="utf-8").strip():
        report.results.append(
            CheckResult(
                "submit/ai_usage.md",
                "fail",
                "AI usage log missing or empty.",
                code="MISSING_AI_USAGE",
                hint='State "no AI used" or list tools you used.',
            )
        )
        return
    text = ai_path.read_text(encoding="utf-8").strip().lower()
    if "no ai" in text:
        return
    if len(text) < 20:
        report.results.append(
            CheckResult(
                "submit/ai_usage.md",
                "fail",
                "Describe AI tools used or state that none were used.",
                code="INVALID_AI_USAGE",
                hint="Add tool names and how you used them.",
            )
        )


def _validate_helpers(report: ValidationReport, helpers: Path) -> None:
    if not helpers.is_file():
        report.results.append(
            CheckResult(
                "src/takehome/helpers.py",
                "fail",
                "Missing helpers module.",
                code="MISSING_HELPERS",
                hint="Include src/takehome/helpers.py from the kit.",
            )
        )
        return
    source = helpers.read_text(encoding="utf-8")
    try:
        compile(source, str(helpers), "exec")
    except SyntaxError as exc:
        report.results.append(
            CheckResult(
                "src/takehome/helpers.py",
                "fail",
                f"Syntax error: {exc}",
                code="HELPERS_SYNTAX",
                hint="Fix Python syntax errors before submitting.",
            )
        )
        return
    try:
        ast.parse(source)
    except SyntaxError as exc:
        report.results.append(
            CheckResult(
                "src/takehome/helpers.py",
                "fail",
                f"Syntax error: {exc}",
                code="HELPERS_SYNTAX",
                hint="Fix Python syntax errors before submitting.",
            )
        )


def validate_tree(
    root: Path,
    *,
    report_word_limit: int = 600,
    confirmatory_hypothesis_ids: frozenset[str] | None = None,
    confirmatory_set_sha256: str | None = None,
    baseline_requirements: dict[str, str] | None = None,
    confirmatory_predictors: dict[str, str] | None = None,
    hypotheses_locked_at: datetime | None = None,
) -> ValidationReport:
    report = ValidationReport()
    root = root.resolve()

    def add(
        path: str,
        status: Status,
        message: str,
        *,
        code: str = "",
        hint: str = "",
    ) -> None:
        report.results.append(
            CheckResult(path=path, status=status, message=message, code=code, hint=hint)
        )

    tests_dir = root / "tests"
    if not tests_dir.is_dir():
        add(
            "tests/",
            "fail",
            "Missing tests directory.",
            code="MISSING_TESTS",
            hint="Include the tests/ folder from the starter kit.",
        )

    data_dir = root / "data"
    if data_dir.is_dir() and any(data_dir.iterdir()):
        add(
            "data/",
            "warn",
            "data/ will be stripped; we mount the official pack.",
            code="DATA_DIR_PRESENT",
            hint="Remove bundled data files from your zip before submitting.",
        )

    nb = root / "notebooks" / "analysis.ipynb"
    if not nb.is_file():
        add(
            "notebooks/analysis.ipynb",
            "fail",
            "Missing required notebook.",
            code="MISSING_NOTEBOOK",
            hint="Include notebooks/analysis.ipynb.",
        )
    else:
        try:
            nbformat.read(nb, as_version=4)
            found = _notebook_headings(nb)
            for required in NOTEBOOK_HEADINGS:
                if not any(required.lower() in h.lower() for h in found):
                    add(
                        "notebooks/analysis.ipynb",
                        "fail",
                        f"Missing section heading: {required}",
                        code="NOTEBOOK_MISSING_HEADING",
                        hint=f"Add a markdown heading for {required}.",
                    )
        except Exception as exc:  # noqa: BLE001
            add(
                "notebooks/analysis.ipynb",
                "fail",
                f"Invalid notebook: {exc}",
                code="INVALID_NOTEBOOK",
                hint="Ensure the notebook is valid nbformat v4.",
            )

    _validate_helpers(report, root / "src" / "takehome" / "helpers.py")

    for q in ("q1.sql", "q2.sql", "q3.sql"):
        _validate_sql_file(report, root / "queries" / q, f"queries/{q}")

    report_md = root / "submit" / "report.md"
    if not report_md.is_file():
        add(
            "submit/report.md",
            "fail",
            "Missing report.",
            code="MISSING_REPORT",
            hint="Add submit/report.md with required headings.",
        )
    else:
        text = report_md.read_text(encoding="utf-8")
        for heading in REPORT_HEADINGS:
            if not _heading_present(text, heading):
                add(
                    "submit/report.md",
                    "fail",
                    f"Missing heading: {heading}",
                    code="REPORT_MISSING_HEADING",
                    hint=f"Add a markdown heading: {heading}",
                )
        words = len(re.findall(r"\w+", text))
        if words > report_word_limit:
            add(
                "submit/report.md",
                "fail",
                f"Report is {words} words; limit is {report_word_limit}.",
                code="REPORT_WORD_LIMIT",
                hint=f"Shorten the report to {report_word_limit} words or fewer.",
            )
        elif words > int(report_word_limit * 0.95):
            add(
                "submit/report.md",
                "warn",
                f"Report is {words} words; limit is {report_word_limit}.",
                code="REPORT_WORD_LIMIT_WARN",
                hint=f"Stay under {report_word_limit} words.",
            )
        embedded = _count_embedded_images(text)
        if embedded > MAX_EMBEDDED_IMAGE_BYTES:
            add(
                "submit/report.md",
                "fail",
                "Embedded images exceed 1 MB.",
                code="REPORT_IMAGE_TOO_LARGE",
                hint="Link images externally or shrink embedded assets.",
            )

    _validate_experiment_log(
        report,
        root / "submit" / "experiment_log.csv",
        confirmatory_hypothesis_ids,
        hypotheses_locked_at,
    )
    _validate_results_json(
        report,
        root / "submit" / "results.json",
        confirmatory_hypothesis_ids,
        confirmatory_predictors,
    )
    _validate_ai_usage(report, root / "submit" / "ai_usage.md")
    _validate_requirements(report, root / "requirements.txt", baseline_requirements)

    if report.ok and not any(r.path == "." for r in report.results):
        add(".", "pass", "Validation passed.", code="OK")
    return report


def render_validate_cli_script() -> str:
    """Standalone script embedded in the candidate kit."""
    return '''#!/usr/bin/env python3
"""Validate take-home submission layout (generated from submission_schema.py)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from takehome.validation import validate_tree  # noqa: E402


def main() -> int:
    report = validate_tree(ROOT)
    print(json.dumps(report.to_dict(), indent=2))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
'''
