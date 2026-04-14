#!/usr/bin/env python3
"""CASP17 batch submission tool with strict local validation.

This tool is designed for CASP17 server-group submissions and follows the
rules published at:
https://predictioncenter.org/casp17/index.cgi\?page\=format

Default behavior:
- Scans one directory for .pdb files.
- Validates each file in strict mode before upload.
- Submits via HTTPS POST using fields: email, prediction_file.
- Skips already-successful model keys unless --force-resubmit is set.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


DEFAULT_ENDPOINT = "https://predictioncenter.org/casp17/submit"
DEFAULT_CHECKPOINT = "checkpoint.json"
DEFAULT_REPORT = "submission_report.json"
DEFAULT_GLOB = "*.pdb"
DEFAULT_MAX_QA_MODELS = 1
DEFAULT_MAX_LG_MODELS = 1


class ValidationError(Exception):
    pass


@dataclass
class Issue:
    severity: str  # ERROR or WARN
    code: str
    message: str
    line: Optional[int] = None

    def as_dict(self) -> Dict[str, object]:
        out: Dict[str, object] = {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
        }
        if self.line is not None:
            out["line"] = self.line
        return out


@dataclass
class ModelBlock:
    index: int
    start_line: int
    end_line: int
    lines: List[str]


@dataclass
class ParsedFile:
    path: Path
    lines: List[str]
    pfrmat: str
    target: str
    author: str
    model_blocks: List[ModelBlock] = field(default_factory=list)


@dataclass
class ValidationResult:
    parsed: Optional[ParsedFile]
    issues: List[Issue] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not any(i.severity == "ERROR" for i in self.issues)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_text_ascii(path: Path) -> List[str]:
    data = path.read_bytes()
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValidationError(f"Non-ASCII content: {exc}")
    return text.splitlines()


def parse_model_blocks(lines: List[str], issues: List[Issue]) -> List[ModelBlock]:
    blocks: List[ModelBlock] = []
    start_idx: Optional[int] = None
    model_no: Optional[int] = None

    for i, line in enumerate(lines, start=1):
        raw = line.strip()
        if raw.startswith("MODEL"):
            m = re.match(r"^MODEL\s+(\d+)\s*$", raw)
            if not m:
                issues.append(Issue("ERROR", "MODEL_FORMAT", "MODEL line must be 'MODEL <int>'", i))
                continue
            if start_idx is not None:
                issues.append(Issue("ERROR", "MODEL_NESTED", "Encountered MODEL before END", i))
                continue
            start_idx = i
            model_no = int(m.group(1))
        elif raw == "END":
            if start_idx is None:
                issues.append(Issue("ERROR", "END_WITHOUT_MODEL", "END appears outside MODEL block", i))
                continue
            block_lines = lines[start_idx - 1 : i]
            blocks.append(ModelBlock(index=model_no or -1, start_line=start_idx, end_line=i, lines=block_lines))
            start_idx = None
            model_no = None

    if start_idx is not None:
        issues.append(Issue("ERROR", "MODEL_UNCLOSED", "MODEL block missing END", start_idx))

    if not blocks:
        issues.append(Issue("ERROR", "NO_MODEL", "At least one MODEL...END block is required"))

    return blocks


def parse_header(lines: List[str], issues: List[Issue]) -> Tuple[str, str, str]:
    if len(lines) < 3:
        issues.append(Issue("ERROR", "HEADER_TOO_SHORT", "File must contain at least 3 header lines"))
        return "", "", ""

    first = lines[0].strip()
    second = lines[1].strip()
    third = lines[2].strip()

    if not first.startswith("PFRMAT "):
        issues.append(Issue("ERROR", "PFRMAT_LINE1", "Line 1 must start with PFRMAT", 1))
    if not second.startswith("TARGET "):
        issues.append(Issue("ERROR", "TARGET_LINE2", "Line 2 must start with TARGET", 2))
    if not third.startswith("AUTHOR "):
        issues.append(Issue("ERROR", "AUTHOR_LINE3", "Line 3 must start with AUTHOR", 3))

    pfrmat_vals = []
    target_vals = []
    author_vals = []
    for i, line in enumerate(lines, start=1):
        s = line.strip()
        if s.startswith("PFRMAT "):
            pfrmat_vals.append((i, s.split(None, 1)[1].strip()))
        elif s.startswith("TARGET "):
            target_vals.append((i, s.split(None, 1)[1].strip()))
        elif s.startswith("AUTHOR "):
            author_vals.append((i, s.split(None, 1)[1].strip()))

    if len(pfrmat_vals) != 1:
        issues.append(Issue("ERROR", "PFRMAT_COUNT", "Exactly one PFRMAT record is required"))
    if len(target_vals) != 1:
        issues.append(Issue("ERROR", "TARGET_COUNT", "Exactly one TARGET record is required"))
    if len(author_vals) != 1:
        issues.append(Issue("ERROR", "AUTHOR_COUNT", "Exactly one AUTHOR record is required"))

    pfrmat = pfrmat_vals[0][1] if pfrmat_vals else ""
    target = target_vals[0][1] if target_vals else ""
    author = author_vals[0][1] if author_vals else ""

    if pfrmat not in {"TS", "QA", "LG"}:
        issues.append(Issue("ERROR", "PFRMAT_VALUE", "PFRMAT must be TS, QA, or LG"))

    if target and not re.fullmatch(r"[THRML]\d{4}", target):
        issues.append(Issue("ERROR", "TARGET_FORMAT", "TARGET must match [T|H|R|M|L]xxxx"))

    reg_code = re.fullmatch(r"\d{4}-\d{4}-\d{4}", author)
    if not reg_code:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", author):
            issues.append(Issue("ERROR", "AUTHOR_FORMAT", "AUTHOR must be registration code or server group name"))

    return pfrmat, target, author


def ensure_method_before_first_model(lines: List[str], issues: List[Issue]) -> None:
    first_model_line: Optional[int] = None
    for i, line in enumerate(lines, start=1):
        if line.strip().startswith("MODEL"):
            first_model_line = i
            break
    if first_model_line is None:
        return

    method_ok = False
    for line in lines[: first_model_line - 1]:
        if line.strip().startswith("METHOD"):
            method_ok = True
            break
    if not method_ok:
        issues.append(Issue("ERROR", "METHOD_MISSING", "At least one METHOD record is required before first MODEL"))


def validate_line_lengths(lines: List[str], issues: List[Issue]) -> None:
    for i, line in enumerate(lines, start=1):
        if len(line) > 80:
            issues.append(Issue("ERROR", "LINE_TOO_LONG", "PDB/record line exceeds 80 columns", i))


def parse_atom_fields(line: str) -> Optional[Tuple[str, str, str, float, float]]:
    if not (line.startswith("ATOM  ") or line.startswith("HETATM")):
        return None
    if len(line) < 66:
        return None
    chain = line[21].strip() or "_"
    resseq = line[22:26].strip()
    icode = line[26].strip() or "_"
    try:
        occ = float(line[54:60].strip())
        bfac = float(line[60:66].strip())
    except ValueError:
        return None
    return chain, resseq, icode, occ, bfac


def _is_record_keyword(s: str) -> bool:
    if not s:
        return False
    head = s.split(None, 1)[0]
    known = {
        "PFRMAT",
        "TARGET",
        "AUTHOR",
        "METHOD",
        "MODEL",
        "END",
        "PARENT",
        "TER",
        "SCORE",
        "QSCORE",
        "STOICH",
        "LIGAND",
        "LSCORE",
        "AFFNTY",
        "REMARK",
        "ATOM",
        "HETATM",
    }
    return head in known


def validate_common_model_block(block: ModelBlock, issues: List[Issue]) -> None:
    closed_residues = set()
    current_residue: Optional[Tuple[str, str, str]] = None
    residue_bfactors: Dict[Tuple[str, str, str], List[float]] = {}

    for offset, line in enumerate(block.lines, start=block.start_line):
        is_atom = line.startswith("ATOM  ") or line.startswith("HETATM")
        if is_atom and len(line) != 80:
            issues.append(Issue("ERROR", "ATOM_WIDTH", "ATOM/HETATM lines must be exactly 80 columns", offset))

        parsed = parse_atom_fields(line)
        if is_atom and len(line) < 66:
            issues.append(Issue("ERROR", "ATOM_SHORT", "ATOM/HETATM line too short for occupancy/B-factor fields", offset))
            continue
        if is_atom and parsed is None:
            issues.append(Issue("ERROR", "ATOM_PARSE", "Unable to parse occupancy/B-factor in ATOM/HETATM line", offset))
            continue
        if parsed is None:
            continue
        chain, resseq, icode, occ, bfac = parsed

        if not resseq:
            issues.append(Issue("ERROR", "RESSEQ_MISSING", "ATOM/HETATM residue number is missing", offset))

        key = (chain, resseq, icode)
        if current_residue is None:
            current_residue = key
        elif key != current_residue:
            closed_residues.add(current_residue)
            if key in closed_residues:
                issues.append(Issue("ERROR", "DUPLICATE_RES", "Duplicate non-contiguous residue identifier in one MODEL", offset))
            current_residue = key

        if not (occ == 0.0 or 0.01 <= occ <= 1.00):
            issues.append(Issue("ERROR", "OCC_RANGE", "Occupancy must be 0.00 or within [0.01, 1.00]", offset))

        if not (0.0 <= bfac <= 100.0):
            issues.append(Issue("ERROR", "BFACTOR_RANGE", "B-factor must be within [0, 100]", offset))

        residue_bfactors.setdefault(key, []).append(bfac)

    if residue_bfactors:
        residue_avg = []
        for vals in residue_bfactors.values():
            residue_avg.append(sum(vals) / float(len(vals)))
        if max(residue_avg) - min(residue_avg) < 1e-9:
            issues.append(Issue("ERROR", "BFACTOR_UNIFORM", "All residues have identical B-factor in one MODEL"))


def validate_parent_line(parent_line: str) -> bool:
    s = parent_line.strip()
    if not s.startswith("PARENT "):
        return False
    payload = s.split(None, 1)[1].strip()
    if payload == "N/A":
        return True

    tokens = payload.split()
    if len(tokens) > 5:
        return False

    pat = re.compile(r"^[0-9][A-Za-z0-9]{3}(?:_[A-Za-z0-9])?$")
    return all(bool(pat.fullmatch(t)) for t in tokens)


def validate_ts(parsed: ParsedFile, issues: List[Issue]) -> None:
    blocks = parsed.model_blocks
    if len(blocks) > 6:
        issues.append(Issue("ERROR", "TS_MODEL_COUNT", "TS submission may include at most 6 models"))

    seen_idx = set()
    for b in blocks:
        if not (1 <= b.index <= 6):
            issues.append(Issue("ERROR", "TS_MODEL_INDEX", f"TS model index out of range: {b.index}", b.start_line))
        if b.index in seen_idx:
            issues.append(Issue("ERROR", "TS_MODEL_DUP", f"Duplicate TS model index: {b.index}", b.start_line))
        seen_idx.add(b.index)

        has_parent = False
        has_ter = False
        has_atom = False
        for rel_idx, line in enumerate(b.lines):
            j = b.start_line + rel_idx
            s = line.strip()
            if s.startswith("PARENT"):
                has_parent = True
                if s == "PARENT":
                    payload = ""
                    k = rel_idx + 1
                    while k < len(b.lines):
                        nxt = b.lines[k].strip()
                        if not nxt:
                            k += 1
                            continue
                        if _is_record_keyword(nxt):
                            break
                        payload = nxt
                        break
                    if not payload or not validate_parent_line(f"PARENT {payload}"):
                        issues.append(Issue("ERROR", "PARENT_FORMAT", "Invalid or missing PARENT payload", j))
                elif not validate_parent_line(s):
                    issues.append(Issue("ERROR", "PARENT_FORMAT", "Invalid PARENT format", j))
            if s == "TER":
                has_ter = True
            if line.startswith("ATOM  ") or line.startswith("HETATM"):
                has_atom = True

        if not has_parent:
            issues.append(Issue("ERROR", "PARENT_REQUIRED", "TS model requires PARENT record", b.start_line))
        if not has_ter:
            issues.append(Issue("ERROR", "TER_REQUIRED", "TS model requires TER record", b.start_line))
        if not has_atom:
            issues.append(Issue("ERROR", "TS_NO_ATOM", "TS model has no ATOM/HETATM records", b.start_line))

        if b.index == 6:
            issues.append(Issue("WARN", "MODEL6_NOTICE", "Model 6 is accepted only for organizer-provided MSA cases", b.start_line))


def validate_qa(parsed: ParsedFile, issues: List[Issue], max_models: int) -> None:
    blocks = parsed.model_blocks
    if max_models < 1:
        issues.append(Issue("ERROR", "QA_CFG", "max_qa_models must be >= 1"))
        return

    if len(blocks) < 1:
        issues.append(Issue("ERROR", "QA_MODEL_COUNT", "QA file must contain at least one MODEL block"))
    if len(blocks) > max_models:
        issues.append(Issue("ERROR", "QA_MODEL_COUNT", f"QA file must contain at most {max_models} MODEL block(s)"))

    seen_idx = set()
    for b in blocks:
        if not (1 <= b.index <= max_models):
            issues.append(Issue("ERROR", "QA_MODEL_INDEX", f"QA model index must be in [1, {max_models}]", b.start_line))
        if b.index in seen_idx:
            issues.append(Issue("ERROR", "QA_MODEL_DUP", f"Duplicate QA model index: {b.index}", b.start_line))
        seen_idx.add(b.index)

        nonempty_data = 0
        for j, line in enumerate(b.lines[1:-1], start=b.start_line + 1):
            s = line.strip()
            if not s or s.startswith("REMARK"):
                continue
            nonempty_data += 1
            m = re.match(r"^(\S+)\s+([0-9]*\.?[0-9]+)(?:\s+(.*))?$", s)
            if not m:
                issues.append(Issue("ERROR", "QA_LINE_FORMAT", "Invalid QA data line format", j))
                continue
            overall = float(m.group(2))
            if not (0.0 <= overall <= 1.0):
                issues.append(Issue("ERROR", "QA_SCORE_RANGE", "QA overall score must be in [0,1]", j))
            iface_part = (m.group(3) or "").strip()
            if iface_part:
                items = [x.strip() for x in iface_part.split(",") if x.strip()]
                for it in items:
                    mm = re.match(r"^([A-Za-z0-9]{2}):([0-9]*\.?[0-9]+)$", it)
                    if not mm:
                        issues.append(Issue("ERROR", "QA_IFACE_FORMAT", "Invalid QA interface score token", j))
                        continue
                    val = float(mm.group(2))
                    if not (0.0 <= val <= 1.0):
                        issues.append(Issue("ERROR", "QA_IFACE_RANGE", "QA interface score must be in [0,1]", j))

        if nonempty_data == 0:
            issues.append(Issue("ERROR", "QA_EMPTY", "QA MODEL block must include score lines", b.start_line))


def validate_lg(parsed: ParsedFile, issues: List[Issue], max_models: int) -> None:
    blocks = parsed.model_blocks
    if max_models < 1:
        issues.append(Issue("ERROR", "LG_CFG", "max_lg_models must be >= 1"))
        return

    if len(blocks) < 1:
        issues.append(Issue("ERROR", "LG_MODEL_COUNT", "LG file must contain at least one MODEL block"))
    if len(blocks) > max_models:
        issues.append(Issue("ERROR", "LG_MODEL_COUNT", f"LG file must contain at most {max_models} MODEL block(s)"))

    seen_idx = set()
    for b in blocks:
        if not (1 <= b.index <= max_models):
            issues.append(Issue("ERROR", "LG_MODEL_INDEX", f"LG model index must be in [1, {max_models}]", b.start_line))
        if b.index in seen_idx:
            issues.append(Issue("ERROR", "LG_MODEL_DUP", f"Duplicate LG model index: {b.index}", b.start_line))
        seen_idx.add(b.index)

        ligand_open = False
        ligand_blocks = 0
        affinity_records = 0
        atom_records = 0

        for j, line in enumerate(b.lines[1:-1], start=b.start_line + 1):
            s = line.strip()
            if line.startswith("ATOM  ") or line.startswith("HETATM"):
                atom_records += 1

            if s.startswith("LIGAND"):
                if ligand_open:
                    issues.append(Issue("ERROR", "LG_MDL_UNCLOSED", "Previous LIGAND block missing 'M  END'", j))
                ligand_open = True
                ligand_blocks += 1
                parts = s.split()
                if len(parts) < 3:
                    issues.append(Issue("ERROR", "LIGAND_FORMAT", "LIGAND must include id and name", j))
                elif not re.fullmatch(r"\d{1,3}", parts[1]):
                    issues.append(Issue("ERROR", "LIGAND_ID", "LIGAND id must be numeric", j))

            elif s == "M  END":
                if not ligand_open:
                    issues.append(Issue("ERROR", "LG_MEND_ORPHAN", "M  END appears outside ligand block", j))
                ligand_open = False

            elif s.startswith("LSCORE"):
                mm = re.match(r"^LSCORE\s+([0-9]*\.?[0-9]+)\s*$", s)
                if not mm:
                    issues.append(Issue("ERROR", "LSCORE_FORMAT", "LSCORE format invalid", j))
                else:
                    score = float(mm.group(1))
                    if not (0.0 <= score <= 1.0):
                        issues.append(Issue("ERROR", "LSCORE_RANGE", "LSCORE must be in [0,1]", j))

            elif s.startswith("AFFNTY"):
                affinity_records += 1
                mm = re.match(r"^AFFNTY\s+([0-9]*\.?[0-9]+)\s+(aa|ra|lr)\s*$", s)
                if not mm:
                    issues.append(Issue("ERROR", "AFFNTY_FORMAT", "AFFNTY must be: AFFNTY <value> <aa|ra|lr>", j))
                else:
                    val = float(mm.group(1))
                    typ = mm.group(2)
                    if typ == "lr" and int(val) != val:
                        issues.append(Issue("ERROR", "AFFNTY_LR_INT", "AFFNTY lr value must be integer", j))
                    if val < 0:
                        issues.append(Issue("ERROR", "AFFNTY_NEG", "AFFNTY value must be non-negative", j))

        if ligand_open:
            issues.append(Issue("ERROR", "LG_MDL_UNCLOSED", "LIGAND block missing final 'M  END'", b.end_line))

        if atom_records == 0 and ligand_blocks == 0 and affinity_records == 0:
            issues.append(Issue("ERROR", "LG_EMPTY", "LG model must contain receptor coords, ligand block, and/or AFFNTY", b.start_line))


def validate_file(path: Path, max_qa_models: int = DEFAULT_MAX_QA_MODELS, max_lg_models: int = DEFAULT_MAX_LG_MODELS) -> ValidationResult:
    issues: List[Issue] = []

    try:
        lines = load_text_ascii(path)
    except ValidationError as exc:
        return ValidationResult(parsed=None, issues=[Issue("ERROR", "ASCII", str(exc))])

    validate_line_lengths(lines, issues)
    ensure_method_before_first_model(lines, issues)

    pfrmat, target, author = parse_header(lines, issues)
    blocks = parse_model_blocks(lines, issues)

    parsed = ParsedFile(path=path, lines=lines, pfrmat=pfrmat, target=target, author=author, model_blocks=blocks)

    for b in blocks:
        validate_common_model_block(b, issues)

    if pfrmat == "TS":
        validate_ts(parsed, issues)
    elif pfrmat == "QA":
        validate_qa(parsed, issues, max_models=max_qa_models)
    elif pfrmat == "LG":
        validate_lg(parsed, issues, max_models=max_lg_models)

    return ValidationResult(parsed=parsed, issues=issues)


def load_json(path: Path, fallback: Dict[str, object]) -> Dict[str, object]:
    if not path.exists():
        return fallback
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def extract_accession(text: str) -> Optional[str]:
    m = re.search(r"\b([THRML]\d{4}(?:TS|QA|LG)\d{3}_\d+)\b", text)
    return m.group(1) if m else None


def should_retry_http(code: int) -> bool:
    return 500 <= code <= 599


def submit_once(path: Path, endpoint: str, email: str, timeout_sec: int) -> Tuple[int, str]:
    try:
        import requests  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise RuntimeError("The 'requests' package is required for uploads. Install via: pip install requests") from exc

    with path.open("rb") as fp:
        files = {"prediction_file": (path.name, fp, "text/plain")}
        data = {"email": email}
        response = requests.post(endpoint, data=data, files=files, timeout=timeout_sec)
    return response.status_code, response.text


def submit_with_retry(
    path: Path,
    endpoint: str,
    email: str,
    timeout_sec: int,
    max_retries: int,
) -> Dict[str, object]:
    backoff = [5, 15, 60, 120]
    attempts = 0
    final_code = -1
    final_text = ""
    final_error = ""

    for attempt in range(max_retries + 1):
        attempts = attempt + 1
        try:
            code, text = submit_once(path, endpoint, email, timeout_sec)
            final_code = code
            final_text = text
            if should_retry_http(code) and attempt < max_retries:
                time.sleep(backoff[min(attempt, len(backoff) - 1)])
                continue
            break
        except Exception as exc:
            final_error = str(exc)
            if attempt < max_retries:
                time.sleep(backoff[min(attempt, len(backoff) - 1)])
                continue
            break

    accession = extract_accession(final_text)
    lower = final_text.lower()

    success = False
    if accession:
        success = True
    elif final_code == 200 and "error" not in lower and "reject" not in lower:
        success = True

    return {
        "success": success,
        "attempts": attempts,
        "http_code": final_code,
        "response_text": final_text,
        "error": final_error,
        "accession": accession,
    }


def discover_files(input_dir: Path, pattern: str) -> List[Path]:
    files = sorted([p for p in input_dir.glob(pattern) if p.is_file()])
    return files


def email_domain_ok(email: str, allowed_domain: Optional[str]) -> bool:
    if allowed_domain is None:
        return True
    if "@" not in email:
        return False
    domain = email.split("@", 1)[1].lower()
    allow = allowed_domain.lower()
    return domain == allow or domain.endswith("." + allow)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CASP17 strict batch submitter")
    parser.add_argument("--config", type=str, default=None, help="JSON config file")
    parser.add_argument("--input-dir", type=str, default=None, help="Directory containing submission files")
    parser.add_argument("--glob", type=str, default=None, help="Glob pattern for files (default: *.pdb)")
    parser.add_argument("--email", type=str, default=None, help="Submission email address")
    parser.add_argument("--allowed-domain", type=str, default=None, help="Optional email domain guard")
    parser.add_argument("--endpoint", type=str, default=None, help="Submission endpoint")
    parser.add_argument("--timeout", type=int, default=None, help="HTTP timeout seconds")
    parser.add_argument("--max-retries", type=int, default=None, help="Retry count for network/5xx")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint JSON path")
    parser.add_argument("--report", type=str, default=None, help="Report JSON path")
    parser.add_argument("--target-filter", type=str, default=None, help="Comma-separated target list")
    parser.add_argument("--retry-failed-only", action="store_true", help="Only process files previously marked as FAILED")
    parser.add_argument("--force-resubmit", action="store_true", help="Allow overwriting previously successful target/category/model keys")
    parser.add_argument("--dry-run", action="store_true", help="Validate only, do not upload")
    parser.add_argument("--verbose", action="store_true", help="Print detailed per-file diagnostics")
    parser.add_argument("--max-qa-models", type=int, default=None, help="Max MODEL blocks allowed for QA (default: 1)")
    parser.add_argument("--max-lg-models", type=int, default=None, help="Max MODEL blocks allowed for LG (default: 1)")
    return parser.parse_args()


def merge_config(args: argparse.Namespace) -> Dict[str, object]:
    cfg: Dict[str, object] = {}
    if args.config:
        cfg = load_json(Path(args.config), {})

    def pick(name: str, default: object) -> object:
        val = getattr(args, name)
        if val is not None:
            return val
        return cfg.get(name.replace("_", "-"), cfg.get(name, default))

    merged: Dict[str, object] = {
        "input_dir": pick("input_dir", "."),
        "glob": pick("glob", DEFAULT_GLOB),
        "email": pick("email", ""),
        "allowed_domain": pick("allowed_domain", None),
        "endpoint": pick("endpoint", DEFAULT_ENDPOINT),
        "timeout": int(pick("timeout", 120)),
        "max_retries": int(pick("max_retries", 3)),
        "checkpoint": pick("checkpoint", DEFAULT_CHECKPOINT),
        "report": pick("report", DEFAULT_REPORT),
        "target_filter": pick("target_filter", ""),
        "retry_failed_only": bool(args.retry_failed_only or cfg.get("retry_failed_only", False)),
        "force_resubmit": bool(args.force_resubmit or cfg.get("force_resubmit", False)),
        "dry_run": bool(args.dry_run or cfg.get("dry_run", False)),
        "verbose": bool(args.verbose or cfg.get("verbose", False)),
        "max_qa_models": int(pick("max_qa_models", DEFAULT_MAX_QA_MODELS)),
        "max_lg_models": int(pick("max_lg_models", DEFAULT_MAX_LG_MODELS)),
    }
    return merged


def build_model_keys(target: str, category: str, model_indices: Iterable[int]) -> List[str]:
    return [f"{target}|{category}|{i}" for i in sorted(set(model_indices))]


def main() -> int:
    args = parse_args()
    cfg = merge_config(args)

    email = str(cfg["email"]).strip()
    if not email:
        print("ERROR: --email is required (or set in config).", file=sys.stderr)
        return 2

    if not email_domain_ok(email, cfg.get("allowed_domain")):
        print("ERROR: email domain does not match --allowed-domain constraint.", file=sys.stderr)
        return 2

    input_dir = Path(str(cfg["input_dir"])).resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"ERROR: input directory does not exist: {input_dir}", file=sys.stderr)
        return 2

    checkpoint_path = Path(str(cfg["checkpoint"])).resolve()
    report_path = Path(str(cfg["report"])).resolve()

    checkpoint = load_json(
        checkpoint_path,
        {"version": 1, "updated_at": None, "submissions": [], "successful_model_keys": {}, "failed_hashes": {}},
    )

    files = discover_files(input_dir, str(cfg["glob"]))
    if not files:
        print("No files found for pattern.")
        return 0

    target_filter = {x.strip() for x in str(cfg["target_filter"]).split(",") if x.strip()}

    report: Dict[str, object] = {
        "created_at": utc_now(),
        "config": {
            "input_dir": str(input_dir),
            "glob": cfg["glob"],
            "endpoint": cfg["endpoint"],
            "dry_run": cfg["dry_run"],
            "retry_failed_only": cfg["retry_failed_only"],
            "force_resubmit": cfg["force_resubmit"],
            "max_qa_models": cfg["max_qa_models"],
            "max_lg_models": cfg["max_lg_models"],
        },
        "results": [],
        "summary": {},
    }

    successful = 0
    failed = 0
    skipped = 0
    validation_failed = 0

    failed_hashes: Dict[str, str] = checkpoint.get("failed_hashes", {})
    successful_keys: Dict[str, Dict[str, object]] = checkpoint.get("successful_model_keys", {})
    submissions: List[Dict[str, object]] = checkpoint.get("submissions", [])

    for path in files:
        file_hash = sha256_file(path)
        entry: Dict[str, object] = {
            "file": str(path),
            "hash": file_hash,
            "started_at": utc_now(),
            "status": "UNKNOWN",
            "issues": [],
        }

        if cfg["retry_failed_only"] and file_hash not in failed_hashes:
            entry["status"] = "SKIPPED"
            entry["reason"] = "not_marked_failed"
            skipped += 1
            report["results"].append(entry)
            continue

        vr = validate_file(
            path,
            max_qa_models=int(cfg["max_qa_models"]),
            max_lg_models=int(cfg["max_lg_models"]),
        )
        entry["issues"] = [i.as_dict() for i in vr.issues]
        if cfg["verbose"]:
            print(f"[{path.name}] valid={vr.valid} issues={len(vr.issues)}")

        if vr.parsed is None:
            entry["status"] = "VALIDATION_FAILED"
            validation_failed += 1
            failed_hashes[file_hash] = "parse_failure"
            report["results"].append(entry)
            continue

        parsed = vr.parsed
        entry["target"] = parsed.target
        entry["category"] = parsed.pfrmat
        entry["model_indices"] = [b.index for b in parsed.model_blocks]

        if target_filter and parsed.target not in target_filter:
            entry["status"] = "SKIPPED"
            entry["reason"] = "target_filter"
            skipped += 1
            report["results"].append(entry)
            continue

        if not vr.valid:
            entry["status"] = "VALIDATION_FAILED"
            validation_failed += 1
            failed_hashes[file_hash] = "validation_failed"
            report["results"].append(entry)
            continue

        model_keys = build_model_keys(parsed.target, parsed.pfrmat, [b.index for b in parsed.model_blocks])

        if not cfg["force_resubmit"]:
            existing = [k for k in model_keys if k in successful_keys]
            if existing:
                entry["status"] = "SKIPPED"
                entry["reason"] = "already_successful_model_key"
                entry["existing_keys"] = existing
                skipped += 1
                report["results"].append(entry)
                continue

        if cfg["dry_run"]:
            entry["status"] = "DRY_RUN_OK"
            successful += 1
            report["results"].append(entry)
            continue

        submit_result = submit_with_retry(
            path=path,
            endpoint=str(cfg["endpoint"]),
            email=email,
            timeout_sec=int(cfg["timeout"]),
            max_retries=int(cfg["max_retries"]),
        )

        entry["attempts"] = submit_result["attempts"]
        entry["http_code"] = submit_result["http_code"]
        entry["accession"] = submit_result["accession"]
        entry["response_excerpt"] = str(submit_result.get("response_text", ""))[:1000]
        entry["network_error"] = submit_result.get("error", "")

        if submit_result["success"]:
            entry["status"] = "SUBMITTED"
            successful += 1

            submissions.append(
                {
                    "time": utc_now(),
                    "file": str(path),
                    "hash": file_hash,
                    "target": parsed.target,
                    "category": parsed.pfrmat,
                    "models": [b.index for b in parsed.model_blocks],
                    "status": "SUBMITTED",
                    "accession": submit_result.get("accession"),
                    "http_code": submit_result.get("http_code"),
                }
            )

            for mk in model_keys:
                successful_keys[mk] = {
                    "time": utc_now(),
                    "file": str(path),
                    "hash": file_hash,
                    "accession": submit_result.get("accession"),
                }

            if file_hash in failed_hashes:
                failed_hashes.pop(file_hash, None)

        else:
            entry["status"] = "FAILED"
            failed += 1
            failed_hashes[file_hash] = "upload_failed"
            submissions.append(
                {
                    "time": utc_now(),
                    "file": str(path),
                    "hash": file_hash,
                    "target": parsed.target,
                    "category": parsed.pfrmat,
                    "models": [b.index for b in parsed.model_blocks],
                    "status": "FAILED",
                    "http_code": submit_result.get("http_code"),
                    "error": submit_result.get("error", ""),
                }
            )

        report["results"].append(entry)

    report["summary"] = {
        "total_files": len(files),
        "submitted_or_dry_ok": successful,
        "failed": failed,
        "validation_failed": validation_failed,
        "skipped": skipped,
    }

    checkpoint["updated_at"] = utc_now()
    checkpoint["successful_model_keys"] = successful_keys
    checkpoint["failed_hashes"] = failed_hashes
    checkpoint["submissions"] = submissions

    save_json(checkpoint_path, checkpoint)
    save_json(report_path, report)

    print("Run complete")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Report: {report_path}")

    return 0 if failed == 0 and validation_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
