from __future__ import annotations
import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import joblib
import numpy as np

CPP_KEYWORDS = set("""
alignas alignof and and_eq asm auto bitand bitor bool break case catch char
char16_t char32_t class compl const const_cast constexpr continue decltype
default delete do double dynamic_cast else enum explicit export extern false
float for friend goto if inline int long mutable namespace new noexcept not
not_eq nullptr operator or or_eq private protected public register
reinterpret_cast return short signed sizeof static static_assert static_cast
struct switch template this thread_local throw true try typedef typeid typename
union unsigned using virtual void volatile wchar_t while xor xor_eq
""".split())

PYTHON_KEYWORDS = set("""
False None True and as assert async await break class continue def del elif
else except finally for from global if import in is lambda nonlocal not or
pass raise return try while with yield
""".split())

IDENTIFIER_RE = re.compile(r"\b[_A-Za-z][_A-Za-z0-9]*\b")
SNAKE_CASE_RE = re.compile(r"^[a-z]+(_[a-z0-9]+)+$")
CAMEL_CASE_RE = re.compile(r"^[a-z]+(?:[A-Z][a-z0-9]*)+$")
UPPER_CASE_RE = re.compile(r"^[A-Z0-9_]+$")

def _detect_language(code: str, hint: Optional[str]) -> str:
	if hint:
		h = hint.lower()
		if "py" in h: return "python"
		if "cc" in h or "cpp" in h or "c++" in h: return "cpp"
	if "#include" in code: return "cpp"
	if re.search(r"^\s*def\s+", code, re.MULTILINE): return "python"
	return "unknown"

def _line_features(lines: List[str]) -> Dict[str, float]:
	if not lines:
		return {"line_count": 0.0, "avg_line_length": 0.0, "std_line_length": 0.0,
		        "blank_line_ratio": 0.0, "comment_ratio": 0.0}
	
	lengths = [len(line) for line in lines]
	n = len(lines)
	avg = sum(lengths) / n
	std = math.sqrt(sum((x - avg) ** 2 for x in lengths) / n)
	blank = sum(1 for line in lines if not line.strip())
	comment = sum(1 for line in lines if line.strip().startswith(("//", "#", "/*")))
	
	return {"line_count": float(n), "avg_line_length": avg, "std_line_length": std,
	        "blank_line_ratio": blank / n, "comment_ratio": comment / n}

def _indent_features(lines: List[str]) -> Dict[str, float]:
	indent_units, space_indents = [], []
	tab_lines = space_lines = 0
	
	for line in lines:
		if not line.strip(): continue
		prefix_len = len(line) - len(line.lstrip(" \t"))
		if not prefix_len: continue
		prefix = line[:prefix_len]
		indent_units.append(prefix_len)
		if prefix[0] == "\t":
			tab_lines += 1
		elif prefix[0] == " ":
			space_lines += 1
			if set(prefix) == {" "}: space_indents.append(prefix_len)
	
	total = tab_lines + space_lines
	if space_indents:
		gcd_val = space_indents[0]
		for length in space_indents[1:]: gcd_val = math.gcd(gcd_val, length)
		dominant = float(gcd_val)
		avg_space = sum(space_indents) / len(space_indents)
	else:
		dominant = avg_space = 0.0
	
	if indent_units:
		avg_indent = sum(indent_units) / len(indent_units)
		std_indent = math.sqrt(sum((v - avg_indent) ** 2 for v in indent_units) / len(indent_units))
	else:
		avg_indent = std_indent = 0.0
	
	return {"indent_tab_ratio": tab_lines / total if total else 0.0,
	        "indent_space_ratio": space_lines / total if total else 0.0,
	        "avg_indent_width": avg_indent, "std_indent_width": std_indent,
	        "avg_space_indent_width": avg_space, "dominant_space_indent": dominant}

def _brace_features(lines: List[str]) -> Dict[str, float]:
	same_no_space = same_space = next_line = total = 0
	
	for line in lines:
		if "{" not in line: continue
		idx = -1
		while (idx := line.find("{", idx + 1)) != -1:
			total += 1
			prefix = line[:idx]
			if not prefix.rstrip():
				next_line += 1
			elif prefix.endswith(" "):
				same_space += 1
			else:
				same_no_space += 1
	
	if not total:
		return {"brace_same_line_no_space_ratio": 0.0, "brace_same_line_space_ratio": 0.0,
		        "brace_next_line_ratio": 0.0}
	
	return {"brace_same_line_no_space_ratio": same_no_space / total,
	        "brace_same_line_space_ratio": same_space / total,
	        "brace_next_line_ratio": next_line / total}

def _comma_features(code: str) -> Dict[str, float]:
	spaced = len(re.findall(r",\s", code))
	tight = len(re.findall(r",(?=\S)", code))
	total = spaced + tight
	if not total: return {"comma_space_ratio": 0.0, "comma_no_space_ratio": 0.0}
	return {"comma_space_ratio": spaced / total, "comma_no_space_ratio": tight / total}

def _include_features(lines: List[str]) -> Dict[str, float]:
	includes = [line for line in lines if line.strip().startswith("#include")]
	if not includes: return {"include_count": 0.0, "uses_bits_header": 0.0}
	return {"include_count": float(len(includes)),
	        "uses_bits_header": 1.0 if any("<bits/stdc++.h>" in line for line in includes) else 0.0}

def _import_features(lines: List[str]) -> Dict[str, float]:
	imports = [line for line in lines if line.strip().startswith(("import", "from"))]
	return {"import_count": float(len(imports))}

def _identifier_features(code: str, language: str) -> Dict[str, float]:
	identifiers = IDENTIFIER_RE.findall(code)
	if not identifiers:
		return {"avg_identifier_length": 0.0, "single_char_identifier_ratio": 0.0,
		        "snake_case_ratio": 0.0, "camel_case_ratio": 0.0, "upper_case_ratio": 0.0}
	
	keywords = CPP_KEYWORDS if language == "cpp" else PYTHON_KEYWORDS if language == "python" else set()
	filtered = [t for t in identifiers if t not in keywords] or identifiers
	
	n = len(filtered)
	avg_len = sum(len(t) for t in filtered) / n
	single = sum(1 for t in filtered if len(t) == 1)
	snake = sum(1 for t in filtered if SNAKE_CASE_RE.match(t))
	camel = sum(1 for t in filtered if CAMEL_CASE_RE.match(t))
	upper = sum(1 for t in filtered if UPPER_CASE_RE.match(t))
	
	return {"avg_identifier_length": avg_len, "single_char_identifier_ratio": single / n,
	        "snake_case_ratio": snake / n, "camel_case_ratio": camel / n,
	        "upper_case_ratio": upper / n}

def _keyword_density(code: str, language: str) -> Dict[str, float]:
	keywords = CPP_KEYWORDS if language == "cpp" else PYTHON_KEYWORDS if language == "python" else set()
	tokens = IDENTIFIER_RE.findall(code)
	if not tokens: return {"keyword_density": 0.0}
	hits = sum(1 for token in tokens if token in keywords)
	return {"keyword_density": hits / len(tokens)}

def extract_features(code: str, language_hint: Optional[str] = None) -> Dict[str, float]:
	language = _detect_language(code, language_hint)
	lines = code.splitlines()
	
	features = {"language_cpp": 1.0 if language == "cpp" else 0.0,
	            "language_python": 1.0 if language == "python" else 0.0}
	
	features.update(_line_features(lines))
	features.update(_indent_features(lines))
	features.update(_brace_features(lines))
	features.update(_comma_features(code))
	features.update(_identifier_features(code, language))
	features.update(_keyword_density(code, language))
	
	if language == "cpp":
		features.update(_include_features(lines))
		features["import_count"] = 0.0
	elif language == "python":
		features.update(_import_features(lines))
		features.update({"include_count": 0.0, "uses_bits_header": 0.0})
	else:
		features.update({"include_count": 0.0, "uses_bits_header": 0.0, "import_count": 0.0})
	
	return features

def load_artifacts(artifacts_dir: Path) -> tuple[Any, List[str]]:
	pipeline_path = artifacts_dir / "v0.joblib"
	if not pipeline_path.exists():
		raise FileNotFoundError(f"Missing pipeline artifact: {pipeline_path}")
	
	pipeline = joblib.load(pipeline_path)
	feature_order = getattr(pipeline, "_feature_order", None)
	if feature_order is None:
		fallback = artifacts_dir / "feature_order.json"
		if fallback.exists():
			feature_order = json.loads(fallback.read_text(encoding="utf-8"))
		else:
			raise ValueError("Feature order metadata absent. Ensure pipeline was exported with embedded order.")
	
	if not isinstance(feature_order, list):
		raise TypeError("Feature order metadata must be a list")
	
	return pipeline, [str(f) for f in feature_order]

def embed_code(pipeline: Any, feature_order: List[str], code: str, *, language: Optional[str]) -> List[float]:
	features = extract_features(code, language)
	row = np.array([[features.get(name, 0.0) for name in feature_order]], dtype=float)
	return pipeline.transform(row)[0].tolist()

def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Generate style embeddings for source code.")
	parser.add_argument("--model-dir", type=Path, default=Path("../artifacts"),
	                    help="Directory containing v0.joblib and optional feature_order.json")
	parser.add_argument("--code-path", type=Path, help="Path to source file")
	parser.add_argument("--code", type=str, help="Inline code snippet")
	parser.add_argument("--language", type=str, help="Language hint (cpp/python)")
	parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
	return parser.parse_args(argv)

def _load_code(args: argparse.Namespace) -> tuple[str, Optional[str]]:
	if args.code_path:
		return args.code_path.read_text(encoding="utf-8", errors="replace"), str(args.code_path)
	if args.code is not None:
		return args.code, None
	data = sys.stdin.read()
	if not data:
		raise ValueError("No code supplied. Use --code-path, --code, or stdin.")
	return data, None

def main(argv: Optional[Sequence[str]] = None) -> int:
	args = _parse_args(argv)
	code, origin = _load_code(args)
	pipeline, feature_order = load_artifacts(args.model_dir)
	embedding = embed_code(pipeline, feature_order, code, language=args.language)

	payload = {"embedding": embedding, "dimension": len(embedding),
	           "source": origin, "language_hint": args.language}
	
	json.dump(payload, sys.stdout, ensure_ascii=False, indent=2 if args.pretty else None)
	if args.pretty: sys.stdout.write("\n")
	return 0

if __name__ == "__main__":
	raise SystemExit(main())
