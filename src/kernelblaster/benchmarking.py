from __future__ import annotations

import csv
import hashlib
import io
import math
from pathlib import Path
import re
import statistics
from typing import Any, Iterable


BENCHMARK_SCHEMA_VERSION = "1.0"
BENCHMARK_MARKER = "KERNELBLASTER_BENCHMARK_JSON "


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _matching_delimiter(text: str, start: int, opening: str, closing: str) -> int:
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return index
    raise ValueError(f"Unbalanced {opening}{closing} delimiters.")


def find_launch_definition(source: str) -> tuple[str, tuple[int, int]]:
    """Find the host launcher definition, excluding declarations."""
    matches: list[tuple[str, tuple[int, int]]] = []
    pattern = re.compile(
        r"(?:inline\s+)?void\s+launch_gpu_implementation\s*\(",
        flags=re.MULTILINE,
    )
    for match in pattern.finditer(source):
        opening_parenthesis = source.find("(", match.start())
        closing_parenthesis = _matching_delimiter(
            source, opening_parenthesis, "(", ")"
        )
        cursor = closing_parenthesis + 1
        while cursor < len(source) and source[cursor].isspace():
            cursor += 1
        if cursor >= len(source) or source[cursor] != "{":
            continue
        closing_brace = _matching_delimiter(source, cursor, "{", "}")
        matches.append(
            (source[match.start() : closing_brace + 1], (match.start(), closing_brace + 1))
        )
    if len(matches) != 1:
        raise ValueError(
            "CUDA source must contain exactly one launch_gpu_implementation definition."
        )
    return matches[0]


def normalize_cuda_source(source: str) -> tuple[str, list[str]]:
    """Remove explicit host synchronization only from the launcher body."""
    replacements = [
        (
            re.compile(
                r"CUDA_CHECK\s*\(\s*cudaDeviceSynchronize\s*\(\s*\)\s*\)\s*;"
            ),
            "CUDA_CHECK(cudaGetLastError());",
            "cudaDeviceSynchronize via CUDA_CHECK",
        ),
        (
            re.compile(r"cudaDeviceSynchronize\s*\(\s*\)\s*;"),
            "(void)cudaGetLastError();",
            "cudaDeviceSynchronize",
        ),
        (
            re.compile(r"cudaStreamSynchronize\s*\([^;]*?\)\s*;"),
            "(void)cudaGetLastError();",
            "cudaStreamSynchronize",
        ),
    ]
    launcher, span = find_launch_definition(source)
    normalized_launcher = launcher
    applied: list[str] = []
    for pattern, replacement, label in replacements:
        normalized_launcher, count = pattern.subn(replacement, normalized_launcher)
        if count:
            applied.extend([label] * count)
    normalized = source[: span[0]] + normalized_launcher + source[span[1] :]
    return normalized, applied


def find_launch_declaration(driver: str) -> tuple[str, tuple[int, int]]:
    matches = list(
        re.finditer(
            r"void\s+launch_gpu_implementation\s*\(.*?\)\s*;",
            driver,
            flags=re.DOTALL,
        )
    )
    if len(matches) != 1:
        raise ValueError(
            "Driver must contain exactly one launch_gpu_implementation declaration."
        )
    match = matches[0]
    return match.group(0), match.span()


def _balanced_call_end(text: str, opening_parenthesis: int) -> int:
    depth = 0
    for index in range(opening_parenthesis, len(text)):
        char = text[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                cursor = index + 1
                while cursor < len(text) and text[cursor].isspace():
                    cursor += 1
                if cursor >= len(text) or text[cursor] != ";":
                    raise ValueError("Kernel launcher call is not terminated by a semicolon.")
                return cursor + 1
    raise ValueError("Kernel launcher call has unbalanced parentheses.")


def find_launch_call(driver: str) -> tuple[str, tuple[int, int]]:
    _declaration, declaration_span = find_launch_declaration(driver)
    for match in re.finditer(r"\blaunch_gpu_implementation\s*\(", driver):
        if declaration_span[0] <= match.start() < declaration_span[1]:
            continue
        opening = driver.find("(", match.start())
        end = _balanced_call_end(driver, opening)
        return driver[match.start() : end], (match.start(), end)
    raise ValueError("Driver does not call launch_gpu_implementation.")


def split_compilation_units(driver: str, cuda_source: str) -> tuple[str, str, str]:
    declaration, span = find_launch_declaration(driver)
    main = driver[: span[0]] + driver[span[1] :]
    main = '#include "cuda_model.cuh"\n' + main
    header = "#include <cstdint>\n#include <torch/torch.h>\n" + declaration + "\n"
    cuda = '#include "cuda_model.cuh"\n' + cuda_source
    cuda = cuda.replace(
        "inline void launch_gpu_implementation", "void launch_gpu_implementation"
    ).replace('extern "C"', "")
    return main, header, cuda


def instrument_driver(
    driver: str,
    *,
    seed: int,
    warmup: int,
    repetitions: int,
    inner_loops: int,
) -> str:
    call, span = find_launch_call(driver)
    if min(warmup, repetitions) < 1 or inner_loops < 0:
        raise ValueError("Warmup/repetitions must be positive and inner_loops non-negative.")

    repeated_call = "\n".join("        " + line for line in call.splitlines())
    block = f"""{{
    constexpr int kb_warmup = {warmup};
    constexpr int kb_repetitions = {repetitions};
    int kb_inner_loops = {inner_loops};
    cudaEvent_t kb_start, kb_stop;
    cudaEventCreate(&kb_start);
    cudaEventCreate(&kb_stop);

    for (int kb_i = 0; kb_i < kb_warmup; ++kb_i) {{
{repeated_call}
    }}
    cudaDeviceSynchronize();

    if (kb_inner_loops == 0) {{
        constexpr int kb_calibration_loops = 10;
        cudaEventRecord(kb_start);
        for (int kb_i = 0; kb_i < kb_calibration_loops; ++kb_i) {{
{repeated_call}
        }}
        cudaEventRecord(kb_stop);
        cudaEventSynchronize(kb_stop);
        float kb_calibration_ms = 0.0f;
        cudaEventElapsedTime(&kb_calibration_ms, kb_start, kb_stop);
        const double kb_single_us = std::max(
            0.001, static_cast<double>(kb_calibration_ms) * 1000.0 /
                kb_calibration_loops);
        kb_inner_loops = std::clamp(
            static_cast<int>(std::ceil(1000.0 / kb_single_us)), 1, 10000);
    }}

    std::vector<double> kb_samples_us;
    kb_samples_us.reserve(kb_repetitions);
    for (int kb_rep = 0; kb_rep < kb_repetitions; ++kb_rep) {{
        cudaEventRecord(kb_start);
        for (int kb_i = 0; kb_i < kb_inner_loops; ++kb_i) {{
{repeated_call}
        }}
        cudaEventRecord(kb_stop);
        cudaEventSynchronize(kb_stop);
        float kb_elapsed_ms = 0.0f;
        cudaEventElapsedTime(&kb_elapsed_ms, kb_start, kb_stop);
        kb_samples_us.push_back(
            static_cast<double>(kb_elapsed_ms) * 1000.0 / kb_inner_loops);
    }}
    cudaEventDestroy(kb_start);
    cudaEventDestroy(kb_stop);

    std::cout << "{BENCHMARK_MARKER}{{\\\"inner_loops\\\":"
              << kb_inner_loops << ",\\\"samples_us\\\":[";
    for (size_t kb_i = 0; kb_i < kb_samples_us.size(); ++kb_i) {{
        if (kb_i) std::cout << ',';
        std::cout << std::setprecision(12) << kb_samples_us[kb_i];
    }}
    std::cout << "]}}" << std::endl;
}}"""

    instrumented = driver[: span[0]] + block + driver[span[1] :]
    includes = (
        "#include <algorithm>\n"
        "#include <cmath>\n"
        "#include <cuda_runtime.h>\n"
        "#include <iomanip>\n"
        "#include <vector>\n"
    )
    instrumented = includes + instrumented
    if "torch::manual_seed(" not in instrumented:
        instrumented = instrumented.replace(
            "int main() {",
            f"int main() {{\n    torch::manual_seed({seed});",
            1,
        )
    return instrumented


def instrument_profiler_driver(driver: str) -> str:
    """Limit Nsight Compute collection to the target launcher call."""
    call, span = find_launch_call(driver)
    block = (
        "cudaProfilerStart();\n"
        f"    {call}\n"
        "    cudaProfilerStop();"
    )
    instrumented = driver[: span[0]] + block + driver[span[1] :]
    return "#include <cuda_profiler_api.h>\n" + instrumented


def percentile(values: Iterable[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("Cannot calculate a percentile of an empty sequence.")
    if not 0 <= fraction <= 1:
        raise ValueError("Percentile fraction must be between zero and one.")
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def latency_summary(values: Iterable[float]) -> dict[str, float | int]:
    samples = [float(value) for value in values]
    if not samples:
        raise ValueError("At least one latency sample is required.")
    mean = statistics.fmean(samples)
    standard_deviation = statistics.pstdev(samples)
    return {
        "count": len(samples),
        "median_us": statistics.median(samples),
        "mean_us": mean,
        "p10_us": percentile(samples, 0.10),
        "p90_us": percentile(samples, 0.90),
        "min_us": min(samples),
        "max_us": max(samples),
        "stddev_us": standard_deviation,
        "cv_percent": (standard_deviation / mean * 100.0) if mean else 0.0,
    }


def session_spread_percent(values: Iterable[float]) -> float:
    medians = [float(value) for value in values]
    if not medians:
        raise ValueError("At least one session median is required.")
    minimum = min(medians)
    if minimum <= 0:
        raise ValueError("Session medians must be positive.")
    return (max(medians) / minimum - 1.0) * 100.0


def ncu_metric_names(csv_text: str) -> list[str]:
    """Extract the actual Metric Name column from an NCU raw CSV export."""
    rows = list(csv.reader(io.StringIO(csv_text)))
    header_index: int | None = None
    metric_column: int | None = None
    for index, row in enumerate(rows):
        normalized = [cell.strip().lower() for cell in row]
        if "metric name" in normalized:
            header_index = index
            metric_column = normalized.index("metric name")
            break
    if header_index is None or metric_column is None:
        return []

    names: list[str] = []
    for row in rows[header_index + 1 :]:
        if len(row) <= metric_column:
            continue
        name = row[metric_column].strip()
        if name and name not in names:
            names.append(name)
    return names


def comparison_validity(
    *,
    baseline_source_sha256: str,
    candidate_source_sha256: str,
    baseline_session_medians: Iterable[float],
    candidate_session_medians: Iterable[float],
    speedup: float,
    max_session_spread_percent: float,
) -> dict[str, Any]:
    baseline_values = [float(value) for value in baseline_session_medians]
    candidate_values = [float(value) for value in candidate_session_medians]
    if len(baseline_values) != len(candidate_values) or not baseline_values:
        raise ValueError("Baseline and candidate require paired session medians.")

    baseline_spread = session_spread_percent(baseline_values)
    candidate_spread = session_spread_percent(candidate_values)
    stable = max(baseline_spread, candidate_spread) <= max_session_spread_percent
    same_source = baseline_source_sha256 == candidate_source_sha256
    self_check_passed = (0.95 <= speedup <= 1.05) if same_source else None
    formal_valid = stable and (self_check_passed is not False)
    return {
        "comparison_kind": "self_check" if same_source else "candidate",
        "baseline_session_spread_percent": baseline_spread,
        "candidate_session_spread_percent": candidate_spread,
        "max_session_spread_percent": max_session_spread_percent,
        "stable": stable,
        "self_check_passed": self_check_passed,
        "formal_valid": formal_valid,
    }


def write_compilation_units(
    directory: Path,
    driver: str,
    cuda_source: str,
) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    main, header, cuda = split_compilation_units(driver, cuda_source)
    (directory / "main.cpp").write_text(main, encoding="utf-8")
    (directory / "cuda_model.cuh").write_text(header, encoding="utf-8")
    (directory / "cuda_model.cu").write_text(cuda, encoding="utf-8")
