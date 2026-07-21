#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synchronize and verify the repository's living portfolio status blocks."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Callable


ROOT_DIR = Path(__file__).resolve().parent.parent
STATUS_PATH = Path("portfolio/status.json")
START_TEMPLATE = "<!-- {marker}:START -->"
END_TEMPLATE = "<!-- {marker}:END -->"

TARGETS: tuple[tuple[Path, str, str], ...] = (
    (Path("README.md"), "PORTFOLIO_STATUS", "root_en"),
    (Path("README.zh-CN.md"), "PORTFOLIO_STATUS", "root_zh"),
    (Path("docs/portfolio/README.md"), "PORTFOLIO_PROGRESS", "index_en"),
    (Path("docs/portfolio/README.zh-CN.md"), "PORTFOLIO_PROGRESS", "index_zh"),
    (Path("docs/portfolio/validation.md"), "VALIDATION_STATUS", "validation_en"),
    (
        Path("docs/portfolio/validation.zh-CN.md"),
        "VALIDATION_STATUS",
        "validation_zh",
    ),
    (Path("docs/portfolio/architecture.md"), "ARCHITECTURE_STATUS", "architecture_en"),
    (
        Path("docs/portfolio/architecture.zh-CN.md"),
        "ARCHITECTURE_STATUS",
        "architecture_zh",
    ),
    (Path("docs/portfolio/rmsnorm-case-study.md"), "RMSNORM_STATUS", "rmsnorm_en"),
    (
        Path("docs/portfolio/rmsnorm-case-study.zh-CN.md"),
        "RMSNORM_STATUS",
        "rmsnorm_zh",
    ),
)

RELEVANT_PREFIXES = (
    "artifacts/portfolio-v1.0/",
    "portfolio/case_studies/",
    "portfolio/suites/",
    "scripts/analyze_",
    "scripts/benchmark_",
    "src/kernelblaster/benchmarking.py",
)
DOCUMENTATION_PREFIXES = (
    "README.md",
    "README.zh-CN.md",
    "CONTRIBUTING.md",
    "docs/",
    "portfolio/status.json",
)
MACHINE_PATH_PATTERNS = (
    re.compile(r"/home/[^/\s`'\"<>]+/(?:src/)?KernelBlaster"),
    re.compile(r"[A-Za-z]:\\Users\\[^\\\s`'\"<>]+\\", re.IGNORECASE),
    re.compile(r"\\\\wsl(?:\.localhost|\$)", re.IGNORECASE),
)


class DocumentationSyncError(RuntimeError):
    """Raised when the living documentation cannot be trusted."""


def _load_json(
    path: Path,
    *,
    allowed_schema_versions: tuple[str, ...] = ("1.0",),
) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DocumentationSyncError(f"Cannot read valid JSON from {path}: {error}") from error
    if payload.get("schema_version") not in allowed_schema_versions:
        allowed = ", ".join(allowed_schema_versions)
        raise DocumentationSyncError(f"{path} must use schema_version {allowed}.")
    return payload


def _resolve_source(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise DocumentationSyncError(f"Status source must be repository-relative: {relative}")
    resolved = (root / candidate).resolve()
    if not resolved.is_file():
        raise DocumentationSyncError(f"Referenced status source does not exist: {relative}")
    return resolved


def _geomean(values: list[float]) -> float:
    if not values or any(value <= 0 for value in values):
        raise DocumentationSyncError("Geometric means require positive non-empty values.")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _deep_case_speedup(path: Path) -> float:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    matches = [
        row
        for row in rows
        if row.get("task_id") == "036"
        and row.get("candidate") == "v3c"
        and row.get("comparison_scope") == "upstream_baseline"
    ]
    if len(matches) != 1:
        raise DocumentationSyncError("Expected one V3c-to-upstream RMSNorm result.")
    return float(matches[0]["speedup"])


def load_context(root: Path = ROOT_DIR) -> dict[str, Any]:
    status = _load_json(
        root / STATUS_PATH,
        allowed_schema_versions=("2.0",),
    )
    sources = status.get("sources")
    if not isinstance(sources, dict):
        raise DocumentationSyncError("portfolio/status.json requires a sources object.")
    resolved = {name: _resolve_source(root, str(path)) for name, path in sources.items()}
    environment = _load_json(resolved["environment"])
    comparison = _load_json(resolved["comparison"])
    targeted_v2 = _load_json(
        resolved["targeted_validation_v2"],
        allowed_schema_versions=("2.0",),
    )
    targeted_results = targeted_v2.get("results")
    if not isinstance(targeted_results, list) or {
        str(row.get("task_id")) for row in targeted_results
    } != {"004", "007", "036", "040", "095"}:
        raise DocumentationSyncError(
            "Schema-v2 targeted validation must contain 004/007/036/040/095."
        )
    rows = comparison.get("results")
    if not isinstance(rows, list) or len(rows) != 10:
        raise DocumentationSyncError("Core 10 comparison must contain exactly ten results.")
    task_ids = {str(row.get("task_id")) for row in rows}
    expected = {"004", "007", "019", "023", "026", "036", "040", "047", "088", "095"}
    if task_ids != expected:
        raise DocumentationSyncError(f"Unexpected Core 10 task IDs: {sorted(task_ids)}")

    all10 = {
        "attempted_upstream": _geomean([float(row["attempted_speedup"]) for row in rows]),
        "strict_upstream": _geomean([float(row["portfolio_speedup"]) for row in rows]),
        "attempted_pytorch": _geomean(
            [float(row["candidate_vs_pytorch_best"]) for row in rows]
        ),
        "strict_pytorch": _geomean(
            [float(row["selected_vs_pytorch_best"]) for row in rows]
        ),
        "verified": sum(bool(row["verified_improvement"]) for row in rows),
        "candidate_wins": sum(float(row["candidate_vs_pytorch_best"]) > 1.0 for row in rows),
    }
    new9_rows = [row for row in rows if str(row["task_id"]) != "036"]
    new9 = {
        "attempted_upstream": _geomean(
            [float(row["attempted_speedup"]) for row in new9_rows]
        ),
        "strict_upstream": _geomean(
            [float(row["portfolio_speedup"]) for row in new9_rows]
        ),
        "attempted_pytorch": _geomean(
            [float(row["candidate_vs_pytorch_best"]) for row in new9_rows]
        ),
        "strict_pytorch": _geomean(
            [float(row["selected_vs_pytorch_best"]) for row in new9_rows]
        ),
        "verified": sum(bool(row["verified_improvement"]) for row in new9_rows),
        "candidate_wins": sum(
            float(row["candidate_vs_pytorch_best"]) > 1.0 for row in new9_rows
        ),
    }

    summary = comparison.get("summary", {})
    comparisons = {
        "attempted_candidate_geomean_speedup": all10["attempted_upstream"],
        "all10_selected_portfolio_geomean_speedup": all10["strict_upstream"],
        "attempted_candidate_vs_pytorch_best_geomean": all10["attempted_pytorch"],
        "selected_vs_pytorch_best_geomean": all10["strict_pytorch"],
    }
    for key, value in comparisons.items():
        if not math.isclose(float(summary.get(key, -1)), value, rel_tol=1e-9):
            raise DocumentationSyncError(f"Comparison summary disagrees with rows for {key}.")

    gpu = environment.get("gpu", {})
    container = environment.get("container", {})
    return {
        "status": status,
        "sources": sources,
        "environment": environment,
        "comparison": comparison,
        "targeted_v2": targeted_v2,
        "targeted_v2_results": targeted_results,
        "rows": rows,
        "all10": all10,
        "new9": new9,
        "rmsnorm_deep_speedup": _deep_case_speedup(resolved["deep_case_results"]),
        "rmsnorm_unified_speedup": float(
            next(row for row in rows if str(row["task_id"]) == "036")["attempted_speedup"]
        ),
        "gpu_name": str(gpu.get("name")),
        "sm": str(gpu.get("target")),
        "driver": str(gpu.get("windows_driver")),
        "cuda": str(container.get("cuda_toolkit")),
        "pytorch": str(container.get("pytorch")),
    }


def _f(value: float) -> str:
    return f"{value:.3f}×"


def _validation_labels(validation: dict[str, Any], *, chinese: bool) -> dict[str, str]:
    """Render the canonical validation states without leaking one locale into another."""
    required = {
        "current_cpu_pytest",
        "official_correctness",
        "live_api_smoke",
        "ncu_counters",
        "cross_gpu",
    }
    missing = sorted(required - validation.keys())
    if missing:
        raise DocumentationSyncError(
            "portfolio/status.json is missing validation fields: " + ", ".join(missing)
        )

    raw = {key: str(validation[key]) for key in required}
    if not chinese:
        return raw

    cpu_match = re.fullmatch(r"(\d+) passed", raw["current_cpu_pytest"])
    correctness_match = re.fullmatch(
        r"historical (\d+/\d+); schema-v2 targeted (\d+/\d+) passed",
        raw["official_correctness"],
    )
    if not cpu_match or not correctness_match:
        raise DocumentationSyncError(
            "Cannot localize CPU/correctness status from portfolio/status.json."
        )
    if raw["live_api_smoke"] != "NOT RUN (historical HTTP 401; credential not revalidated)":
        raise DocumentationSyncError(
            "Cannot localize live_api_smoke status from portfolio/status.json."
        )
    if raw["ncu_counters"] != "blocked: ERR_NVGPUCTRPERM":
        raise DocumentationSyncError(
            "Cannot localize ncu_counters status from portfolio/status.json."
        )
    if raw["cross_gpu"] != "NOT RUN (Day 11-14 out of scope)":
        raise DocumentationSyncError(
            "Cannot localize cross_gpu status from portfolio/status.json."
        )
    return {
        "current_cpu_pytest": f"{cpu_match.group(1)} 项通过",
        "official_correctness": (
            f"历史 {correctness_match.group(1)}；schema v2 定向验证 "
            f"{correctness_match.group(2)} 通过"
        ),
        "live_api_smoke": "未运行（历史记录为 HTTP 401；凭据尚未重新验证）",
        "ncu_counters": "阻塞：ERR_NVGPUCTRPERM",
        "cross_gpu": "未运行（Day 11–14 不在本阶段范围）",
    }


def _evidence_links(
    context: dict[str, Any], *, chinese: bool, prefix: str = ""
) -> str:
    sources = context["sources"]
    targeted_report = (
        sources["targeted_validation_report_zh"]
        if chinese
        else sources["targeted_validation_report_en"]
    )
    labels = (
        ("Schema v2 定向验证", "Schema-v2 targeted validation", targeted_report),
        ("Schema v2 结果 JSON", "Schema-v2 result JSON", sources["targeted_validation_v2"]),
        ("中文完整报告", "Full Chinese report", sources["comparison_report_zh"]),
        ("英文摘要", "English summary", sources["comparison_summary_en"]),
        ("逐题 JSON", "Per-task JSON", sources["comparison"]),
        ("对比图", "Comparison figure", sources["comparison_figure"]),
        ("原始文件哈希", "Raw-file hashes", sources["raw_sha256"]),
        ("候选清单", "Candidate manifest", sources["candidate_manifest"]),
    )
    links = [
        f"[{zh if chinese else en}]({prefix}{path})" for zh, en, path in labels
    ]
    return " · ".join(links)


def _root_block(context: dict[str, Any], *, chinese: bool) -> str:
    status = context["status"]
    validation = _validation_labels(status["validation"], chinese=chinese)
    n9, a10 = context["new9"], context["all10"]
    if chinese:
        return f"""当前 Fork 已在 **{context['gpu_name']}（{context['sm']}）** 上完成 Day 1–10 基础设施、RMSNorm 深度案例、Core 10 手工候选和同卡 PyTorch 对比。环境为 WSL2、CUDA {context['cuda']}、驱动 {context['driver']}。

| 验证项目 | 当前状态 |
| --- | --- |
| CPU 测试 | **{validation['current_cpu_pytest']}**（当前分支） |
| CUDA 编译与官方正确性 | **{validation['official_correctness']}** |
| CUDA Events 与同卡 PyTorch | **schema v2 定向验证：004/036/040 正式提升；007 无法定论；095 保持探索** |
| 外部 LLM 冒烟测试 | **{validation['live_api_smoke']}** |
| Nsight Compute 硬件计数器 | **{validation['ncu_counters']}** |
| 跨 GPU 复测 | **{validation['cross_gpu']}** |

| 历史 v1 实测范围 | 相对仓库原版（诊断 / 旧严格口径） | 相对 PyTorch 最快方法（诊断 / 旧严格口径） |
| --- | ---: | ---: |
| 本轮新增九题 | {_f(n9['attempted_upstream'])} / {_f(n9['strict_upstream'])} | {_f(n9['attempted_pytorch'])} / {_f(n9['strict_pytorch'])} |
| 完整 Core 10（含 RMSNorm） | {_f(a10['attempted_upstream'])} / {_f(a10['strict_upstream'])} | {_f(a10['attempted_pytorch'])} / {_f(a10['strict_pytorch'])} |

上述严格值作为不可变的历史 v1 证据保留。独立的 schema v2 定向复测已续认 004/036/040，将 007 标为无法定论，并让 095 保持探索；它不是完整 Core 10 或 Agent 结果。新口径还检查 p99/max 误差回归、NaN/Inf 和五次确定性。当前 Agent Pilot 与 Core 10 搜索均未重新运行。

{_evidence_links(context, chinese=True)}"""
    return f"""This fork has completed the Day 1–10 infrastructure, the RMSNorm deep case, manual Core 10 candidates, and a same-GPU PyTorch comparison on **{context['gpu_name']} ({context['sm']})**. The measured environment is WSL2, CUDA {context['cuda']}, and driver {context['driver']}.

| Validation item | Current status |
| --- | --- |
| CPU tests | **{validation['current_cpu_pytest']}** on the current branch |
| CUDA build and official correctness | **{validation['official_correctness']}** |
| CUDA Events and same-GPU PyTorch | **schema v2 targeted: 004/036/040 improved; 007 inconclusive; 095 exploratory** |
| External LLM smoke | **{validation['live_api_smoke']}** |
| Nsight Compute counters | **{validation['ncu_counters']}** |
| Cross-GPU rerun | **{validation['cross_gpu']}** |

| Historical v1 scope | Versus upstream (diagnostic / old strict gate) | Versus fastest PyTorch method (diagnostic / old strict gate) |
| --- | ---: | ---: |
| Nine new candidates | {_f(n9['attempted_upstream'])} / {_f(n9['strict_upstream'])} | {_f(n9['attempted_pytorch'])} / {_f(n9['strict_pytorch'])} |
| Full Core 10, including RMSNorm | {_f(a10['attempted_upstream'])} / {_f(a10['strict_upstream'])} | {_f(a10['attempted_pytorch'])} / {_f(a10['strict_pytorch'])} |

These immutable strict values remain historical v1 evidence. A separate targeted schema-v2 rerun renewed 004/036/040, classified 007 as inconclusive, and kept 095 exploratory; it is not a full Core 10 or Agent result. The new gate also checks p99/max error regression, NaN/Inf, and five-run determinism. Neither the Agent Pilot nor Core 10 search has been rerun.

{_evidence_links(context, chinese=False)}"""


def _index_block(context: dict[str, Any], *, chinese: bool) -> str:
    n9, a10 = context["new9"], context["all10"]
    if chinese:
        return f"""**更新日期：{context['status']['updated_at']}**

- Day 1–2：Provider、Recorder、Suite、dry-run 与 CPU 测试已完成。
- Day 3–7：WSL2/RTX 3080、容器、编译、正确性和 CUDA Events 基准设施已完成；历史 API 冒烟为 401，当前凭据尚未重新验证。
- Day 8–10：RMSNorm V0–V3c 已完成，独立深度结果 {_f(context['rmsnorm_deep_speedup'])}；统一 Core 10 复测 {_f(context['rmsnorm_unified_speedup'])}。
- 后续 Core 10：历史 v1 的九个新增候选全部正确，旧严格口径通过 {n9['verified']}/9；schema v2 定向复测确认 004/036/040，007 无法定论，095 保持探索。
- PyTorch 对照：完整十题严格组合为 {_f(a10['strict_pytorch'])}，总体基本持平；不稳定任务在报告中保持显式标记。

{_evidence_links(context, chinese=True, prefix='../../')}"""
    return f"""**Last updated: {context['status']['updated_at']}**

- Days 1–2: provider, recorder, suite validation, dry-run, and CPU tests are complete.
- Days 3–7: WSL2/RTX 3080, container, compilation, correctness, and CUDA Events infrastructure are complete; the historical API smoke returned 401 and the current credential has not been revalidated.
- Days 8–10: RMSNorm V0–V3c is complete, with {_f(context['rmsnorm_deep_speedup'])} in the independent deep run and {_f(context['rmsnorm_unified_speedup'])} in the unified Core 10 rerun.
- Core 10 follow-up: historical v1 validated all nine new candidates and {n9['verified']}/9 passed its old strict gate; targeted schema v2 confirmed 004/036/040, left 007 inconclusive, and kept 095 exploratory.
- PyTorch comparison: the strict full-ten ratio is {_f(a10['strict_pytorch'])}, effectively parity; unstable tasks remain explicitly labeled.

{_evidence_links(context, chinese=False, prefix='../../')}"""


def _validation_block(context: dict[str, Any], *, chinese: bool) -> str:
    validation = context["status"]["validation"]
    v2_source = context["sources"]["targeted_validation_v2"]
    if chinese:
        return f"""| 门禁 | 当前状态 | 规范证据 |
| --- | --- | --- |
| Provider/Recorder/Suite CPU 测试 | 通过 — {_validation_labels(validation, chinese=True)['current_cpu_pytest']} | `tests/` |
| 真实网关冒烟 | 未运行 — 历史 HTTP 401 尚未重新验证 | 历史 `artifacts/portfolio-v1.0/results/analysis_summary.json` |
| RTX 3080 容器与 `sm_86` 构建 | 通过 | `artifacts/portfolio-v1.0/environment/environment.json` |
| 官方候选正确性 | 历史 v1 通过 — 10/10；schema v2 定向 5/5 通过 | `{v2_source}` |
| RMSNorm 边界正确性 | 通过 | 已提交的 `edge_driver.cpp` 与深度案例 artifacts |
| CUDA Events 计时 | schema v2 定向确认：004/036/040 提升；007 无法定论；095 探索 | `{v2_source}` |
| 同卡 PyTorch 对比 | 已完成 | `pytorch_core10_rtx3080.csv` |
| NCU 硬件计数器 | 阻塞 — `ERR_NVGPUCTRPERM` | 环境清单与历史验证报告 |
| 跨 GPU 对比 | 未运行 — 延后至 Day 11–14 | 未发布性能声明 |

历史手工跟进验证了全部十个候选，并在旧门槛下改进 {context['all10']['verified']}/10。相关声明作为不可变历史证据保留。schema v2 定向结果只覆盖五题，不能外推为完整 Core 10 或 Agent 搜索结论。"""
    return f"""| Gate | Current status | Canonical evidence |
| --- | --- | --- |
| Provider/Recorder/Suite CPU tests | PASSED — {validation['current_cpu_pytest']} | `tests/` |
| Real gateway smoke | NOT RUN — historical HTTP 401 has not been revalidated | historical `artifacts/portfolio-v1.0/results/analysis_summary.json` |
| RTX 3080 container and `sm_86` build | PASSED | `artifacts/portfolio-v1.0/environment/environment.json` |
| Official candidate correctness | HISTORICAL V1 PASSED — 10/10; schema-v2 targeted 5/5 passed | `{v2_source}` |
| RMSNorm edge correctness | PASSED | committed `edge_driver.cpp` and deep-case artifacts |
| CUDA Events timing | schema-v2 targeted confirmation: 004/036/040 improved; 007 inconclusive; 095 exploratory | `{v2_source}` |
| Same-GPU PyTorch comparison | COMPLETED | `pytorch_core10_rtx3080.csv` |
| NCU hardware counters | BLOCKED — `ERR_NVGPUCTRPERM` | environment manifest and historical validation report |
| Cross-GPU comparison | NOT RUN — deferred Day 11–14 | no performance claim published |

The historical manual follow-up validated all ten candidates and improved {context['all10']['verified']}/10 under the old gate. Those claims remain immutable historical evidence. The targeted schema-v2 result covers only five tasks and must not be generalized to a full Core 10 or Agent-search claim."""


def _architecture_block(context: dict[str, Any], *, chinese: bool) -> str:
    targeted = context["targeted_v2"]["summary"]
    if chinese:
        return f"""当前实测状态（{context['status']['updated_at']}）：

- RTX 3080 / `{context['sm']}` CUDA 构建、正确性与 CUDA Events：**已完成**
- 同卡 PyTorch eager/out/fused 对比：**已完成**
- 历史 v1 手工 Core 10 严格验证提升：**{context['all10']['verified']}/10**
- Schema v2 定向确认：**{targeted['improved']} 项提升、{targeted['inconclusive']} 项无法定论、{targeted['exploratory']} 项探索中**
- LLM 在线冒烟：**尚未重跑；历史结果为 HTTP 401**；没有 Agent Core 10 搜索声明
- NCU 计数器归因：**受 `ERR_NVGPUCTRPERM` 阻塞**
- 跨 GPU 对比：**未运行；延后至 Day 11–14**

规范状态位于 `portfolio/status.json`；实测数值从已提交的对比 JSON 派生。`scripts/sync_portfolio_docs.py --check` 会拒绝过期的生成区块和失效的证据链接。"""
    return f"""Current measured state ({context['status']['updated_at']}):

- RTX 3080 / `{context['sm']}` CUDA build, correctness, and CUDA Events: **completed**
- Same-GPU PyTorch eager/out/fused comparison: **completed**
- Historical v1 manual Core 10 strict verified improvements: **{context['all10']['verified']}/10**
- Targeted schema-v2 confirmation: **{targeted['improved']} improved, {targeted['inconclusive']} inconclusive, {targeted['exploratory']} exploratory**
- LLM live smoke: **not rerun; historical result was HTTP 401**; no Agent Core 10 search claim
- NCU counter attribution: **blocked by `ERR_NVGPUCTRPERM`**
- Cross-GPU comparison: **not run; deferred Day 11–14**

Canonical status lives in `portfolio/status.json`; measured values are derived from the checked-in comparison JSON. `scripts/sync_portfolio_docs.py --check` rejects stale generated blocks and broken evidence links."""


def _rmsnorm_block(context: dict[str, Any], *, chinese: bool) -> str:
    v2_rmsnorm = next(
        row for row in context["targeted_v2_results"] if row["task_id"] == "036"
    )
    v2_speedup = float(v2_rmsnorm["performance_gate"]["median_speedup"])
    if chinese:
        return f"""当前状态：**已在 {context['gpu_name']}（{context['sm']}）上验证**。

- V0–V3c 及基准规范化后的源码均通过官方 Driver。
- V1–V3c 也通过 `edge_driver.cpp` 中的小尺寸、奇数空间尺寸、63/64/65 通道和边界用例。
- 独立 V3c 深度测试相对配对 V0 测得 {_f(context['rmsnorm_deep_speedup'])}。
- 后续统一 Core 10 复测测得 {_f(context['rmsnorm_unified_speedup'])}；该差异作为跨运行证据保留，不做静默平均。
- Schema v2 的五 Session 定向确认测得配对中位加速 {_f(v2_speedup)}，并通过 bootstrap 与稳定性门槛。
- NCU 硬件计数器归因仍受 `ERR_NVGPUCTRPERM` 阻塞；CUDA Events 与源码推导的映射证据分开报告。"""
    return f"""Current status: **validated on {context['gpu_name']} ({context['sm']})**.

- V0–V3c and the benchmark-normalized sources pass the official Driver.
- V1–V3c also pass the small, odd-spatial, 63/64/65-channel, and boundary cases in `edge_driver.cpp`.
- The independent V3c deep run measured {_f(context['rmsnorm_deep_speedup'])} versus its paired V0.
- The later unified Core 10 rerun measured {_f(context['rmsnorm_unified_speedup'])}; the difference is retained as run-to-run evidence, not silently averaged.
- The targeted five-Session schema-v2 confirmation measured {_f(v2_speedup)} median paired speedup and passed the bootstrap and stability gates.
- NCU hardware-counter attribution remains blocked by `ERR_NVGPUCTRPERM`; CUDA Events and code-derived mapping evidence are reported separately."""


RENDERERS: dict[str, Callable[[dict[str, Any]], str]] = {
    "root_en": lambda context: _root_block(context, chinese=False),
    "root_zh": lambda context: _root_block(context, chinese=True),
    "index_en": lambda context: _index_block(context, chinese=False),
    "index_zh": lambda context: _index_block(context, chinese=True),
    "validation_en": lambda context: _validation_block(context, chinese=False),
    "validation_zh": lambda context: _validation_block(context, chinese=True),
    "architecture_en": lambda context: _architecture_block(context, chinese=False),
    "architecture_zh": lambda context: _architecture_block(context, chinese=True),
    "rmsnorm_en": lambda context: _rmsnorm_block(context, chinese=False),
    "rmsnorm_zh": lambda context: _rmsnorm_block(context, chinese=True),
}


def replace_block(text: str, marker: str, body: str) -> str:
    start = START_TEMPLATE.format(marker=marker)
    end = END_TEMPLATE.format(marker=marker)
    if text.count(start) != 1 or text.count(end) != 1:
        raise DocumentationSyncError(
            f"Expected exactly one {start!r} and {end!r} marker pair."
        )
    start_index = text.index(start) + len(start)
    end_index = text.index(end)
    if start_index > end_index:
        raise DocumentationSyncError(f"Markers are reversed for {marker}.")
    normalized = body.strip()
    return text[:start_index] + "\n" + normalized + "\n" + text[end_index:]


def expected_documents(root: Path, context: dict[str, Any]) -> dict[Path, str]:
    expected: dict[Path, str] = {}
    for relative, marker, renderer_name in TARGETS:
        path = root / relative
        if not path.is_file():
            raise DocumentationSyncError(f"Documentation target is missing: {relative}")
        current = path.read_text(encoding="utf-8")
        expected[relative] = replace_block(current, marker, RENDERERS[renderer_name](context))
    return expected


def _markdown_links(text: str) -> list[str]:
    return [match.group(1).strip() for match in re.finditer(r"!?\[[^\]]*\]\(([^)]+)\)", text)]


def validate_links(root: Path, documents: dict[Path, str]) -> None:
    for relative, text in documents.items():
        for target in _markdown_links(text):
            cleaned = target.strip("<>").split("#", 1)[0]
            if not cleaned or cleaned.startswith(("http://", "https://", "mailto:")):
                continue
            resolved = (root / relative.parent / cleaned).resolve()
            if not resolved.exists():
                raise DocumentationSyncError(f"Broken link in {relative}: {target}")
        for pattern in MACHINE_PATH_PATTERNS:
            if pattern.search(text):
                raise DocumentationSyncError(
                    f"Machine-specific absolute path found in {relative}: {pattern.pattern}"
                )


def validate_artifact_hashes(root: Path, context: dict[str, Any]) -> None:
    failures: list[str] = []
    manifest_sources = {
        name: value
        for name, value in context["sources"].items()
        if name.startswith("artifact_sha256")
    }
    for source_name, relative_manifest in manifest_sources.items():
        manifest_path = root / relative_manifest
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        artifact_root = manifest_path.parent
        for relative, expected in manifest.items():
            path = artifact_root / relative
            if not path.is_file():
                failures.append(f"{source_name}:missing:{relative}")
                continue
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                failures.append(f"{source_name}:sha256:{relative}")
    if failures:
        raise DocumentationSyncError(
            "Artifact SHA256 verification failed: " + ", ".join(failures[:10])
        )


def changed_files(root: Path, base_ref: str) -> list[str]:
    completed = subprocess.run(
        ["git", "diff", "--name-only", f"{base_ref}...HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise DocumentationSyncError(
            f"Cannot compare documentation with {base_ref}: {completed.stderr.strip()}"
        )
    return [line.strip().replace("\\", "/") for line in completed.stdout.splitlines() if line.strip()]


def validate_change_policy(paths: list[str]) -> None:
    relevant = [path for path in paths if path.startswith(RELEVANT_PREFIXES)]
    documented = [path for path in paths if path.startswith(DOCUMENTATION_PREFIXES)]
    if relevant and not documented:
        raise DocumentationSyncError(
            "Benchmark/candidate/artifact changes require README/docs or portfolio/status.json "
            f"in the same change. Relevant files include: {relevant[:5]}"
        )


def synchronize(
    *,
    root: Path = ROOT_DIR,
    write: bool,
    base_ref: str | None = None,
) -> list[Path]:
    context = load_context(root)
    expected = expected_documents(root, context)
    validate_links(root, expected)
    validate_artifact_hashes(root, context)
    changed: list[Path] = []
    for relative, wanted in expected.items():
        path = root / relative
        current = path.read_text(encoding="utf-8")
        if current == wanted:
            continue
        changed.append(relative)
        if write:
            path.write_text(wanted, encoding="utf-8", newline="\n")
    if not write and changed:
        joined = ", ".join(str(path) for path in changed)
        raise DocumentationSyncError(
            f"Documentation is stale: {joined}. Run scripts/sync_portfolio_docs.py --write."
        )
    if base_ref:
        validate_change_policy(changed_files(root, base_ref))
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="Rewrite generated Markdown blocks.")
    mode.add_argument("--check", action="store_true", help="Fail when generated blocks are stale.")
    parser.add_argument(
        "--base-ref",
        help="Optional Git base ref used to require docs in benchmark/result changes.",
    )
    args = parser.parse_args()
    if args.write and args.base_ref:
        parser.error("--base-ref is only valid with --check.")
    try:
        changed = synchronize(write=args.write, base_ref=args.base_ref)
    except DocumentationSyncError as error:
        print(f"docs-sync: {error}", file=sys.stderr)
        return 2
    action = "updated" if args.write else "verified"
    print(f"docs-sync: {action}; changed={len(changed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
