from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any

import kmapper as km
import numpy as np
import pandas as pd


SIZE_COLORSCALE = [
    [0.0, "rgb(68, 1, 84)"],
    [0.1, "rgb(72, 35, 116)"],
    [0.2, "rgb(64, 67, 135)"],
    [0.3, "rgb(52, 94, 141)"],
    [0.4, "rgb(41, 120, 142)"],
    [0.5, "rgb(32, 144, 140)"],
    [0.6, "rgb(34, 167, 132)"],
    [0.7, "rgb(68, 190, 112)"],
    [0.8, "rgb(121, 209, 81)"],
    [0.9, "rgb(189, 222, 38)"],
    [1.0, "rgb(253, 231, 36)"],
]


JSON_COLUMNS = [
    "corpus_counts",
    "lexical_categories",
    "top_words",
    "distinctive_words",
    "top_word_contexts",
    "distinctive_word_contexts",
    "top_features",
    "top_features_by_prevalence",
    "topic_evidence_words",
    # Старое поле из прошлой версии статистик:
    # "top_sae_features",
]


def parse_json_cell(value: Any, fallback: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str) or not value:
        return fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return fallback


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    return value


def fmt_float(value: Any, default: float = 0.0) -> float:
    if value is None or pd.isna(value):
        return default
    return float(value)


def fmt_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    if pd.isna(value):
        return default
    return bool(value)


def words_for_hover(items: dict[str, int] | list[dict[str, Any]], limit: int = 12) -> str:
    if isinstance(items, dict):
        return ", ".join(str(key) for key in list(items.keys())[:limit])

    return ", ".join(str(item.get("word", "")) for item in items[:limit])


def feature_for_hover(items: list[dict[str, Any]], limit: int = 10) -> list[str]:
    parts = []
    for item in items[:limit]:
        feature = item.get("feature")
        prevalence = fmt_float(item.get("prevalence"))
        activation = fmt_float(item.get("abs_total_activations", item.get("total_activation")))
        parts.append(f"{feature} p={prevalence:.2f} L1={activation:.1f}")

    return parts


def node_stats_payload(nodes: pd.DataFrame) -> dict[str, Any]:
    sizes = nodes["n_points"].astype(float)
    min_log = float(np.log1p(sizes.min()))
    max_log = float(np.log1p(sizes.max()))

    payload = {}
    for _, row in nodes.iterrows():
        top_words = parse_json_cell(row.get("top_words", ""), {})
        distinctive = parse_json_cell(row.get("distinctive_words", ""), [])
        top_word_contexts = parse_json_cell(row.get("top_word_contexts", ""), {})
        distinctive_word_contexts = parse_json_cell(row.get("distinctive_word_contexts", ""), {})
        corpus_counts = parse_json_cell(row.get("corpus_counts", ""), {})
        lexical_categories = parse_json_cell(row.get("lexical_categories", ""), {})
        top_features = parse_json_cell(row.get("top_features", ""), [])
        top_features_by_prevalence = parse_json_cell(
            row.get("top_features_by_prevalence", ""),
            [],
        )
        topic_evidence_words = parse_json_cell(row.get("topic_evidence_words", ""), [])

        # Старое поле из прошлой версии статистик:
        # top_sae_features = parse_json_cell(row.get("top_sae_features", ""), [])

        n_points = float(row["n_points"])
        if max_log <= min_log:
            color_value = 1.0
        else:
            color_value = (float(np.log1p(n_points)) - min_log) / (max_log - min_log)

        payload[str(row["node_id"])] = {
            "node_id": str(row["node_id"]),
            "color_value": color_value,
            "n_points": int(row["n_points"]),
            "component_id": int(row.get("component_id", -1)),
            "component_size": int(row.get("component_size", 0)),
            "degree": int(row.get("degree", 0)),
            "weighted_degree": fmt_float(row.get("weighted_degree")),
            "intersection_degree_ratio": fmt_float(row.get("intersection_degree_ratio")),
            "neighbor_overlap_mean": fmt_float(row.get("neighbor_overlap_mean")),
            "neighbor_overlap_max": fmt_float(row.get("neighbor_overlap_max")),
            "neighbor_overlap_share_mean": fmt_float(row.get("neighbor_overlap_share_mean")),
            "dominant_corpus": str(row["dominant_corpus"]),
            "dominant_corpus_share": fmt_float(row["dominant_corpus_share"]),
            "corpus_entropy_norm": fmt_float(row["corpus_entropy_norm"]),
            "doc_entropy_norm": fmt_float(row["doc_entropy_norm"]),
            "word_entropy_norm": fmt_float(row.get("word_entropy_norm")),
            "unique_docs": int(row["unique_docs"]),
            "corpus_counts": corpus_counts,
            "lexical_categories": lexical_categories,
            "top_words": words_for_hover(top_words, 15),
            "distinctive_words": words_for_hover(distinctive, 15),
            "top_word_contexts": top_word_contexts,
            "distinctive_word_contexts": distinctive_word_contexts,
            "topic_label": str(row.get("topic_label", "")) if not pd.isna(row.get("topic_label", "")) else "",
            "topic_description": str(row.get("topic_description", "")) if not pd.isna(row.get("topic_description", "")) else "",
            "topic_type": str(row.get("topic_type", "unclear")) if not pd.isna(row.get("topic_type", "unclear")) else "unclear",
            "topic_confidence": fmt_float(row.get("topic_confidence")),
            "topic_evidence_words": ", ".join(str(x) for x in topic_evidence_words[:20]),
            "topic_possible_artifact": fmt_bool(row.get("topic_possible_artifact", False)),
            "sae_norm_entropy": fmt_float(row.get("sae_norm_entropy")),
            "sae_active_features": int(row.get("sae_active_features", 0)),
            "top_features": feature_for_hover(top_features, 12),
            "top_features_by_prevalence": feature_for_hover(top_features_by_prevalence, 12),
            # Старое поле панели:
            # "top_sae_features": sae_for_hover(top_sae_features, 12),
            "rec_err_relative_mean": fmt_float(row.get("rec_err_relative_mean")),
            "rec_err_relative_median": fmt_float(row.get("rec_err_relative_median")),
            "rec_err_relative_std": fmt_float(row.get("rec_err_relative_std")),
            "sae_L0_mean": fmt_float(row.get("sae_L0_mean")),
            "sae_L0_median": fmt_float(row.get("sae_L0_median")),
            "sae_L0_std": fmt_float(row.get("sae_L0_std")),
            "sae_L1_mean": fmt_float(row.get("sae_L1_mean")),
            "sae_L1_median": fmt_float(row.get("sae_L1_median")),
            "sae_L1_std": fmt_float(row.get("sae_L1_std")),
            "sae_L2_mean": fmt_float(row.get("sae_L2_mean")),
            "sae_L2_median": fmt_float(row.get("sae_L2_median")),
            "sae_L2_std": fmt_float(row.get("sae_L2_std")),
            "sae_max_mean": fmt_float(row.get("sae_max_mean")),
            "sae_max_median": fmt_float(row.get("sae_max_median")),
            "act_L2_mean": fmt_float(row.get("act_L2_mean")),
            "feat_L2_mean": fmt_float(row.get("feat_L2_mean")),
        }

    return payload


def lens_matrix(lens: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    if {"umap_0", "umap_1"}.issubset(lens.columns):
        columns = ["umap_0", "umap_1"]
    else:
        columns = sorted(
            [col for col in lens.columns if col.startswith("lens_dim_")],
            key=lambda col: int(col.rsplit("_", 1)[-1]),
        )[:2]

    if len(columns) < 2:
        raise ValueError("Need at least two lens columns for native Kepler visualization")

    return lens[columns].to_numpy(), columns


def strip_native_member_tooltips(html_text: str) -> str:
    marker = "const graph = "
    start = html_text.find(marker)
    if start < 0:
        return html_text

    payload_start = start + len(marker)
    payload_end = html_text.find(";\n", payload_start)
    if payload_end < 0:
        return html_text

    graph_payload = json.loads(html_text[payload_start:payload_end])
    for node in graph_payload.get("nodes", []):
        tooltip = node.get("tooltip")
        if tooltip is not None:
            tooltip["custom_tooltips"] = []

    graph_json = json.dumps(
        graph_payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).replace("</", "<\\/")
    return html_text[:payload_start] + graph_json + html_text[payload_end:]


def kepler_native_injection(stats: dict[str, Any]) -> str:
    jsonable_stats = to_jsonable(stats)
    node_colors = {
        node_id: stat.get("color_value", 0.0)
        for node_id, stat in jsonable_stats.items()
    }
    node_stat_ids = {}
    node_stat_scripts = []

    for index, (node_id, stat) in enumerate(jsonable_stats.items()):
        script_id = f"codex-node-stat-{index}"
        node_stat_ids[node_id] = script_id
        stat_json = json.dumps(
            stat,
            ensure_ascii=False,
            separators=(",", ":"),
        ).replace("</", "<\\/")
        node_stat_scripts.append(
            f'<script type="application/json" id="{script_id}">{stat_json}</script>'
        )

    colors_json = json.dumps(
        node_colors,
        ensure_ascii=False,
        separators=(",", ":"),
    ).replace("</", "<\\/")
    colorscale_json = json.dumps(
        SIZE_COLORSCALE,
        ensure_ascii=False,
        separators=(",", ":"),
    ).replace("</", "<\\/")
    stat_ids_json = json.dumps(
        node_stat_ids,
        ensure_ascii=False,
        separators=(",", ":"),
    ).replace("</", "<\\/")
    stat_scripts_html = "\n".join(node_stat_scripts)
    colorbar_gradient = """
      linear-gradient(
        to bottom,
        rgb(253, 231, 36) 0%,
        rgb(189, 222, 38) 10%,
        rgb(121, 209, 81) 20%,
        rgb(68, 190, 112) 30%,
        rgb(34, 167, 132) 40%,
        rgb(32, 144, 140) 50%,
        rgb(41, 120, 142) 60%,
        rgb(52, 94, 141) 70%,
        rgb(64, 67, 135) 80%,
        rgb(72, 35, 116) 90%,
        rgb(68, 1, 84) 100%
      )
    """.strip()

    return f"""
{stat_scripts_html}
<div id="codex-colorbar">
  <div class="codex-colorbar-title">log node size</div>
  <div class="codex-colorbar-body">
    <div class="codex-colorbar-labels">
      <div>1.0</div>
      <div>0.0</div>
    </div>
    <div class="codex-colorbar-gradient"></div>
  </div>
  <div class="codex-colorbar-subtitle">normalized</div>
</div>
<script>
  const codexNodeColors = {colors_json};
  const codexNodeSizeColorScale = {colorscale_json};
  const codexNodeStatElementIds = {stat_ids_json};
  const codexNodeStatsCache = Object.create(null);

  function codexParseRgb(rgbText) {{
    const match = String(rgbText || "").match(/rgb\\s*\\(\\s*(\\d+)\\s*,\\s*(\\d+)\\s*,\\s*(\\d+)\\s*\\)/i);
    if (!match) return [0, 0, 0];
    return [Number(match[1]), Number(match[2]), Number(match[3])];
  }}

  function codexColorFromScale(value) {{
    let number = Number(value);
    if (!Number.isFinite(number)) number = 0;
    number = Math.max(0, Math.min(1, number));

    for (let i = 0; i < codexNodeSizeColorScale.length - 1; i += 1) {{
      const left = codexNodeSizeColorScale[i];
      const right = codexNodeSizeColorScale[i + 1];
      if (number >= left[0] && number <= right[0]) {{
        const span = right[0] - left[0];
        const t = span > 0 ? (number - left[0]) / span : 0;
        const leftRgb = codexParseRgb(left[1]);
        const rightRgb = codexParseRgb(right[1]);
        const r = Math.round(leftRgb[0] + (rightRgb[0] - leftRgb[0]) * t);
        const g = Math.round(leftRgb[1] + (rightRgb[1] - leftRgb[1]) * t);
        const b = Math.round(leftRgb[2] + (rightRgb[2] - leftRgb[2]) * t);
        return `rgb(${{r}}, ${{g}}, ${{b}})`;
      }}
    }}

    return codexNodeSizeColorScale[codexNodeSizeColorScale.length - 1][1];
  }}

  function codexApplyNodeSizeColors() {{
    d3.selectAll(".node").each(function(d) {{
      const nodeId = d && d.tooltip ? (d.tooltip.node_id || d.name) : (d && d.name);
      const colorValue = codexNodeColors[nodeId];
      if (!Number.isFinite(Number(colorValue))) return;

      const nodeSelection = d3.select(this);
      let circle = nodeSelection.select("circle");
      if (circle.empty()) {{
        circle = nodeSelection.select(".circle");
      }}
      if (!circle.empty()) {{
        circle
          .style("fill", codexColorFromScale(colorValue))
          .attr("fill", codexColorFromScale(colorValue));
      }}
    }});
  }}

  function codexGetNodeStat(nodeId) {{
    if (!nodeId) return null;
    if (Object.prototype.hasOwnProperty.call(codexNodeStatsCache, nodeId)) {{
      return codexNodeStatsCache[nodeId];
    }}

    const elementId = codexNodeStatElementIds[nodeId];
    if (!elementId) {{
      codexNodeStatsCache[nodeId] = null;
      return null;
    }}

    const element = document.getElementById(elementId);
    if (!element) {{
      codexNodeStatsCache[nodeId] = null;
      return null;
    }}

    try {{
      codexNodeStatsCache[nodeId] = JSON.parse(element.textContent);
    }} catch (error) {{
      console.error("Failed to parse node statistics", nodeId, error);
      codexNodeStatsCache[nodeId] = null;
    }}
    return codexNodeStatsCache[nodeId];
  }}

  function codexEsc(value) {{
    return String(value ?? "").replace(/[&<>"']/g, ch => ({{"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}}[ch]));
  }}

  function codexFmt(value, digits = 3) {{
    const number = Number(value);
    if (!Number.isFinite(number)) return "n/a";
    return number.toFixed(digits);
  }}

  function codexPills(text, emptyLabel = "none after TRASH_WORDS filter") {{
    const items = String(text || "").split(",").map(x => x.trim()).filter(Boolean);
    if (!items.length) {{
      return `<span class="codex-muted">${{codexEsc(emptyLabel)}}</span>`;
    }}
    return items.map(x => `<span class="codex-pill">${{codexEsc(x)}}</span>`).join("");
  }}

  function codexMetric(label, value) {{
    return `<div class="codex-metric"><div class="codex-k">${{codexEsc(label)}}</div><div class="codex-v">${{codexEsc(value)}}</div></div>`;
  }}

  function codexContextList(values) {{
    if (!values || !values.length) return `<p class="codex-small">n/a</p>`;
    return `<div class="codex-small">${{values.map(v => `<div>${{codexEsc(v)}}</div>`).join("")}}</div>`;
  }}

  function codexFeatureLines(value) {{
    let items = [];
    if (Array.isArray(value)) {{
      items = value;
    }} else if (typeof value === "string" && value.trim()) {{
      items = value.split(/,\\s*(?=\\d+\\s+p=|feature\\s+)/);
    }}

    items = items.map(item => String(item || "").trim()).filter(Boolean);
    if (!items.length) return `<p class="codex-small">n/a</p>`;

    return `
      <div class="codex-feature-list">
        ${{items.map(item => `<div class="codex-feature-line">${{codexEsc(item)}}</div>`).join("")}}
      </div>
    `;
  }}

  function codexWordContexts(contextsByWord, maxWords = 8, maxExamples = 2) {{
    if (!contextsByWord || typeof contextsByWord !== "object") return `<p class="codex-small">n/a</p>`;
    const entries = Object.entries(contextsByWord).slice(0, maxWords);
    if (!entries.length) return `<p class="codex-small">n/a</p>`;

    return entries.map(([word, examples]) => {{
      const rows = Array.isArray(examples) ? examples.slice(0, maxExamples) : [];
      const renderedRows = rows.map(example => `
        <div class="codex-context-row">
          <div><b>original word:</b> ${{codexEsc(example && example.word ? example.word : "")}}</div>
          <div><b>context:</b> ${{codexEsc(example && example.context ? example.context : "")}}</div>
        </div>
      `).join("");
      return `
        <div class="codex-word-context">
          <div class="codex-word-context-title">${{codexEsc(word)}}</div>
          ${{renderedRows || `<div class="codex-muted">n/a</div>`}}
        </div>
      `;
    }}).join("");
  }}

  function codexStatsHtml(stat) {{
    if (!stat) return "";
    const hasTopic = Boolean(
      stat.topic_label ||
      stat.topic_description ||
      stat.topic_evidence_words
    );
    const topicBlock = hasTopic ? `
        <div class="codex-topic-block">
          <h4>Node Topic</h4>
          <div class="codex-metric-grid">
            ${{codexMetric("label", stat.topic_label || "n/a")}}
            ${{codexMetric("type", stat.topic_type || "unclear")}}
            ${{codexMetric("confidence", codexFmt(stat.topic_confidence, 2))}}
            ${{codexMetric("possible artifact", stat.topic_possible_artifact ? "true" : "false")}}
          </div>
          <p class="codex-small">${{codexEsc(stat.topic_description || "n/a")}}</p>
          <h4>Evidence words</h4>
          <div class="codex-pill-row">${{codexPills(stat.topic_evidence_words, "no evidence words")}}</div>
        </div>
    ` : "";

    return `
      <div id="codex-node-stats">
        <hr><br>
        <h3>Node Statistics</h3>
        ${{topicBlock}}

        <div class="codex-metric-grid">
          ${{codexMetric("points", stat.n_points)}}
          ${{codexMetric("component id", stat.component_id)}}
          ${{codexMetric("component size", stat.component_size)}}
          ${{codexMetric("degree", stat.degree)}}
          ${{codexMetric("weighted degree", codexFmt(stat.weighted_degree, 1))}}
          ${{codexMetric("intersection / degree", codexFmt(stat.intersection_degree_ratio, 3))}}
          ${{codexMetric("dominant corpus", `${{stat.dominant_corpus}} (${{codexFmt(stat.dominant_corpus_share, 3)}})`)}}
          ${{codexMetric("corpus entropy", codexFmt(stat.corpus_entropy_norm, 3))}}
          ${{codexMetric("doc entropy", codexFmt(stat.doc_entropy_norm, 3))}}
          ${{codexMetric("word entropy filtered", codexFmt(stat.word_entropy_norm, 3))}}
          ${{codexMetric("SAE entropy", codexFmt(stat.sae_norm_entropy, 3))}}
          ${{codexMetric("active SAE", stat.sae_active_features)}}
          ${{codexMetric("overlap mean", codexFmt(stat.neighbor_overlap_mean, 2))}}
          ${{codexMetric("overlap max", codexFmt(stat.neighbor_overlap_max, 0))}}
          ${{codexMetric("overlap share", codexFmt(stat.neighbor_overlap_share_mean, 3))}}
          ${{codexMetric("rec err mean", codexFmt(stat.rec_err_relative_mean, 3))}}
          ${{codexMetric("SAE L0 mean", codexFmt(stat.sae_L0_mean, 1))}}
          ${{codexMetric("SAE L2 mean", codexFmt(stat.sae_L2_mean, 1))}}
        </div>

        <details class="codex-details">
          <summary>Words</summary>
          <h4>Corpus counts</h4>
          <p class="codex-small">${{codexEsc(JSON.stringify(stat.corpus_counts))}}</p>
          <h4>Lexical categories</h4>
          <p class="codex-small">${{codexEsc(JSON.stringify(stat.lexical_categories))}}</p>
          <h4>Distinctive words (filtered)</h4>
          <div class="codex-pill-row">${{codexPills(stat.distinctive_words)}}</div>
          <h4>Top words (filtered)</h4>
          <div class="codex-pill-row">${{codexPills(stat.top_words)}}</div>
          <h4>Contexts for distinctive words</h4>
          ${{codexWordContexts(stat.distinctive_word_contexts, 8, 2)}}
          <h4>Contexts for top words</h4>
          ${{codexWordContexts(stat.top_word_contexts, 8, 2)}}
        </details>

        <details class="codex-details">
          <summary>SAE details</summary>
          <h4>Top SAE features by activation</h4>
          ${{codexFeatureLines(stat.top_features)}}
          <h4>Top SAE features by prevalence</h4>
          ${{codexFeatureLines(stat.top_features_by_prevalence)}}
          <!-- Старое поле из прошлой версии:
          <h4>Top SAE features</h4>
          <p class="codex-small">${{codexEsc(stat.top_sae_features)}}</p>
          -->

          <h4>Activation / SAE</h4>
          <p class="codex-small">
            act_L2 mean: ${{codexFmt(stat.act_L2_mean, 2)}}<br>
            feat_L2 mean: ${{codexFmt(stat.feat_L2_mean, 2)}}<br>
            rec err mean / median / std: ${{codexFmt(stat.rec_err_relative_mean, 3)}} / ${{codexFmt(stat.rec_err_relative_median, 3)}} / ${{codexFmt(stat.rec_err_relative_std, 3)}}<br>
            SAE L0 mean / median / std: ${{codexFmt(stat.sae_L0_mean, 1)}} / ${{codexFmt(stat.sae_L0_median, 1)}} / ${{codexFmt(stat.sae_L0_std, 1)}}<br>
            SAE L1 mean / median / std: ${{codexFmt(stat.sae_L1_mean, 1)}} / ${{codexFmt(stat.sae_L1_median, 1)}} / ${{codexFmt(stat.sae_L1_std, 1)}}<br>
            SAE L2 mean / median / std: ${{codexFmt(stat.sae_L2_mean, 1)}} / ${{codexFmt(stat.sae_L2_median, 1)}} / ${{codexFmt(stat.sae_L2_std, 1)}}<br>
            SAE max mean / median: ${{codexFmt(stat.sae_max_mean, 1)}} / ${{codexFmt(stat.sae_max_median, 1)}}
          </p>
        </details>

      </div>
    `;
  }}

  function codexRenderNodeStats(d) {{
    const nodeId = d && d.tooltip ? (d.tooltip.node_id || d.name) : "";
    const old = document.getElementById("codex-node-stats");
    if (old && old.dataset.nodeId === nodeId) return;
    if (old) old.remove();
    if (!d || !d.tooltip) return;
    const stat = codexGetNodeStat(nodeId);
    const details = document.querySelector("#tooltip_content_focus_node .details");
    if (details && stat) {{
      details.insertAdjacentHTML("afterbegin", codexStatsHtml(stat));
      const inserted = document.getElementById("codex-node-stats");
      if (inserted) inserted.dataset.nodeId = nodeId;
    }}
    codexRemoveNativeMemberDistribution();
    codexRemoveNativeKeplerStats();
  }}

  function codexRemoveNativeMemberDistribution() {{
    const root = document.querySelector("#tooltip_content_focus_node .details");
    if (!root) return;

    const candidates = root.querySelectorAll("div, section, details, table");
    candidates.forEach(el => {{
      if (el.closest("#codex-node-stats")) return;
      const text = (el.textContent || "").toLowerCase();
      if (text.includes("member distribution")) {{
        el.style.display = "none";
      }}
    }});
  }}

  function codexRemoveNativeKeplerStats() {{
    const root = document.querySelector("#tooltip_content_focus_node .details");
    if (!root) return;

    const candidates = Array.from(root.querySelectorAll("div, section, details, table"))
      .filter(el => {{
        if (el.closest("#codex-node-stats") || el.querySelector("#codex-node-stats")) return false;
        const text = (el.textContent || "").toLowerCase();
        return text.includes("projection statistics") || text.includes("cluster statistics");
      }});

    candidates.forEach(el => {{
      if (candidates.some(other => other !== el && el.contains(other))) return;
      if (el.closest("#codex-node-stats") || el.querySelector("#codex-node-stats")) return;
      el.style.display = "none";
    }});
  }}

  graph.nodes.forEach(node => {{
    const colorValue = codexNodeColors[node.name];
    if (Number.isFinite(Number(colorValue))) {{
      node.color = [[Number(colorValue)]];
    }}
    if (node.tooltip) {{
      node.tooltip.custom_tooltips = [];
    }}
  }});
</script>
<style>
  html,
  body {{
    overflow-x: hidden !important;
  }}
  #tooltip_content,
  #tooltip_content_focus_node,
  #tooltip_content_focus_node .details,
  #codex-node-stats {{
    max-width: min(560px, calc(100vw - 32px));
    box-sizing: border-box;
    overflow-x: hidden;
  }}
  #tooltip_content_focus_node *,
  #codex-node-stats * {{
    box-sizing: border-box;
    max-width: 100%;
  }}
  #tooltip_content_focus_node,
  #tooltip_content_focus_node .details,
  #codex-node-stats {{
    contain: layout paint style;
    overflow-wrap: anywhere;
    word-break: break-word;
  }}
  #codex-node-stats {{
    content-visibility: auto;
    contain-intrinsic-size: 900px;
  }}
  body.codex-map-moving #tooltip_content {{
    visibility: hidden;
    pointer-events: none;
  }}
  .highlight,
  .node.highlight,
  .node .circle.highlight {{
    filter: none !important;
  }}
  #display .node.highlight .circle,
  #display .node .circle.highlight,
  .node.highlight .circle,
  .node .circle.highlight {{
    stroke: #f2d16b !important;
    stroke-opacity: 0.95 !important;
    stroke-width: 9px !important;
  }}
  #codex-node-stats h3 {{ margin-top: 0; }}
  #codex-node-stats h4 {{ margin: 12px 0 5px; font-size: 1em; }}
  #codex-colorbar {{
    position: fixed;
    right: 16px;
    top: 50%;
    transform: translateY(-50%);
    z-index: 10000;
    pointer-events: none;
    color: rgba(255,255,255,0.92);
    background: rgba(18,18,24,0.78);
    border: 1px solid rgba(255,255,255,0.22);
    border-radius: 6px;
    padding: 8px;
    font-size: 11px;
    line-height: 1.2;
    box-shadow: 0 2px 12px rgba(0,0,0,0.28);
  }}
  .codex-colorbar-title {{
    font-weight: 700;
    margin-bottom: 6px;
    white-space: nowrap;
  }}
  .codex-colorbar-body {{
    display: flex;
    align-items: stretch;
    gap: 5px;
  }}
  .codex-colorbar-labels {{
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    height: 160px;
    text-align: right;
    opacity: 0.9;
  }}
  .codex-colorbar-gradient {{
    width: 16px;
    height: 160px;
    border-radius: 6px;
    border: 1px solid rgba(255,255,255,0.35);
    box-sizing: border-box;
    overflow: hidden;
    background-clip: padding-box;
    background: {colorbar_gradient};
  }}
  .codex-colorbar-subtitle {{
    margin-top: 5px;
    opacity: 0.72;
    white-space: nowrap;
  }}
  .codex-topic-block {{
    border: 1px solid rgba(255,255,255,0.18);
    background: rgba(255,255,255,0.05);
    padding: 8px;
    margin: 8px 0 12px;
  }}
  .codex-topic-block h4:first-child {{ margin-top: 0; }}
  .codex-details {{
    border-top: 1px solid rgba(255,255,255,0.14);
    margin-top: 10px;
    padding-top: 8px;
  }}
  .codex-details summary {{
    cursor: pointer;
    font-weight: 700;
  }}
  .codex-metric-grid {{
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 6px;
    margin: 8px 0 10px;
  }}
  .codex-metric {{
    border: 1px solid rgba(255,255,255,0.20);
    padding: 6px;
    background: rgba(255,255,255,0.06);
  }}
  .codex-k {{ font-size: 0.78em; opacity: 0.75; }}
  .codex-v {{ font-weight: 700; overflow-wrap: anywhere; }}
  .codex-pill-row {{ display: flex; flex-wrap: wrap; gap: 4px; }}
  #tooltip_content_focus_node .membership-information,
  #tooltip_content_focus_node .member-distribution,
  #tooltip_content_focus_node .histogram,
  #tooltip_content_focus_node .histogram-container,
  #tooltip_content_focus_node .color_function,
  #tooltip_content_focus_node .color-function,
  #tooltip_content_focus_node .distribution,
  #tooltip_content_focus_node .distribution-container,
  #tooltip_content_focus_node .projection-statistics,
  #tooltip_content_focus_node .projection_statistics,
  #tooltip_content_focus_node .cluster-statistics,
  #tooltip_content_focus_node .cluster_statistics {{
    display: none !important;
  }}
  .codex-muted {{
    opacity: 0.65;
    font-size: 0.84em;
    font-style: italic;
  }}
  .codex-pill {{
    display: inline-block;
    border: 1px solid rgba(255,255,255,0.25);
    padding: 2px 5px;
    border-radius: 999px;
    font-size: 0.82em;
    background: rgba(255,255,255,0.08);
  }}
  .codex-word-context {{
    border-top: 1px solid rgba(255,255,255,0.12);
    margin-top: 8px;
    padding-top: 6px;
  }}
  .codex-word-context-title {{
    font-weight: 700;
    margin-bottom: 4px;
  }}
  .codex-context-row {{
    font-size: 0.86em;
    line-height: 1.45;
    margin: 0 0 6px 10px;
    overflow-wrap: anywhere;
    word-break: break-word;
  }}
  .codex-feature-list {{
    font-size: 0.86em;
    line-height: 1.45;
  }}
  .codex-feature-line {{
    margin: 0 0 3px 0;
    overflow-wrap: anywhere;
    word-break: break-word;
  }}
  .codex-small {{
    font-size: 0.86em;
    line-height: 1.45;
    overflow-wrap: anywhere;
    word-break: break-word;
  }}
</style>
<script>
  const codexOriginalSetFocusNode = set_focus_node;
  let codexRenderTimer = null;
  let codexMapMovingTimer = null;

  function codexMarkMapMoving() {{
    document.body.classList.add("codex-map-moving");
    if (codexMapMovingTimer) {{
      clearTimeout(codexMapMovingTimer);
    }}
    codexMapMovingTimer = setTimeout(function() {{
      document.body.classList.remove("codex-map-moving");
      codexMapMovingTimer = null;
    }}, 180);
  }}

  function codexScheduleNodeStats(d) {{
    if (codexRenderTimer) {{
      clearTimeout(codexRenderTimer);
      codexRenderTimer = null;
    }}

    if (!d || !d.tooltip) {{
      codexRenderNodeStats(null);
      return;
    }}

    const nodeId = d.tooltip.node_id || d.name;
    const old = document.getElementById("codex-node-stats");
    if (old && old.dataset.nodeId === nodeId) return;

    codexRenderTimer = setTimeout(function() {{
      codexRenderTimer = null;
      if (focus_node && focus_node.name === d.name) {{
        codexRenderNodeStats(d);
      }}
    }}, 120);
  }}

  set_focus_node = function(d) {{
    codexOriginalSetFocusNode(d);
    codexRemoveNativeMemberDistribution();
    codexRemoveNativeKeplerStats();
    setTimeout(codexRemoveNativeKeplerStats, 50);
    codexScheduleNodeStats(focus_node);
    codexApplyNodeSizeColors();
  }};

  d3.select("#canvas svg")
    .on("wheel.codexMapMoving", codexMarkMapMoving)
    .on("mousedown.codexMapMoving", codexMarkMapMoving)
    .on("mousemove.codexMapMoving", function(event) {{
      if (event.buttons) {{
        codexMarkMapMoving();
      }}
    }})
    .on("touchmove.codexMapMoving", codexMarkMapMoving);

  d3.select(window)
    .on("mousemove.codexMapMoving", function(event) {{
      if (event.buttons) {{
        codexMarkMapMoving();
      }}
    }})
    .on("mouseup.codexMapMoving", function() {{
      document.body.classList.remove("codex-map-moving");
    }});

  // Native Kepler member distribution still uses dummy point-level color_values;
  // visible node color is injected from node-level n_points.
  update_color_functions();
  codexApplyNodeSizeColors();
</script>
"""


def build_kepler_native_html(
    config: str,
    mapper_out: Path,
    out_path: Path,
) -> None:
    raw_graph = json.loads((mapper_out / f"graph_{config}.json").read_text(encoding="utf-8"))
    nodes_with_topics_path = mapper_out / f"nodes_{config}_with_topics.csv"
    nodes_path = nodes_with_topics_path if nodes_with_topics_path.exists() else mapper_out / f"nodes_{config}.csv"
    nodes = pd.read_csv(nodes_path)
    lens = pd.read_parquet(mapper_out / f"lens_{config}.parquet").sort_values("point_id").reset_index(drop=True)
    lens_values, lens_names = lens_matrix(lens)

    mapper = km.KeplerMapper(verbose=0)
    html_text = mapper.visualize(
        raw_graph,
        color_values=np.zeros(len(lens)),
        color_function_name=["log node size"],
        node_color_function="mean",
        colorscale=SIZE_COLORSCALE,
        custom_tooltips=None,
        custom_meta={
            "config": config,
            "node color": "log1p(n_points), normalized to [0, 1]",
            "note": "Node statistics are injected into the native Kepler Mapper Cluster Details pane.",
        },
        path_html=str(out_path),
        title=f"Kepler Mapper with node statistics: {config}",
        save_file=False,
        lens=lens_values,
        lens_names=lens_names,
        include_searchbar=False,
    )

    html_text = strip_native_member_tooltips(html_text)
    injection = kepler_native_injection(node_stats_payload(nodes))
    html_text = html_text.replace("</body>", injection + "\n</body>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_text, encoding="utf-8")


def infer_configs(mapper_out: Path) -> list[str]:
    return sorted(path.stem.replace("graph_", "", 1) for path in mapper_out.glob("graph_*.json"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapper-out", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--configs", nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs = args.configs or infer_configs(args.mapper_out)

    for config in configs:
        out_path = args.out_dir / f"kepler_native_stats_{config}.html"
        print(f"Writing native Kepler Mapper graph for {config}", flush=True)
        build_kepler_native_html(
            config=config,
            mapper_out=args.mapper_out,
            out_path=out_path,
        )

    print(f"Saved outputs to {args.out_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
