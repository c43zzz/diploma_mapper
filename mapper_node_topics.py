from __future__ import annotations

import argparse
import json
import math
import re
import numpy as np
import pandas as pd
import typing as tp

from pathlib import Path

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
]


TOPIC_TYPES = {
    "semantic_topic",
    "corpus_style",
    "format_artifact",
    "linguistic_pattern",
    "mixed",
    "unclear",
}


def parse_json_cell(value: tp.Any, fallback: tp.Any) -> tp.Any:
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str) or not value:
        return fallback
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return fallback


def to_jsonable(value: tp.Any):
    """
    Приводит все к формату, который нормально сохрнаяется в json
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    
    if isinstance(value, np.integer):
        return int(value)
    
    if isinstance(value, np.floating):
        return float(value)
    
    if isinstance(value, dict):
        return {
            str(key): to_jsonable(value_dict) for key, value_dict in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            to_jsonable(value_iter) for value_iter in value
        ]
    
    return value


def json_fallback_for_column(col: str) -> tp.Any:
    """
    Возвращает дефолтное значение нужного типа при парсинге json
    """
    if col in {
        "corpus_counts",
        "lexical_categories",
        "top_words",
        "top_word_contexts",
        "distinctive_word_contexts",
    }:
        return {}
    return []


def load_nodes_csv(path: Path) -> pd.DataFrame:
    """
    Загружает ноды из csv после обработки в mapper_node_stats.py
    """
    
    nodes = pd.read_csv(path)

    for col in JSON_COLUMNS:
        if col in nodes.columns:
            nodes[col] = nodes[col].apply(
                lambda value, fallback=json_fallback_for_column(col): parse_json_cell(
                    value,
                    fallback,
                )
            )

    return nodes


def save_nodes_csv(nodes: pd.DataFrame, path: Path) -> None:
    """
    Сохраняем все, что посчитали в CSV
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    nodes_copy = nodes.copy()

    for col in JSON_COLUMNS:
        if col in nodes_copy.columns:
            nodes_copy[col] = nodes_copy[col].apply(
                lambda x: json.dumps(to_jsonable(x), ensure_ascii=False)
            )

    nodes_copy.to_csv(path, index=False)



def scalar_value(value: tp.Any, default: tp.Any = "n/a") -> tp.Any:
    """
    Заменяет значения дефолтом, если они не скаляры
    """
    
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    return value


def fmt_float(value: tp.Any, digits: int = 3, default: str = "n/a") -> str:
    """
    Просто обрезает число до digits знаков после запятой
    """
    
    value = scalar_value(value, None)
    if value is None:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return f"{number:.{digits}f}"


def row_json(row: pd.Series, col: str) -> tp.Any:
    return parse_json_cell(row.get(col, ""), json_fallback_for_column(col))


def word_list(value: tp.Any, limit: int) -> list[str]:
    """
    Возвращает список слов в нужном формате ждля нужного отображения
    """
    if isinstance(value, dict):
        return [str(word) for word in list(value.keys())[:limit]]
    if isinstance(value, list):
        words = []
        for item in value[:limit]:
            if isinstance(item, dict):
                words.append(str(item.get("word", "")))
            else:
                words.append(str(item))
        return [word for word in words if word]
    return []


def format_top_words(value: tp.Any, max_words: int) -> str:
    if not isinstance(value, dict):
        return "[]"

    lines = []
    for word, count in list(value.items())[:max_words]:
        lines.append(f"- {word}: count={count}")
    return "\n".join(lines) if lines else "[]"


def format_json_block(value: tp.Any) -> str:
    return json.dumps(to_jsonable(value), ensure_ascii=False, indent=2)


def format_distinctive_words(value: tp.Any, max_words: int) -> str:
    items = value if isinstance(value, list) else []
    lines = []
    for item in items[:max_words]:
        if not isinstance(item, dict):
            lines.append(f"- {item}")
            continue
        word = item.get("word", "")
        tf_idf = fmt_float(item.get("tf_idf"), 5)
        count = scalar_value(item.get("count"), "n/a")
        lines.append(f"- {word}: tf_idf={tf_idf}, count={count}")
    return "\n".join(lines) if lines else "[]"


def format_word_contexts(
    contexts_by_word: dict[str, list[dict[str, str]]],
    max_words: int,
    max_examples_per_word: int = 1,
) -> str:
    if not isinstance(contexts_by_word, dict) or not contexts_by_word:
        return "[]"

    lines = []
    for word, examples in list(contexts_by_word.items())[:max_words]:
        lines.append(f"word: {word}")
        for example in examples[:max_examples_per_word]:
            if not isinstance(example, dict):
                continue
            lines.append(f"- original word: {example.get('word', '')}")
            lines.append(f"  context: {example.get('context', '')}")

    return "\n".join(lines) if lines else "[]"


def make_node_topic_card(row: pd.Series, word_limit: int = 8) -> str:
    """
    Делает для каждой ноды карточку в промпте
    """
    word_limit = min(word_limit, 8)
    lexical_categories = row_json(row, "lexical_categories")
    top_words = row_json(row, "top_words")
    distinctive_words = row_json(row, "distinctive_words")
    top_word_contexts = row_json(row, "top_word_contexts")
    distinctive_word_contexts = row_json(row, "distinctive_word_contexts")

    return f"""
=== NODE START ===
node_id: {scalar_value(row.get("node_id"))}
n_points: {scalar_value(row.get("n_points"))}

Lexical categories:
{format_json_block(lexical_categories)}

Top words:
{format_top_words(top_words, word_limit)}

Distinctive words:
{format_distinctive_words(distinctive_words, word_limit)}

Contexts for distinctive words:
{format_word_contexts(distinctive_word_contexts, max_words=word_limit, max_examples_per_word=2)}

Contexts for top words:
{format_word_contexts(top_word_contexts, max_words=word_limit, max_examples_per_word=2)}

SAE aggregate hints:
- sae_norm_entropy: {fmt_float(row.get("sae_norm_entropy"))}
- sae_L2_mean: {fmt_float(row.get("sae_L2_mean"))}
=== NODE END ===
""".strip()


def prompt_header() -> str:
    return """
You describe Mapper graph nodes from language model activation analysis.

Each node contains word-level points. Each point is a word selected from a text and represented using Gemma-2-2B activations.
SAE source: Gemma-2-2B, layer 13, GemmaScope residual SAE 16k

This prompt may contain only one batch of nodes from a larger Mapper graph.
It is okay that this is only a subset of all graph nodes.

Do not force semantic topics. A node can be:
- semantic_topic
- corpus_style
- format_artifact
- linguistic_pattern
- mixed
- unclear

Use only the evidence in each node card.
If evidence is weak, mixed, mostly lexical/formatting, or too noisy, say that and use low confidence.

Return strict JSON only with this schema:
{
  "nodes": [
    {
      "node_id": "...",
      "topic_label": "short label, 2-5 words",
      "topic_description": "2-4 sentence description",
      "topic_type": "semantic_topic | corpus_style | format_artifact | linguistic_pattern | mixed | unclear",
      "confidence": 0.0,
      "evidence_words": ["word1", "word2", "word3"],
      "possible_artifact": true
    }
  ]
}

Important:
- include exactly one object for each node_id listed below;
- preserve node_id exactly;
- return JSON only for node_id values present in this prompt;
- do not invent node_id values from other batches;
- confidence must be between 0 and 1.
- Prefer returning the result as a downloadable JSON file named <input_file_name>_response.json when the interface supports files.
- If file output is not available, return raw strict JSON in the chat.
- Do not use markdown fences.
- Do not add explanations before or after JSON.
""".strip()


def export_nodes_topic_prompt(
    nodes: pd.DataFrame,
    prompt_path: Path,
    min_points: int = 10,
    word_limit: int = 8,
) -> pd.DataFrame:
    """
    Строит промпт по нужному пути и сохраняет его. 
    Также строит индексынй csv файл, по которому восстанавливает ответы модели для разных нод
    """
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    selected = nodes[nodes["n_points"].astype(int) >= min_points].copy()

    cards = [
        make_node_topic_card(row, word_limit=word_limit)
        for _, row in selected.iterrows()
    ]
    prompt_text = prompt_header() + "\n\n" + "\n\n".join(cards) + "\n"
    prompt_path.write_text(prompt_text, encoding="utf-8")

    index = selected.reset_index().rename(columns={"index": "row_index"})[
        ["row_index", "node_id", "n_points"]
    ]
    index.to_csv(prompt_path.with_suffix(".index.csv"), index=False)
    return index


def export_nodes_topic_prompt_batches(
    nodes: pd.DataFrame,
    prompt_dir: Path,
    min_points: int = 10,
    word_limit: int = 8,
    batch_size: int = 200,
) -> pd.DataFrame:
    """
    Делает все то же, что и верхняя функция, но только теперь делает это по батчам, 
    чтобы промпт не разрастался
    """
    prompt_dir.mkdir(parents=True, exist_ok=True)
    selected = nodes[nodes["n_points"].astype(int) >= min_points].copy()
    selected = selected.reset_index().rename(columns={"index": "row_index"})

    index_rows = []
    n_batches = math.ceil(len(selected) / batch_size) if len(selected) else 0

    for batch_id in range(n_batches):
        start = batch_id * batch_size
        end = start + batch_size
        batch = selected.iloc[start:end]
        prompt_file = f"batch_{batch_id:03d}.md"
        prompt_path = prompt_dir / prompt_file

        cards = [
            make_node_topic_card(row, word_limit=word_limit)
            for _, row in batch.iterrows()
        ]
        prompt_text = prompt_header() + "\n\n" + "\n\n".join(cards) + "\n"
        prompt_path.write_text(prompt_text, encoding="utf-8")

        for _, row in batch.iterrows():
            index_rows.append({
                "batch_id": batch_id,
                "prompt_file": prompt_file,
                "row_index": row["row_index"],
                "node_id": row["node_id"],
                "n_points": row["n_points"],
            })

    index = pd.DataFrame(
        index_rows,
        columns=["batch_id", "prompt_file", "row_index", "node_id", "n_points"],
    )
    index.to_csv(prompt_dir / "index.csv", index=False)
    return index


def strip_markdown_json_fence(text: str) -> str:
    """
    Эта функция для нормального парсинга ответа LLM в формате JSON
    """
    text = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def parse_model_json(text: str) -> dict:
    clean_text = strip_markdown_json_fence(str(text))
    try:
        return json.loads(clean_text)
    except json.JSONDecodeError as error:
        first = clean_text.find("{")
        last = clean_text.rfind("}")
        if first >= 0 and last > first:
            try:
                return json.loads(clean_text[first:last + 1])
            except json.JSONDecodeError:
                pass
        return {
            "parse_error": True,
            "error": str(error),
            "raw_text": text,
        }


def clamp01(value: tp.Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return min(1.0, max(0.0, number))


def normalize_bool(value: tp.Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def normalize_words(value: tp.Any) -> list[str]:
    if isinstance(value, str):
        value = [part.strip() for part in value.split(",")]
    if not isinstance(value, list):
        return []
    return [str(word).strip() for word in value if str(word).strip()]


def normalize_topic_result(result: tp.Any) -> dict:
    """
    Обрабатываем ответ модели про нашу конкретную ноду и что в ней содержится
    """
    if isinstance(result, str):
        result = parse_model_json(result)
    if not isinstance(result, dict):
        result = {"parse_error": True, "raw_text": str(result)}

    if result.get("parse_error"):
        return {
            "topic_label": "parse error",
            "topic_description": "Model output could not be parsed as JSON.",
            "topic_type": "unclear",
            "topic_confidence": 0.0,
            "topic_evidence_words": [],
            "topic_possible_artifact": True,
        }

    topic_type = str(result.get("topic_type", "unclear")).strip()
    if topic_type not in TOPIC_TYPES:
        topic_type = "unclear"

    topic_label = str(result.get("topic_label", "")).strip()
    if not topic_label:
        topic_label = "unknown"

    return {
        "topic_label": topic_label,
        "topic_description": str(result.get("topic_description", "")).strip(),
        "topic_type": topic_type,
        "topic_confidence": clamp01(result.get("topic_confidence", result.get("confidence", 0.0))),
        "topic_evidence_words": normalize_words(
            result.get("topic_evidence_words", result.get("evidence_words", []))
        ),
        "topic_possible_artifact": normalize_bool(
            result.get("topic_possible_artifact", result.get("possible_artifact", False))
        ),
    }


def small_node_topic_result() -> dict:
    return {
        "topic_label": "small node",
        "topic_description": "Node is too small for reliable automatic topic description.",
        "topic_type": "unclear",
        "topic_confidence": 0.0,
        "topic_evidence_words": [],
        "topic_possible_artifact": True,
    }


def not_processed_topic_result() -> dict:
    return {
        "topic_label": "not processed",
        "topic_description": "No topic description was imported for this node.",
        "topic_type": "unclear",
        "topic_confidence": 0.0,
        "topic_evidence_words": [],
        "topic_possible_artifact": True,
    }


def iter_response_files(response_path: Path) -> list[Path]:
    """
    Берет все файлы подходящего формата в папке, в которой ждется ответы нашей модели по топикам
    """
    if response_path.is_file():
        return [response_path]
    if response_path.is_dir():
        files = []
        for pattern in ("*.json", "*.txt", "*.md"):
            files.extend(response_path.glob(pattern))
        return sorted(set(files))
    raise FileNotFoundError(f"Response path not found: {response_path}")


def load_topic_response_items(response_path: Path) -> tuple[dict[str, dict], bool, dict[str, tp.Any]]:
    response_files = iter_response_files(response_path)
    is_single_file = response_path.is_file() # Случай одного файлы
    response_by_node = {}
    stats = {
        "n_files": len(response_files),
        "n_parsed_files": 0,
        "n_failed_files": 0,
        "n_items": 0,
        "n_duplicate_node_ids": 0,
        "parse_errors": [],
    }

    for path in response_files:
        response = parse_model_json(path.read_text(encoding="utf-8"))
        if isinstance(response, dict) and response.get("parse_error"):
            stats["n_failed_files"] += 1
            stats["parse_errors"].append({
                "path": str(path),
                "error": response.get("error", "parse error"),
            })
            continue

        response_items = response.get("nodes", []) if isinstance(response, dict) else []
        if not isinstance(response_items, list):
            stats["n_failed_files"] += 1
            stats["parse_errors"].append({
                "path": str(path),
                "error": "response does not contain a nodes list",
            })
            continue

        stats["n_parsed_files"] += 1
        for item in response_items:
            if not isinstance(item, dict) or "node_id" not in item:
                continue
            node_id = str(item["node_id"])
            if node_id in response_by_node:
                stats["n_duplicate_node_ids"] += 1
            response_by_node[node_id] = item
            stats["n_items"] += 1

    had_global_parse_error = (
        is_single_file and stats["n_failed_files"] > 0
    ) or (
        not is_single_file and stats["n_parsed_files"] == 0
    )
    return response_by_node, had_global_parse_error, stats


def import_topic_response(
    nodes: pd.DataFrame,
    response_path: Path,
    min_points: int = 10,
) -> pd.DataFrame:
    response_by_node, response_parse_error, stats = load_topic_response_items(response_path)
    print(
        "Loaded topic responses: "
        f"{stats['n_parsed_files']}/{stats['n_files']} files parsed, "
        f"{stats['n_items']} items",
        flush=True,
    )
    if stats["n_failed_files"]:
        print(f"WARNING: Failed to parse {stats['n_failed_files']} response files", flush=True)
        for error in stats["parse_errors"][:10]:
            print(f"  {error['path']}: {error['error']}", flush=True)
    if stats["n_duplicate_node_ids"]:
        print(
            f"WARNING: {stats['n_duplicate_node_ids']} duplicate node_id responses were overwritten",
            flush=True,
        )

    parse_error_result = {
        "parse_error": True,
        "error": "No response files could be parsed.",
    }
    if stats["parse_errors"]:
        parse_error_result["error"] = stats["parse_errors"][0].get("error", "parse error")

    nodes_out = nodes.copy()
    results = []
    for _, row in nodes_out.iterrows():
        node_id = str(row.get("node_id"))
        n_points = int(row.get("n_points", 0))

        if n_points < min_points:
            results.append(small_node_topic_result())
        elif response_parse_error:
            results.append(normalize_topic_result(parse_error_result))
        elif node_id in response_by_node:
            results.append(normalize_topic_result(response_by_node[node_id]))
        else:
            results.append(not_processed_topic_result())

    for col in [
        "topic_label",
        "topic_description",
        "topic_type",
        "topic_confidence",
        "topic_evidence_words",
        "topic_possible_artifact",
    ]:
        nodes_out[col] = [result[col] for result in results]

    return nodes_out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export-prompt")
    export_parser.add_argument("--nodes-in", type=Path, required=True)
    export_parser.add_argument("--prompt-out", type=Path, required=True)
    export_parser.add_argument("--min-points", type=int, default=10)
    export_parser.add_argument("--word-limit", type=int, default=8)
    export_parser.add_argument("--batch-size", type=int, default=0)

    import_parser = subparsers.add_parser("import-response")
    import_parser.add_argument("--nodes-in", type=Path, required=True)
    import_parser.add_argument("--response-in", type=Path, required=True)
    import_parser.add_argument("--nodes-out", type=Path, required=True)
    import_parser.add_argument("--min-points", type=int, default=10)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    nodes = load_nodes_csv(args.nodes_in)

    if args.command == "export-prompt":
        if args.batch_size > 0:
            index = export_nodes_topic_prompt_batches(
                nodes,
                prompt_dir=args.prompt_out,
                min_points=args.min_points,
                word_limit=args.word_limit,
                batch_size=args.batch_size,
            )
            n_batches = int(index["batch_id"].nunique()) if len(index) else 0
            print(f"Selected nodes: {len(index)}", flush=True)
            print(f"Created batches: {n_batches}", flush=True)
            print(f"Saved batch prompts to {args.prompt_out.resolve()}", flush=True)
            print(f"Saved batch index to {(args.prompt_out / 'index.csv').resolve()}", flush=True)
        else:
            index = export_nodes_topic_prompt(
                nodes,
                prompt_path=args.prompt_out,
                min_points=args.min_points,
                word_limit=args.word_limit,
            )
            print(f"Saved prompt to {args.prompt_out.resolve()}", flush=True)
            print(f"Saved prompt index with {len(index)} nodes", flush=True)
    elif args.command == "import-response":
        nodes_with_topics = import_topic_response(
            nodes,
            response_path=args.response_in,
            min_points=args.min_points,
        )
        save_nodes_csv(nodes_with_topics, args.nodes_out)
        print(f"Saved nodes with topics to {args.nodes_out.resolve()}", flush=True)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
