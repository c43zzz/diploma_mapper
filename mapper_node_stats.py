import kmapper as km
import numpy as np
import pandas as pd
import typing as tp
import networkx as nx
import umap
import math
import string
import re
import json

from pathlib import Path
from dataclasses import dataclass
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from collections import Counter
from scipy import sparse


TRASH_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "did", "do",
    "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "if", "in", "into",
    "is", "it", "its", "itself", "just", "me", "more", "most", "my",
    "myself", "no", "nor", "not", "now", "of", "off", "on", "once",
    "only", "or", "other", "our", "ours", "ourselves", "out", "over",
    "own", "same", "she", "should", "so", "some", "such", "than", "that",
    "the", "their", "theirs", "them", "themselves", "then", "there",
    "these", "they", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "we", "were", "what", "when", "where",
    "which", "while", "who", "whom", "why", "will", "with", "you",
    "your", "yours", "yourself", "yourselves"
}
@dataclass
class MapperConfig:
    name: str

    #Lens
    lens_func: str
    n_components: int 
    random_state: int

    #Mapper
    eps: float
    clustering_metric: str
    min_samples: int
    clusterer: str

    #Cover
    n_cubes: int
    perc_overlap: float

    n_neighbors: int = 0
    lens_metric: str = ""



def build_lens(
        config: MapperConfig,
        acts: np.ndarray,
) -> np.ndarray:
    """
    Строит проекции по функции-линзе
    """
    if config.lens_func == "pca":
        lens = PCA(
            n_components=config.n_components,
            random_state=config.random_state
        )
        projections = lens.fit_transform(acts)

    elif config.lens_func == "umap":
        lens = umap.UMAP(
            n_neighbors=config.n_neighbors,
            n_components=config.n_components,
            metric=config.lens_metric,
            random_state=config.random_state
        )
        projections = lens.fit_transform(acts)
    else:
        raise ValueError("Not Valid Lens Function")

    return projections


def build_mapper_and_stats(
        config: MapperConfig,
        data_dir: Path,
        mapper_out_dir: Path,
        acts_name: str = "acts_test.npy",
        metadata_name: str = "metadata_test.parquet",
        sae_sparse_name: str = "sae_sparse_test.npz",
):
    """
    Основная функция. Она считает всю статистику по нодам и дальше сохрнаяет эту статистику в csv формате
    1) Загружаем файлы
    2) Выбираем конфигурацию Mapper
    3) Строим Mapper
    4) Считаем статистики по нодам
    5) Сохраняем статистику в .csv файл
    """
    mapper_out_dir.mkdir(exist_ok=True, parents=True)

    acts_path = data_dir / acts_name
    metadata_path = data_dir / metadata_name
    sae_sparse_path = data_dir / sae_sparse_name

    acts = np.load(acts_path)
    metadata = pd.read_parquet(metadata_path)
    sae_sparse = sparse.load_npz(sae_sparse_path)

    metadata = metadata.sort_values("point_id").reset_index(drop=True)

    lens = build_lens(
        config,
        acts, 
    )

    mapper = build_mapper_graph(
        config,
        acts, 
        lens
    )

    G = mapper_to_nx(mapper)

    nodes = compute_node_stats(
        G,
        metadata,
        sae_sparse,
    )

    graph_path = mapper_out_dir / f"graph_{config.name}.json"
    lens_path = mapper_out_dir / f"lens_{config.name}.parquet"
    nodes_path = mapper_out_dir / f"nodes_{config.name}.csv"

    graph_path.write_text(
        json.dumps(to_jsonable(mapper), ensure_ascii=False),
        encoding="utf-8"
    )


    lens_data = {"point_id": metadata["point_id"].to_numpy()}

    for dim in range(lens.shape[1]):
        lens_data[f"lens_dim_{dim}"] = lens[:, dim]

    pd.DataFrame(lens_data).to_parquet(lens_path)

    saved_nodes_csv(nodes, nodes_path)


def build_mapper_graph(
    config: MapperConfig,
    acts: np.ndarray,
    lens: tp.Any
):
    mapper = km.KeplerMapper(verbose=1)

    if config.clusterer == "dbscan":
        eps = config.eps
        min_samples = config.min_samples
        metric = config.clustering_metric
        clusterer = DBSCAN(
            eps,
            min_samples=min_samples,
            metric=metric
        )
    else:
        raise ValueError("Not Valid Clusterer")

    graph = mapper.map(
        lens,
        acts, 
        clusterer=clusterer,
        cover=km.Cover(
            n_cubes=config.n_cubes,
            perc_overlap=config.perc_overlap,
        )
    )

    return graph

def mapper_to_nx(mapper: dict) -> nx.Graph:
    G = nx.Graph()

    for node, points_id in mapper["nodes"].items():
        G.add_node(
            node, 
            n_points=len(points_id),
            points_id=sorted(points_id)
        )
    
    for u, neighbours in mapper.get("links", {}).items():
        for v in neighbours:
            overlap = len(set(mapper["nodes"][u]) & set(mapper["nodes"][v]))
            G.add_edge(
                u,
                v, 
                overlap=overlap,
                weight=overlap
            )

    return G

def create_distinctive_words(
        node_word_counts: Counter,
        word_node_df: Counter,
        n_nodes: int,
        top_n: int = 20
) -> list:
    """
    Считает характерные слова для этой конкретной ноды по TF-IDF
    """
    distinctive_words = []
    total_node_words = sum(node_word_counts.values())
    if total_node_words > 0:
        for word, count in node_word_counts.items():
            tf = count / total_node_words
            idf = math.log((n_nodes + 1) / (word_node_df[word] + 1))

            tf_idf = tf * idf
            
            distinctive_words.append({
                "word": word,
                "tf_idf": tf_idf,
                "tf": tf,
                "idf": idf,
                "count": count,
                "node_df": word_node_df[word]
            })
    
    distinctive_words = sorted(
        distinctive_words,
        key=lambda x: x["tf_idf"],
        reverse=True
    )

    return distinctive_words[:top_n]



def normalize_word(word: str) -> str:
    norm_word = word.strip(string.whitespace + "“”‘’`\"")
    norm_word = norm_word.lower()
    norm_word = re.sub(r"^[^\w]+|[^\w]+$", "", norm_word, flags=re.UNICODE)

    return norm_word

def normalized_entropy_counter(counts: Counter) -> float:
    summa = sum(counts.values())
    if summa == 0 or len(counts) <= 1:
        return 0.0
    result = 0.0
    for count in counts.values():
        prob = count / summa
        result -= prob * math.log2(prob)
    return result / math.log2(len(counts))

def compact_counter(counter: Counter, top_n: int = 10) -> dict[str, int]:
    """
    Возвращает каунтер в виде словаря, взяв top_n самых часто встречаемых элементов
    """
    result = {}
    for key, value in counter.most_common(top_n):
        result[str(key)] = int(value)

    return result

def lexical_category(word: str) -> str:
    """
    Разбивает на простенькие категории, которые мы можем посчитать алгоритмически
    """

    norm_word = normalize_word(word)

    if re.fullmatch(r"[\d,.\-+/%$£€]+", norm_word):
        return "number_or_money"
    if re.search(r"https?://|www\.|@", norm_word):
        return "url_or_email"
    if norm_word in TRASH_WORDS or len(norm_word) <= 2:
        return "function_or_short"
    if "-" in norm_word:
        return "hyphenated"
    if norm_word.isalpha():
        return "plain_word"
    return "mixed"

def safe_numeric_summary(values: pd.Series) -> dict[str, float]:
    """
    Считает базовые средние статистики по числовой pd.Series
    """
    
    values = pd.to_numeric(values, errors="coerce").astype("float64").dropna()
    if values.empty:
        return {"mean": np.nan, "std": np.nan, "median": np.nan}
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "median": float(values.median())
    }

def normalized_entropy(values: tp.Iterable[float]) -> float:
    if isinstance(values, Counter):
        return normalized_entropy_counter(values)

    arr = np.asarray(values, dtype=np.float64).ravel()
    arr = arr[arr > 0]
    if arr.size == 0:
        return 0.0

    probs = arr / arr.sum()
    entropy = float(-np.sum(probs * np.log2(probs)))
    if arr.size <= 1:
        return 0.0

    return float(entropy / math.log2(arr.size))

def _as_1d_array(values: tp.Any) -> np.ndarray:
    """
    Преваращает любой массив-like элемент в вытянутый dense numpy массив
    """
    
    if sparse.issparse(values):
        values = values.toarray()

    return np.asarray(values).ravel()

def summarize_sae_features(
        points_id: list[int],
        sae_sparse: tp.Any,
        top_n: int
) -> dict[str, tp.Any]:
    """
    Считаем статистику по SAE для одной ноды:
    -Энтропия (нормализованная)
    -Количестов активных фичей
    -Топ активных фичей с частотой (по модулю активаций и по частоте активаций)
    -Что-то про среднюю L2 норму
    """
    node_sae = sae_sparse[points_id]

    abs_node_sae = node_sae.copy()
    abs_node_sae.data = np.abs(abs_node_sae.data)

    bin_node_sae = node_sae.copy()
    bin_node_sae.data = np.ones_like(bin_node_sae.data)

    abs_totals = _as_1d_array(abs_node_sae.sum(axis=0))
    signed_totals = _as_1d_array(node_sae.sum(axis=0))
    activated_totals = _as_1d_array(bin_node_sae.sum(axis=0))

    active_features = np.flatnonzero(activated_totals > 0)
    n_active_feature = len(active_features)
    norm_entropy = normalized_entropy(abs_totals[active_features])

    sae_L2 = _as_1d_array(np.sqrt(node_sae.multiply(node_sae).sum(axis=1)))
    sae_L0_by_point = _as_1d_array(bin_node_sae.sum(axis=1))

    top_by_activations = active_features[
        np.argsort(abs_totals[active_features])[::-1]
    ][:top_n]

    top_features_by_L1 = []
    for feature in top_by_activations:
        feature = int(feature)
        count = int(activated_totals[feature])
        abs_total = float(abs_totals[feature])
        signed_total = float(signed_totals[feature])

        top_features_by_L1.append({
            "feature": feature,
            "abs_total_activations": abs_total,
            "signed_total_activation": signed_total,
            "activated": count,
            "mean_abs_when_active": abs_total / max(count, 1),
            "prevalence": count / len(points_id)
        })

    top_by_prevalence = active_features[
        np.argsort(activated_totals[active_features])[::-1]
    ][:top_n]

    top_features_by_prevalence = []
    for feature in top_by_prevalence:
        feature = int(feature)
        count = int(activated_totals[feature])
        abs_total = float(abs_totals[feature])
        signed_total = float(signed_totals[feature])

        top_features_by_prevalence.append({
            "feature": feature,
            "abs_total_activations": abs_total,
            "signed_total_activation": signed_total,
            "activated": count,
            "prevalence": count / len(points_id),
            "mean_abs_when_active": abs_total / max(count, 1)
        })

    return {
        "sae_norm_entropy": norm_entropy,
        "sae_active_features": n_active_feature,

        "top_features": top_features_by_L1,
        "top_features_by_prevalence": top_features_by_prevalence,

        "sae_L2_mean": sae_L2.mean(),
        "sae_L2_median": np.median(sae_L2),
        "sae_L2_std": np.std(sae_L2, ddof=1) if len(sae_L0_by_point) > 1 else np.std(sae_L2),

        "sae_L0_mean": sae_L0_by_point.mean(),
        "sae_L0_median": np.median(sae_L0_by_point),
        "sae_L0_std": np.std(sae_L0_by_point, ddof=1) if len(sae_L0_by_point) > 1 else np.std(sae_L0_by_point)
    }


def collect_contexts_for_words(
    node_df: pd.DataFrame,
    words: list[str], # Должны состоять из norm_word
    examples_per_word: int = 1,
) -> dict[str, list[dict[str, str]]]:
    """
    Сохраняем контекст для слов, которые мы взяли в ноду. 
    Дальше он передается для отображения в html и промпт для анализа нод с помощью LLM.
    """
    
    if "context_char" not in node_df.columns:
        return {}

    wanted_words = set(words)
    contexts = {word: [] for word in words}

    for _, row in node_df.iterrows():
        norm_word = normalize_word(row["word"])
        if norm_word not in wanted_words:
            continue
        if len(contexts.get(norm_word, [])) >= examples_per_word:
            continue

        context = row.get("context_char")
        if pd.isna(context):
            continue

        contexts.setdefault(norm_word, []).append({
            "word": row["word"],
            "context": context,
        })

    return {
        word: examples
        for word, examples in contexts.items()
        if examples
    }

def graph_neighbour_stats(G: nx.Graph, node: tp.Any, n_points: int) -> dict[str, float | int]:
    """
    Считает графовую статистику по соседям
    """
    
    overlaps = [
        float(G.edges[node, neighbour].get("overlap", 1))
        for neighbour in G.neighbors(node)
    ]
    
    if len(overlaps) == 0: # Изолированная вершина
        return {
            "degree": 0,
            "weighted_degree": 0.0,
            "intersection_degree_ratio": 0.0,
            "neighbor_overlap_mean": 0.0,
            "neighbor_overlap_max": 0.0,
            "neighbor_overlap_share_mean": 0.0
        }

    overlaps_arr = np.asarray(overlaps, dtype=np.float64)
    return {
        "degree": int(G.degree(node)),
        "weighted_degree": float(overlaps_arr.sum()),
        "intersection_degree_ratio": float(overlaps_arr.sum() / G.degree(node)),
        "neighbor_overlap_mean": float(overlaps_arr.mean()),
        "neighbor_overlap_max": float(overlaps_arr.max()),
        "neighbor_overlap_share_mean": float(overlaps_arr.mean() / max(n_points, 1))
    }

def compute_node_stats(
        G: nx.Graph,
        metadata: pd.DataFrame,
        sae_sparse: tp.Any,
        top_words_n: int = 10,
        top_sae_n: int = 10,
):
    """
    Функция считает следующие вещи:
    -Энтропию нормированная по корпусам
    -Нормированная энтропия по SAE фича активациям
    -Топ активных SAE фичей
    -Характерные слова (distinctive words)
    -Топики по словам (LLM)
    -Размер ноды
    -Компоненту связности в графе
    -Степень ноды в графе
    -Top words
    -Top sae features
    """
    
    metadata = metadata.sort_values("point_id").reset_index(drop=True)
    #Проверим на всякий
    if not np.array_equal(metadata["point_id"].to_numpy(), np.arange(len(metadata))):
        raise ValueError("Error In Data Shape. Incorrect!")
    
    component_id_by_node = {}
    component_size_by_id = {}
    for component_id, component_nodes in enumerate(nx.connected_components(G)):
        component_size_by_id[component_id] = len(component_nodes)
        for node in component_nodes:
            component_id_by_node[node] = component_id

    # Считаем статистику для tf-idf и по другим словам
    word_node_df = Counter()
    node_words_count_by_node = {}
    lexical_categories = {}
    top_words = {}

    for node, data in G.nodes(data=True):
        points_id = list(map(int, data["points_id"]))
        node_df = metadata.iloc[points_id]

        node_word_counts = Counter()
        lex = Counter()

        for word in node_df["word"]:
            lex[lexical_category(word)] += 1
            norm_word = normalize_word(word)
            if len(norm_word) > 2 and norm_word not in TRASH_WORDS:
                node_word_counts[norm_word] += 1

        node_words_count_by_node[node] = node_word_counts
        lexical_categories[node] = lex
        top_words[node] = {key: value for key, value in node_word_counts.most_common(top_words_n)}
        for word in node_word_counts:
            word_node_df[word] += 1
    
    # Считаем статистику по нодам
    n_nodes = G.number_of_nodes()
    rows = []
    numeric_cols = [
        col for col in [
            "act_L2",
            "feat_L2",
            "rec_err_L2",
            "rec_err_relative",
            "sae_L1",
            "sae_L0",
            "sae_max"
        ]
        if col in metadata.columns
    ]

    for node, data in G.nodes(data=True):
        points_id = list(map(int, data["points_id"]))
        node_df = metadata.iloc[points_id] 
        n_points = int(data.get("n_points", len(points_id)))

        # Корпуса
        corpus_counts = Counter(node_df["corpus"])
        doc_counts = Counter(node_df["doc_id"])
        top_corpus, top_count = corpus_counts.most_common(1)[0]
        top_corpus_p = top_count / max(n_points, 1)
        
        node_word_counts = node_words_count_by_node[node]
        node_top_words = top_words[node]
        node_lexical_categories = lexical_categories[node]

        # Характерные слова по TF-IDF для этой конкретной ноды
        distinctive_words = create_distinctive_words(
            node_word_counts,
            word_node_df,
            n_nodes,
            top_words_n
        )

        top_word_contexts = collect_contexts_for_words(
            node_df,
            words=list(node_top_words.keys()),
            examples_per_word=1,
        )

        distinctive_word_contexts = collect_contexts_for_words(
            node_df,
            words=[item["word"] for item in distinctive_words],
            examples_per_word=1,
        )

        sae_stats = summarize_sae_features(
            points_id,
            sae_sparse=sae_sparse,
            top_n=top_sae_n
        )

        component_id = component_id_by_node[node]
        row = {
            "node_id": node,
            "n_points": n_points,
            "component_id": component_id,
            "component_size": component_size_by_id[component_id],
            "dominant_corpus": top_corpus,
            "dominant_corpus_share": top_corpus_p,
            "corpus_entropy_norm": normalized_entropy(corpus_counts),
            "doc_entropy_norm": normalized_entropy(doc_counts),
            "unique_docs": len(doc_counts),
            "word_entropy_norm": normalized_entropy(node_word_counts),
            "corpus_counts": compact_counter(corpus_counts, 10),
            "lexical_categories": compact_counter(node_lexical_categories, 10),
            "top_words": node_top_words,
            "distinctive_words": distinctive_words,
            "top_word_contexts": top_word_contexts,
            "distinctive_word_contexts": distinctive_word_contexts,
        }

        row.update(sae_stats)
        row.update(graph_neighbour_stats(G, node, n_points))

        for col in numeric_cols:
            summary = safe_numeric_summary(node_df[col])
            row[f"{col}_mean"] = summary["mean"]
            row[f"{col}_std"] = summary["std"]
            row[f"{col}_median"] = summary["median"]

        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["n_points", "node_id"],
        ascending=[False, True]
    ).reset_index(drop=True)


JSON_COLUMNS = [
    "corpus_counts",
    "lexical_categories",
    "top_words",
    "distinctive_words",
    "top_word_contexts",
    "distinctive_word_contexts",
    "top_features",
    "top_features_by_prevalence",
]

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

def saved_nodes_csv(nodes: pd.DataFrame, path: Path) -> None:
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


