from pathlib import Path

from mapper_node_hover_kepler_native import build_kepler_native_html
from mapper_node_stats import MapperConfig, build_mapper_and_stats
from mapper_node_topics import (
    export_nodes_topic_prompt_batches,
    import_topic_response,
    load_nodes_csv,
    save_nodes_csv,
)




DATA_DIR = Path("test_inference")
EXPERIMENT_NAME = "final_mapper_experiment"
EXPERIMENT_OUT_DIR = Path("test_code") / EXPERIMENT_NAME

REBUILD_MAPPER = True
EXPORT_PROMPTS = True
IMPORT_TOPICS_IF_RESPONSES_EXIST = True
BUILD_HTML = True

MIN_POINTS_FOR_TOPICS = 10
PROMPT_WORD_LIMIT = 8
PROMPT_BATCH_SIZE = 200

ACTS_NAME = "acts_test.npy"
METADATA_NAME = "metadata_test.parquet"
SAE_SPARSE_NAME = "sae_sparse_test.npz"

CONFIGS = [
    MapperConfig(
        name="umap_cosine",
        lens_func="umap",
        n_components=2,
        random_state=42,
        eps=0.5,
        clustering_metric="cosine",
        min_samples=3,
        clusterer="dbscan",
        n_cubes=5,
        perc_overlap=0.5,

        # Нужно только для lens_func="umap".
        n_neighbors=30,
        lens_metric="cosine",
    ),

    MapperConfig(
        name="umap",
        lens_func="umap",  
        n_components=2,
        random_state=42,
        clusterer="dbscan",
        clustering_metric="cosine",
        eps=0.5,
        min_samples=5,
        n_cubes=8,
        perc_overlap=0.3,

        n_neighbors=50,
        lens_metric="cosine"  
    )
]


def has_response_files(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    for pattern in ("*.json", "*.txt", "*.md"):
        if any(path.glob(pattern)):
            return True
    return False


def run_config(config: MapperConfig) -> None:
    config_out_dir = EXPERIMENT_OUT_DIR / config.name
    prompts_dir = config_out_dir / "node_topic_prompts"
    responses_dir = config_out_dir / "node_topic_responses"
    html_dir = config_out_dir / "html_native"

    nodes_path = config_out_dir / f"nodes_{config.name}.csv"
    nodes_with_topics_path = config_out_dir / f"nodes_{config.name}_with_topics.csv"
    html_path = html_dir / f"kepler_native_stats_{config.name}.html"

    config_out_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.mkdir(parents=True, exist_ok=True)
    responses_dir.mkdir(parents=True, exist_ok=True)
    html_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning config: {config.name}", flush=True)
    print(f"Output dir: {config_out_dir.resolve()}", flush=True)

    if REBUILD_MAPPER:
        build_mapper_and_stats(
            config=config,
            data_dir=DATA_DIR,
            mapper_out_dir=config_out_dir,
            acts_name=ACTS_NAME,
            metadata_name=METADATA_NAME,
            sae_sparse_name=SAE_SPARSE_NAME,
        )
    elif not nodes_path.exists():
        raise FileNotFoundError(
            f"REBUILD_MAPPER=False, but nodes file does not exist: {nodes_path}"
        )

    if not nodes_path.exists():
        raise FileNotFoundError(f"Nodes file does not exist: {nodes_path}")

    nodes = load_nodes_csv(nodes_path)

    if EXPORT_PROMPTS:
        index = export_nodes_topic_prompt_batches(
            nodes,
            prompt_dir=prompts_dir,
            min_points=MIN_POINTS_FOR_TOPICS,
            word_limit=PROMPT_WORD_LIMIT,
            batch_size=PROMPT_BATCH_SIZE,
        )
        n_batches = int(index["batch_id"].nunique()) if len(index) else 0
        print(f"Prompt config: {config.name}", flush=True)
        print(f"Prompt nodes: {len(index)}", flush=True)
        print(f"Prompt batches: {n_batches}", flush=True)
        print(f"Saved prompts: {prompts_dir.resolve()}", flush=True)
        print(f"Saved prompt index: {(prompts_dir / 'index.csv').resolve()}", flush=True)

    if IMPORT_TOPICS_IF_RESPONSES_EXIST:
        if has_response_files(responses_dir):
            print(f"Importing topic responses from: {responses_dir.resolve()}", flush=True)
            nodes = load_nodes_csv(nodes_path)
            nodes_with_topics = import_topic_response(
                nodes,
                response_path=responses_dir,
                min_points=MIN_POINTS_FOR_TOPICS,
            )
            save_nodes_csv(nodes_with_topics, nodes_with_topics_path)
            print(f"Saved nodes with topics: {nodes_with_topics_path.resolve()}", flush=True)
        else:
            print(
                f"No topic responses found in {responses_dir.resolve()}. "
                "Put LLM JSON files there and rerun this script.",
                flush=True,
            )

    if BUILD_HTML:
        build_kepler_native_html(
            config=config.name,
            mapper_out=config_out_dir,
            out_path=html_path,
        )
        print(f"Saved HTML: {html_path.resolve()}", flush=True)


def main() -> None:
    print(f"Running Mapper experiment: {EXPERIMENT_NAME}", flush=True)
    print(f"Experiment output dir: {EXPERIMENT_OUT_DIR.resolve()}", flush=True)
    print(f"Configs: {len(CONFIGS)}", flush=True)

    for config in CONFIGS:
        run_config(config)

    print(f"\nExperiment outputs saved to {EXPERIMENT_OUT_DIR.resolve()}", flush=True)


if __name__ == "__main__":
    main()
