from mapper_node_stats import MapperConfig
"""
Здесь собраны все конфигурации, которые использовались для проведения экспериментов
"""
CONFIGS = [
    # Использовался только для теста пайплайна и отладки
    MapperConfig( 
        name="pipeline_first_test",
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
    
    # Не использовался в финальных экспериментах из-за плохо интерпретируемых вершин
    MapperConfig( # Дает слишком общие и большие ноды. Тяжело понять что в них содержиться
        name="umap_base",
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
    ),

    MapperConfig(
        name="umap_freedom_1",
        lens_func="umap",  
        n_components=2,
        random_state=42,
        clusterer="dbscan",
        clustering_metric="cosine",
        eps=0.5,
        min_samples=5,
        n_cubes=10,
        perc_overlap=0.3,

        n_neighbors=30,
        lens_metric="cosine"  
    ),
    
    MapperConfig(
        name="umap_freedom_2",
        lens_func="umap",  
        n_components=2,
        random_state=42,
        clusterer="dbscan",
        clustering_metric="cosine",
        eps=0.5,
        min_samples=5,
        n_cubes=12,
        perc_overlap=0.3,

        n_neighbors=30,
        lens_metric="cosine"  
    ),

    MapperConfig(
        name="umap_freedom",
        lens_func="umap",  
        n_components=2,
        random_state=42,
        clusterer="dbscan",
        clustering_metric="cosine",
        eps=0.3,
        min_samples=5,
        n_cubes=12,
        perc_overlap=0.3,

        n_neighbors=30,
        lens_metric="cosine"  
    ),

    MapperConfig(
        name="pca_restrained_1",
        lens_func="pca",
        n_components=2,
        random_state=42,
        clusterer="dbscan",
        clustering_metric="cosine",
        eps=0.5,
        min_samples=5,
        n_cubes=8,
        perc_overlap=0.3,
    ),

    MapperConfig(
        name="pca_restrained_2",
        lens_func="pca",
        n_components=2,
        random_state=42,
        clusterer="dbscan",
        clustering_metric="cosine",
        eps=0.5,
        min_samples=5,
        n_cubes=10,
        perc_overlap=0.3,
    ),

    MapperConfig(
        name="pca_freedom",
        lens_func="pca",
        n_components=2,
        random_state=42,
        clusterer="dbscan",
        clustering_metric="cosine",
        eps=0.3,
        min_samples=5,
        n_cubes=12,
        perc_overlap=0.3,
    ),
    MapperConfig(
        name="umap_sparse",
        lens_func="umap",  
        n_components=2,
        random_state=42,
        clusterer="dbscan",
        clustering_metric="cosine",
        eps=0.25,
        min_samples=10,
        n_cubes=12,
        perc_overlap=0.3,

        n_neighbors=30,
        lens_metric="cosine"  
    ),

    MapperConfig(
        name="umap_sparse_2",
        lens_func="umap",  
        n_components=2,
        random_state=42,
        clusterer="dbscan",
        clustering_metric="cosine",
        eps=0.3,
        min_samples=10,
        n_cubes=12,
        perc_overlap=0.3,

        n_neighbors=25,
        lens_metric="cosine"  
    ),
]