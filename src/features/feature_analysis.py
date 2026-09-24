import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.utils.logger import get_logger

logger = get_logger("lv_xai.feature_analysis")

def plot_reduction(embedding, labels, title, output_path):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=labels, palette="viridis", alpha=0.7)
    plt.title(title)
    plt.savefig(output_path)
    plt.close()

def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info(f"Loading features from {args.features_csv}")
    df = pd.read_csv(args.features_csv)
    
    X = df.drop(columns=["label"]).values
    y = df["label"].values
    
    # PCA
    logger.info("Performing PCA...")
    pca = PCA(n_components=2)
    pca_emb = pca.fit_transform(X)
    plot_reduction(pca_emb, y, "PCA Feature Distribution", os.path.join(args.output_dir, "pca_features.png"))
    
    # t-SNE
    logger.info("Performing t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_emb = tsne.fit_transform(X)
    plot_reduction(tsne_emb, y, "t-SNE Feature Distribution", os.path.join(args.output_dir, "tsne_features.png"))
    
    # UMAP
    try:
        import umap
        logger.info("Performing UMAP...")
        umap_model = umap.UMAP(n_components=2, random_state=42)
        umap_emb = umap_model.fit_transform(X)
        plot_reduction(umap_emb, y, "UMAP Feature Distribution", os.path.join(args.output_dir, "umap_features.png"))
    except ImportError:
        logger.warning("umap-learn not installed. Skipping UMAP.")

if __name__ == "__main__":
    _main()
