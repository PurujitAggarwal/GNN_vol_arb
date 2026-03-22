"""
Build the GLASSO-based stock graph for the GNN model.

Takes daily log returns and figures out which stocks are connected (conditionally dependent) using Graphical LASSO. Outputs a binary adjacency matrix + normalised version for the GNN layer.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from sklearn.covariance import GraphicalLassoCV, GraphicalLasso

from gnn_vol.config import GLASSO_ALPHA, GLASSO_MAX_ITER, GRAPH_DIR


class GraphBuilder:

    def __init__(self, returns_df: pd.DataFrame):
        self.returns_df = returns_df.dropna()
        self.tickers = list(self.returns_df.columns)
        self.n_stocks = len(self.tickers)

        # filled in by build()
        self.adjacency_matrix = None
        self.normalised_matrix = None
        self.precision_matrix = None
        self.sparsity = None

    def build(self) -> dict:
        """
        Full GLASSO pipeline: fit on returns, extract precision matrix,
        threshold into binary adjacency, normalise for GNN.
        Returns dict with adjacency, normalised, precision, sparsity, tickers.
        """
        returns = self.returns_df.values

        # fit GLASSO -use cross-validation if no alpha specified
        if GLASSO_ALPHA is None:
            model = GraphicalLassoCV(
                max_iter=GLASSO_MAX_ITER,
                cv=5,
                assume_centered=False,
            )
        else:
            model = GraphicalLasso(
                alpha=GLASSO_ALPHA,
                max_iter=GLASSO_MAX_ITER,
                assume_centered=False,
            )

        model.fit(returns)
        self.precision_matrix = model.precision_

        if GLASSO_ALPHA is None:
            print(f"  GLASSO picked alpha: {model.alpha_:.6f}")

        # binary adjacency: edge exists if precision entry is meaningfully non-zero
        A = (np.abs(self.precision_matrix) > 1e-5).astype(float)

        # no self-connections
        np.fill_diagonal(A, 0.0)
        self.adjacency_matrix = A

        # normalise: W = D^(-1/2) A D^(-1/2)
        self.normalised_matrix = self._normalise(A)

        # sparsity sanity check (expect 10-40%)
        n_possible = self.n_stocks * (self.n_stocks - 1)
        n_actual = int(A.sum())
        self.sparsity = (n_actual / n_possible) * 100 if n_possible > 0 else 0.0

        print(f"  Graph: {self.n_stocks} stocks, {n_actual // 2} edges, sparsity {self.sparsity:.1f}%")

        return {
            "adjacency": self.adjacency_matrix,
            "normalised": self.normalised_matrix,
            "precision": self.precision_matrix,
            "sparsity": self.sparsity,
            "tickers": self.tickers,
        }

    def get_node_degrees(self) -> pd.Series:
        """How many connections each stock has."""
        if self.adjacency_matrix is None:
            raise RuntimeError("Call build() first.")

        degrees = self.adjacency_matrix.sum(axis=1).astype(int)
        return pd.Series(degrees, index=self.tickers, name="degree").sort_values(ascending=False)

    def plot_graph(self, save: bool = True, filename: str = "stock_graph.png"):
        """NetworkX visualisation. Nodes coloured by sector, sized by degree."""
        if self.adjacency_matrix is None:
            raise RuntimeError("Call build() first.")

        G = nx.from_numpy_array(self.adjacency_matrix)

        # relabel nodes to ticker names
        mapping = {i: t for i, t in enumerate(self.tickers)}
        G = nx.relabel_nodes(G, mapping)

        node_colours = self._get_sector_colours()

        degs = dict(G.degree())
        node_sizes = [100 + 40 * degs[n] for n in G.nodes()]

        fig, ax = plt.subplots(figsize=(16, 12))
        pos = nx.spring_layout(G, seed=42, k=1.5 / np.sqrt(self.n_stocks))

        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                               node_color=node_colours, alpha=0.85,
                               edgecolors="black", linewidths=0.5)
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.15, width=0.5)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=7, font_weight="bold")

        ax.set_title(f"GLASSO Stock Graph  |  {self.n_stocks} stocks, "
                     f"{G.number_of_edges()} edges, sparsity {self.sparsity:.1f}%",
                     fontsize=14, fontweight="bold")
        ax.axis("off")
        plt.tight_layout()

        if save:
            out_path = GRAPH_DIR / filename
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"  Plot saved: {out_path}")

        return fig

    def save(self, tag: str = "latest"):
        """Dump adjacency + normalised + precision to .npy files."""
        if self.adjacency_matrix is None:
            raise RuntimeError("Call build() first.")

        np.save(GRAPH_DIR / f"adjacency_{tag}.npy", self.adjacency_matrix)
        np.save(GRAPH_DIR / f"normalised_{tag}.npy", self.normalised_matrix)
        np.save(GRAPH_DIR / f"precision_{tag}.npy", self.precision_matrix)
        print(f"  Saved graph matrices to {GRAPH_DIR} (tag: {tag})")

    @staticmethod
    def load(tag: str = "latest") -> dict:
        """Load saved graph matrices back from disk."""
        return {
            "adjacency": np.load(GRAPH_DIR / f"adjacency_{tag}.npy"),
            "normalised": np.load(GRAPH_DIR / f"normalised_{tag}.npy"),
            "precision": np.load(GRAPH_DIR / f"precision_{tag}.npy"),
        }

    # helpers

    @staticmethod
    def _normalise(A: np.ndarray) -> np.ndarray:
        # W = D^(-1/2) A D^(-1/2)
        # nodes with zero degree just get zero rows (no division by zero)
        degrees = A.sum(axis=1)

        d_inv_sqrt = np.zeros_like(degrees)
        mask = degrees > 0
        d_inv_sqrt[mask] = 1.0 / np.sqrt(degrees[mask])

        D = np.diag(d_inv_sqrt)
        return D @ A @ D

    def _get_sector_colours(self) -> list:
        # colour each node by GICS sector if universe.py is available
        try:
            from gnn_vol.universe import UNIVERSE

            sectors = list(UNIVERSE.keys())
            cmap = plt.cm.get_cmap("tab20", len(sectors))
            sector_colour = {s: cmap(i) for i, s in enumerate(sectors)}

            ticker_sector = {
                t: s for s, tickers in UNIVERSE.items() for t in tickers
            }

            return [sector_colour.get(ticker_sector.get(t), (0.7, 0.7, 0.7, 1.0))
                    for t in self.tickers]

        except ImportError:
            return ["steelblue"] * self.n_stocks


if __name__ == "__main__":
    import yfinance as yf
    from gnn_vol.universe import ALL_TICKERS

    # pull 10 years of daily prices
    print("Downloading daily prices from yfinance...")
    raw = yf.download(ALL_TICKERS, start="2016-01-01", end="2026-01-01", auto_adjust=True)

    # yfinance gives MultiIndex columns for multiple tickers
    close = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw

    # daily log returns
    log_returns = np.log(close / close.shift(1)).dropna()

    # drop any tickers that failed to download
    valid = [t for t in ALL_TICKERS if t in log_returns.columns]
    log_returns = log_returns[valid]
    print(f"Returns: {log_returns.shape[0]} days x {log_returns.shape[1]} stocks")

    # build
    builder = GraphBuilder(log_returns)
    result = builder.build()

    # degrees
    degrees = builder.get_node_degrees()
    print(f"\nMost connected:")
    print(degrees.head(10))
    print(f"\nLeast connected:")
    print(degrees.tail(5))

    # plot + save
    builder.plot_graph()
    builder.save()

    print("\nDone.")
