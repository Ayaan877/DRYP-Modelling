from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle, Patch
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
MATRIX_DIR = SCRIPT_DIR / "Crop_Matrices"
OUTPUT_DIR = SCRIPT_DIR / "Crop_Matrix_Plots"
TARGET_YEAR = 2022
TARGET_SEASON = "kharif"

HEATMAP_MIN_OCCURRENCES = 3   # crops with fewer total occurrences are dropped from the heatmap
NETWORK_MIN_OCCURRENCES = 3   # crops with fewer total occurrences are dropped from the network
NETWORK_MIN_EDGE_WEIGHT = 0   # co-occurrence counts below this are not drawn as links
NETWORK_LABEL_ALL_BELOW = 30  # label every node if the network has at most this many nodes
NETWORK_TOP_LABELS = 20       # otherwise only label this many largest nodes
NODE_SIZE = 350
N_BUCKETS = 5                 # colour-ramp buckets for total occurrences (equal crop counts)
DPI = 200

# -----------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------

def build_input_path(year, season):
	return MATRIX_DIR / f"{year}_{season}_crop_cooccurrence_matrix.csv"


def load_matrix(path):
	if not path.exists():
		raise FileNotFoundError(f"Missing matrix file (run make_crop_matrix.py first): {path}")
	matrix = pd.read_csv(path, index_col=0, encoding="utf-8-sig")
	matrix.columns = matrix.columns.astype(str)
	matrix.index = matrix.index.astype(str)
	return matrix.loc[matrix.index, matrix.index]


def filter_by_occurrences(matrix, min_occurrences):
	totals = pd.Series(np.diag(matrix.to_numpy()), index=matrix.index)
	keep = totals[totals >= min_occurrences].index
	return matrix.loc[keep, keep]


def cluster_order(matrix):
	"""Order crops by hierarchical clustering on co-occurrence normalised by the smaller crop total."""
	values = matrix.to_numpy(dtype=float)
	totals = np.diag(values)
	denominator = np.minimum.outer(totals, totals)
	similarity = np.divide(values, denominator, out=np.zeros_like(values), where=denominator > 0)
	np.fill_diagonal(similarity, 1.0)
	similarity = np.clip((similarity + similarity.T) / 2, 0, 1)
	distance = 1 - similarity
	np.fill_diagonal(distance, 0)
	order = leaves_list(linkage(squareform(distance, checks=False), method="average"))
	return matrix.index[order]


def plot_heatmap(matrix, title, output_path):
	order = cluster_order(matrix)
	ordered = matrix.loc[order, order]
	# Diagonal holds total occurrences, which would swamp the colour scale
	display = ordered.to_numpy(dtype=float).copy()
	np.fill_diagonal(display, np.nan)
	display[display == 0] = np.nan

	size = max(10, len(ordered) * 0.17)
	fig, ax = plt.subplots(figsize=(size, size * 0.9))
	cmap = plt.get_cmap("viridis").copy()
	cmap.set_bad("#f2f2f2")
	vmax = np.nanmax(display) if np.isfinite(display).any() else 1
	image = ax.imshow(display, cmap=cmap, norm=LogNorm(vmin=1, vmax=max(vmax, 2)))
	ax.set_xticks(range(len(ordered)))
	ax.set_yticks(range(len(ordered)))
	ax.set_xticklabels(ordered.columns, rotation=90, fontsize=6)
	ax.set_yticklabels(ordered.index, fontsize=6)
	ax.set_title(title)
	fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02, label="Co-occurrences (polygons, log scale)")
	fig.tight_layout()
	fig.savefig(output_path, dpi=DPI)
	plt.close(fig)


def build_graph(matrix, min_edge_weight):
	graph = nx.Graph()
	totals = pd.Series(np.diag(matrix.to_numpy()), index=matrix.index)
	for crop, total in totals.items():
		graph.add_node(crop, total=int(total))
	values = matrix.to_numpy()
	crops = list(matrix.index)
	for i in range(len(crops)):
		for j in range(i + 1, len(crops)):
			weight = values[i, j]
			if weight >= min_edge_weight:
				graph.add_edge(crops[i], crops[j], weight=int(weight))
	graph.remove_nodes_from(list(nx.isolates(graph)))
	return graph


def cluster_layout(graph, communities):
	"""Give each cluster its own non-overlapping circle and lay its nodes out inside it."""
	radii = [0.6 + 0.35 * np.sqrt(len(c)) for c in communities]
	membership = {crop: idx for idx, c in enumerate(communities) for crop in c}

	# Place cluster centres by inter-cluster co-occurrence, then push overlapping circles apart
	quotient = nx.Graph()
	quotient.add_nodes_from(range(len(communities)))
	for u, v, d in graph.edges(data=True):
		a, b = membership[u], membership[v]
		if a != b:
			prev = quotient.get_edge_data(a, b, {"weight": 0})["weight"]
			quotient.add_edge(a, b, weight=prev + d["weight"])
	centres = nx.spring_layout(quotient, weight="weight", seed=42, iterations=200)
	centres = {k: np.array(v) * 3 * max(radii) * np.sqrt(len(communities)) for k, v in centres.items()}
	for _ in range(1000):
		moved = False
		for a in range(len(communities)):
			for b in range(a + 1, len(communities)):
				delta = centres[b] - centres[a]
				dist = np.linalg.norm(delta) or 1e-6
				gap = radii[a] + radii[b] + 0.4 - dist
				if gap > 0:
					push = delta / dist * gap / 2
					centres[a] = centres[a] - push
					centres[b] = centres[b] + push
					moved = True
		if not moved:
			break

	layout = {}
	for idx, community in enumerate(communities):
		sub = graph.subgraph(community)
		if len(community) == 1:
			local = {next(iter(community)): np.zeros(2)}
		else:
			local = nx.spring_layout(sub, weight="weight", seed=42, iterations=200)
		coords = np.array(list(local.values()))
		coords = coords - coords.mean(axis=0)
		scale = np.linalg.norm(coords, axis=1).max() or 1
		for crop, xy in local.items():
			layout[crop] = centres[idx] + (xy - np.array(list(local.values())).mean(axis=0)) / scale * radii[idx] * 0.85
	circles = [(tuple(centres[i]), radii[i]) for i in range(len(communities))]
	return layout, circles


def plot_network(graph, title, output_path):
	if graph.number_of_nodes() == 0:
		print("No network nodes after filtering; skipping network plot.")
		return

	# Clusters = Louvain communities: groups of crops that co-occur with each other
	# far more than expected by chance (modularity maximisation on co-occurrence weights).
	communities = sorted(
		nx.community.louvain_communities(graph, weight="weight", seed=42),
		key=len, reverse=True,
	)
	layout, circles = cluster_layout(graph, communities)

	totals = np.array([graph.nodes[n]["total"] for n in graph.nodes], dtype=float)
	bucket_edges = np.unique(np.quantile(totals, np.linspace(0, 1, N_BUCKETS + 1)))
	n_buckets = len(bucket_edges) - 1
	ramp = plt.get_cmap("YlOrRd")(np.linspace(0.2, 0.95, n_buckets))
	bucket_of = np.clip(np.searchsorted(bucket_edges, totals, side="right") - 1, 0, n_buckets - 1)

	weights = np.array([d["weight"] for _, _, d in graph.edges(data=True)], dtype=float)
	widths = 0.3 + 5 * np.log1p(weights) / np.log1p(weights.max())
	alphas = 0.1 + 0.5 * weights / weights.max()

	fig, ax = plt.subplots(figsize=(18, 14))
	cluster_palette = plt.get_cmap("tab20")
	for idx, ((cx, cy), radius) in enumerate(circles):
		ax.add_patch(Circle(
			(cx, cy), radius, facecolor=cluster_palette(idx % 20), alpha=0.18,
			edgecolor=cluster_palette(idx % 20), linewidth=2,
		))
		ax.text(cx, cy + radius * 1.02, f"Cluster {idx + 1}", ha="center", va="bottom", fontsize=11, fontweight="bold")

	# Draw within-cluster links faintly and between-cluster links more faintly still
	nx.draw_networkx_edges(
		graph, layout, ax=ax, width=widths,
		edge_color=[(0.25, 0.25, 0.25, a) for a in alphas],
	)
	nx.draw_networkx_nodes(
		graph, layout, ax=ax, node_size=NODE_SIZE,
		node_color=[tuple(ramp[b]) for b in bucket_of],
		edgecolors="black", linewidths=0.6,
	)

	if graph.number_of_nodes() <= NETWORK_LABEL_ALL_BELOW:
		labelled = set(graph.nodes)
	else:
		ranked = sorted(graph.nodes, key=lambda n: graph.nodes[n]["total"], reverse=True)
		labelled = set(ranked[:NETWORK_TOP_LABELS])
	nx.draw_networkx_labels(graph, layout, labels={n: n for n in labelled}, font_size=7, ax=ax)

	handles = []
	for b in range(n_buckets):
		low, high = bucket_edges[b], bucket_edges[b + 1]
		handles.append(Patch(facecolor=ramp[b], edgecolor="black", label=f"{low:,.0f} - {high:,.0f}"))
	ax.legend(handles=handles, title="Total occurrences\n(equal-count buckets)", loc="upper left", fontsize=9)

	ax.set_title(
		f"{title}\nColour = total occurrences, link width = co-occurrences, "
		f"shaded circles = detected clusters ({len(communities)})"
	)
	ax.set_aspect("equal")
	ax.autoscale_view()
	ax.axis("off")
	fig.tight_layout()
	fig.savefig(output_path, dpi=DPI)
	plt.close(fig)

	for idx, community in enumerate(communities, start=1):
		members = sorted(community, key=lambda n: graph.nodes[n]["total"], reverse=True)
		print(f"  Cluster {idx} ({len(members)}): {', '.join(members[:10])}{'...' if len(members) > 10 else ''}")


# -----------------------------------------------------------------------
# Run Script
# -----------------------------------------------------------------------

if __name__ == "__main__":
	season = str(TARGET_SEASON).strip().lower()
	input_file = build_input_path(TARGET_YEAR, season)
	label = f"{TARGET_YEAR} {season.capitalize()}"

	print(f"Loading {input_file.name}...")
	matrix = load_matrix(input_file)
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

	heatmap_matrix = filter_by_occurrences(matrix, HEATMAP_MIN_OCCURRENCES)
	heatmap_path = OUTPUT_DIR / f"{TARGET_YEAR}_{season}_crop_cooccurrence_heatmap.png"
	plot_heatmap(heatmap_matrix, f"Crop co-occurrence heatmap, {label} (clustered order)", heatmap_path)
	print(f"Wrote {heatmap_path} ({len(heatmap_matrix)} crops)")

	network_matrix = filter_by_occurrences(matrix, NETWORK_MIN_OCCURRENCES)
	graph = build_graph(network_matrix, NETWORK_MIN_EDGE_WEIGHT)
	network_path = OUTPUT_DIR / f"{TARGET_YEAR}_{season}_crop_cooccurrence_network.png"
	plot_network(graph, f"Crop co-occurrence network, {label}", network_path)
	print(f"Wrote {network_path} ({graph.number_of_nodes()} nodes, {graph.number_of_edges()} links)")
