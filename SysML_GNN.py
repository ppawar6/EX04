import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import xml.etree.ElementTree as ET
import networkx as nx
import torch
from torch_geometric.utils import from_networkx
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.express as px
from pyvis.network import Network
from sklearn.metrics.pairwise import cosine_similarity
import itertools
import copy

# === Step 1: Parse XML ===
tree = ET.parse("sysml_graph.xml")
root = tree.getroot()

G = nx.DiGraph()

# === Step 2: Load MiniLM ===
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def get_embedding(text):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    except Exception:
        return np.zeros(384)

# === Step 3: Parse Nodes and Build type_map ===
type_set = set()
for node in root.findall(".//node"):
    t = node.attrib.get("type")
    if t:
        type_set.add(t)
# Ensure 'Generic' is always present
type_set.add('Generic')
type_map = {t: i for i, t in enumerate(sorted(type_set))}
rev_type_map = {v: k for k, v in type_map.items()}

for node in root.findall(".//node"):
    nid = node.attrib.get("id")
    name = node.attrib.get("name", "").strip()
    if not name:
        name = "Unnamed"

    t = node.attrib.get("type", "Generic")
    doc = node.attrib.get("documentation", "")
    text = f"{name} {doc}".strip()

    emb = get_embedding(text)
    label = type_map.get(t, type_map["Generic"])

    G.add_node(nid,
               name=name,
               type=t,
               documentation=doc,
               x=emb.tolist(),
               y=label,
               label=name)  # Set label for PyVis and other graph use

# === Step 4: Parse Edges ===
for edge in root.findall(".//edge"):
    source = edge.attrib.get("source")
    target = edge.attrib.get("target")
    relation = edge.attrib.get("type", "relatedTo")
    if source in G.nodes and target in G.nodes:
        G.add_edge(source, target, relation=relation)

print("✅ Parsed Nodes:", len(G.nodes))
print("✅ Parsed Edges:", len(G.edges))

# Copy 'name' into 'label' for each node
for node, data in G.nodes(data=True):
    name_value = data.get("name", "")
    data["label"] = str(name_value) if name_value is not None else ""


# === Step 5: Save the Graph ===
G_export = copy.deepcopy(G)

# STEP 5.1: Forcefully replace 'label' with 'name' — no exceptions
for node_id, data in G_export.nodes(data=True):
    name_val = data.get("name", "").strip()
    data["label"] = name_val if name_val else "Unnamed"

# STEP 5.2: Clean up types in node attributes
for _, data in G_export.nodes(data=True):
    for k in list(data.keys()):
        v = data[k]
        if isinstance(v, list):
            data[k] = ", ".join(map(str, v))
        elif isinstance(v, (dict, tuple, set)):
            data[k] = str(v)
        elif not isinstance(v, (str, int, float, bool, type(None))):
            del data[k]

# STEP 5.3: Clean up types in edge attributes
for _, _, data in G_export.edges(data=True):
    for k in list(data.keys()):
        v = data[k]
        if isinstance(v, list):
            data[k] = ", ".join(map(str, v))
        elif isinstance(v, (dict, tuple, set)):
            data[k] = str(v)
        elif not isinstance(v, (str, int, float, bool, type(None))):
            del data[k]

for node, data in G_export.nodes(data=True):
    if "x" in data:
        del data["x"]
for u, v, data in G_export.edges(data=True):
    if "x" in data:
        del data["x"]


print("🔍 Sample node labels:")
for node_id, data in list(G_export.nodes(data=True))[:5]:
    print(f"{node_id}: label = '{data.get('label')}' | name = '{data.get('name')}'")

# STEP 5.4: Save as GraphML
nx.write_graphml(G_export, "sysml_model_graph.graphml")

# STEP 5.5 — PyVis Interactive HTML
net = Network(height="750px", width="100%", directed=True, notebook=False)
net.from_nx(G_export)

for node in net.nodes:
    meta = G_export.nodes[node["id"]]
    tooltip = f"Name: {meta.get('name', '')}<br>Type: {meta.get('type', '')}"
    if 'documentation' in meta and meta['documentation']:
        tooltip += f"<br><br>{meta['documentation']}"
    node["title"] = tooltip
    node["label"] = meta.get("name", "Unnamed")
    node["shape"] = "dot"
    node["size"] = 15

for edge in net.edges:
    edge["arrows"] = "to"
    if "relation" in edge:
        edge["title"] = edge["relation"]

net.write_html("sysml_model_graph_interactive.html")

# === Step 6: GNN with PyTorch Geometric ===
data = from_networkx(G)
data.x = torch.tensor([G.nodes[n]['x'] for n in G.nodes], dtype=torch.float)
data.y = torch.tensor([G.nodes[n]['y'] for n in G.nodes], dtype=torch.long)

print("Classes:", len(type_map))
print("Feature shape:", data.x.shape)

# === Step 7: Simple GCN ===
W1 = GCNConv(data.x.shape[1], 64)
W2 = GCNConv(64, len(type_map))
params = list(W1.parameters()) + list(W2.parameters())
optimizer = torch.optim.Adam(params, lr=0.01)

for epoch in range(50):
    optimizer.zero_grad()
    h = W1(data.x, data.edge_index)
    h = F.relu(h)
    out = W2(h, data.edge_index)
    loss = F.cross_entropy(out, data.y)
    loss.backward()
    optimizer.step()
    #if epoch % 10 == 0:
        #print(f"Epoch {epoch:02d}: Loss = {loss.item():.4f}")

W1.eval()
W2.eval()
with torch.no_grad():
    h = F.relu(W1(data.x, data.edge_index))
    logits = W2(h, data.edge_index)
    preds = torch.argmax(logits, dim=1).numpy()
    embeddings = logits.numpy()

# === Step 8: Save Embeddings ===
labels = [G.nodes[n]['name'] for n in G.nodes]
types = [rev_type_map[y] for y in data.y.numpy()]
df = pd.DataFrame(embeddings, index=labels)
df['type'] = types
df.to_csv("sysml_node_embeddings.csv")

# === Step 9: Visualize with t-SNE and PCA ===
if embeddings.shape[0] > 2 and embeddings.shape[1] > 2:
    tsne_result = TSNE(n_components=2, perplexity=30).fit_transform(embeddings)
    pca_result = PCA(n_components=2).fit_transform(embeddings)

    tsne_df = pd.DataFrame({'x': tsne_result[:,0], 'y': tsne_result[:,1], 'label': types, 'node': labels})
    pca_df = pd.DataFrame({'x': pca_result[:,0], 'y': pca_result[:,1], 'label': types, 'node': labels})

    px.scatter(tsne_df, x="x", y="y", color="label", hover_name="node", title="t-SNE").write_html("tsne_visualization.html")
    px.scatter(pca_df, x="x", y="y", color="label", hover_name="node", title="PCA").write_html("pca_visualization.html")
else:
    print("⚠️ Not enough samples/features for t-SNE or PCA")

# === Step 10: Link Prediction via Cosine Similarity ===
similarity_matrix = cosine_similarity(embeddings)
threshold = 0.95
existing_edges = set(G.edges())
predicted_edges = []
node_list = list(G.nodes)
for i, j in itertools.combinations(range(len(node_list)), 2):
    if (node_list[i], node_list[j]) not in existing_edges:
        sim = similarity_matrix[i, j]
        if sim > threshold:
            predicted_edges.append((node_list[i], node_list[j], sim))

link_df = pd.DataFrame(predicted_edges, columns=["source", "target", "similarity"])
link_df.to_csv("predicted_links.csv", index=False)

print("✅ Script completed.")



import xml.etree.ElementTree as ET

def parse_sysml_graph(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    nodes = []
    edges = []

    for node in root.findall('node'):
        nodes.append({
            'id': node.get('id'),
            'name': node.get('name'),
            'type': node.get('type'),
            'documentation': node.get('documentation', ''),
            'stereotypes': node.get('stereotypes', ''),
            'tags': node.get('tags', ''),
            'package': node.get('package', '')
        })

    for edge in root.findall('edge'):
        edges.append({
            'source': edge.get('source'),
            'target': edge.get('target'),
            'label': edge.get('label'),
            'roleSource': edge.get('roleSource', ''),
            'roleTarget': edge.get('roleTarget', ''),
            'multiplicitySource': edge.get('multiplicitySource', ''),
            'multiplicityTarget': edge.get('multiplicityTarget', ''),
            'aggregation': edge.get('aggregation', '')
        })

    return pd.DataFrame(nodes), pd.DataFrame(edges)
