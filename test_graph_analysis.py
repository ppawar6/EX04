import unittest
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class TestGraphAnalysis(unittest.TestCase):

    def test_orphan_node_detection(self):
        G = nx.DiGraph()
        G.add_nodes_from(["A", "B", "C"])
        G.add_edge("A", "B")

        isolated_nodes = list(nx.isolates(G))

        self.assertEqual(isolated_nodes, ["C"])

    def test_candidate_links_skip_reverse_relationship(self):
        G = nx.DiGraph()
        G.add_nodes_from(["A", "B", "C"])

        # Existing relationship is B -> A.
        G.add_edge("B", "A")

        node_list = list(G.nodes)
        embeddings = np.array([
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0]
        ])

        similarity_matrix = cosine_similarity(embeddings)
        existing_edges = set(G.edges())
        predicted_edges = []

        for i in range(len(node_list)):
            for j in range(i + 1, len(node_list)):
                source = node_list[i]
                target = node_list[j]

                forward_exists = (source, target) in existing_edges
                reverse_exists = (target, source) in existing_edges

                if forward_exists or reverse_exists:
                    continue

                if similarity_matrix[i, j] > 0.95:
                    predicted_edges.append((source, target))

        # A-B must not be suggested because B -> A already exists.
        self.assertNotIn(("A", "B"), predicted_edges)

        # A-C should still be a valid candidate.
        self.assertIn(("A", "C"), predicted_edges)


if __name__ == "__main__":
    unittest.main() 