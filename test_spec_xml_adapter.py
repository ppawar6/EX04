"""Run with: python -m unittest -v test_spec_xml_adapter.py"""
import unittest
from pathlib import Path
from spec_xml_adapter import parse_spec_xml, build_graph

BASE = Path(__file__).resolve().parent
FILES = ['sysml_graph.xml', 'Human-CGenAI_interaction.xml',
         'Inventory+Control+System.xml', 'WaterSupply.xml']

class TestAdapter(unittest.TestCase):
    def test_each_input_produces_graph(self):
        for filename in FILES:
            with self.subTest(file=filename):
                nodes, edges = parse_spec_xml(BASE / filename)
                graph = build_graph(nodes, edges)
                self.assertGreater(len(nodes), 0)
                self.assertGreater(len(edges), 0)
                self.assertEqual(len(nodes), len({n.id for n in nodes}))
                self.assertTrue(all(e.src in graph and e.dst in graph for e in edges))
                self.assertTrue(all('name' in data and 'type' in data
                                    for _, data in graph.nodes(data=True)))
    def test_preprocessed_graph_count(self):
        nodes, edges = parse_spec_xml(BASE / 'sysml_graph.xml')
        self.assertEqual(len(nodes), 6035)
        self.assertEqual(len(edges), 546)



    
    def test_inventory_control_flow(self):
        nodes, edges = parse_spec_xml(
            BASE / 'Inventory+Control+System.xml'
        )

        graph = build_graph(nodes, edges)

        action_id = '8740266_1056969131608_328561_294'
        target_id = '8740266_1056969131623_556384_318'

        self.assertIn(action_id, graph.nodes)
        self.assertEqual(
            graph.nodes[action_id]['name'],
            'Find Shipper'
        )
        self.assertEqual(
            graph.nodes[action_id]['type'],
            'CallBehaviorAction'
        )

        self.assertTrue(graph.has_edge(action_id, target_id))
        self.assertIn(
            'ControlFlow',
            graph[action_id][target_id]['relation'].split('|')
        )

        control_flows = [
            edge for edge in edges
            if edge.type == 'ControlFlow'
        ]
        self.assertEqual(len(control_flows), 29)


    
    def test_graph_integrity(self):
        import tempfile
        import networkx as nx

        nodes, edges = parse_spec_xml(
            BASE / 'Inventory+Control+System.xml'
        )
        graph = build_graph(nodes, edges)

        # Node identifiers must be unique.
        node_ids = [node.id for node in nodes]
        self.assertEqual(len(node_ids), len(set(node_ids)))

        # Every extracted relationship must have valid endpoints.
        for edge in edges:
            with self.subTest(relation=edge.type):
                self.assertIn(edge.src, graph.nodes)
                self.assertIn(edge.dst, graph.nodes)

        # Test GraphML export and re-import.
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'test_graph.graphml'
            nx.write_graphml(graph, path)

            loaded_graph = nx.read_graphml(path)

            self.assertEqual(
                graph.number_of_nodes(),
                loaded_graph.number_of_nodes()
            )
            self.assertEqual(
                graph.number_of_edges(),
                loaded_graph.number_of_edges()
            )
    
    def test_water_supply_uml_types(self):
        nodes, edges = parse_spec_xml(
            BASE / 'WaterSupply.xml'
        )

        # No extracted UML type should be an element ID.
        invalid_types = [
            (node.id, node.type)
            for node in nodes
            if node.type.startswith('_17_0_4_')
        ]

        self.assertEqual(invalid_types, [])

        # Check actual model element classifications.
        node_types = {node.type for node in nodes}
        self.assertIn('Class', node_types)
        self.assertIn('Association', node_types)
        self.assertIn('Port', node_types)

    
    def test_inventory_use_case_relationships(self):
        from collections import Counter

        nodes, edges = parse_spec_xml(
            BASE / 'Inventory+Control+System.xml'
        )

        counts = Counter(edge.type for edge in edges)

        self.assertEqual(counts['ControlFlow'], 29)
        self.assertEqual(counts['includes'], 3)
        self.assertEqual(counts['extends'], 1)
        self.assertEqual(counts['generalizes'], 4)

        graph = build_graph(nodes, edges)

        self.assertEqual(graph.number_of_nodes(), 180)
        self.assertEqual(graph.number_of_edges(), 236)


    
    def test_inventory_profile_exclusion(self):
        nodes, edges = parse_spec_xml(
            BASE / 'Inventory+Control+System.xml'
        )

        node_ids = {node.id for node in nodes}

        profile_ids = {
            '_9_0_be00301_1108044721245_236588_411',
            'magicdraw_1046861421236_248912_50',
        }

        # Profile definitions are intentionally not imported
        # into the primary system graph.
        self.assertTrue(profile_ids.isdisjoint(node_ids))

        # Only generalizations whose endpoints are included
        # in the graph are retained.
        generalizations = [
            edge for edge in edges
            if edge.type == 'generalizes'
        ]

        self.assertEqual(len(generalizations), 4)

        for edge in generalizations:
            self.assertIn(edge.src, node_ids)
            self.assertIn(edge.dst, node_ids)



    def test_invalid_xml_raises_parse_error(self):
        import tempfile
        import xml.etree.ElementTree as ET
        from pathlib import Path

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "invalid.xml"
            path.write_text(
                "<graph><node>",
                encoding="utf-8"
            )

            with self.assertRaises(ET.ParseError):
                parse_spec_xml(path)

    def test_unsupported_root_is_rejected(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "unsupported.xml"
            path.write_text(
                "<other/>",
                encoding="utf-8"
            )

            with self.assertRaises(ValueError):
                parse_spec_xml(path)


if __name__ == '__main__':
    unittest.main()
