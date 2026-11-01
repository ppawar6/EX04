
"""EX04 Person 1: namespace-aware XML/XMI to unified graph adapter.

Uses only Python standard library and NetworkX. No ML dependencies required.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass, field
from pathlib import Path
import xml.etree.ElementTree as ET
import networkx as nx

XMI = '{http://www.omg.org/spec/XMI/20131001}'
UML = '{http://www.omg.org/spec/UML/20131001}'


@dataclass
class Node:
    id: str
    type: str
    attrs: dict = field(default_factory=dict)


@dataclass
class Edge:
    src: str
    dst: str
    type: str
    attrs: dict = field(default_factory=dict)


def local_name(tag: str) -> str:
    return tag.rsplit('}', 1)[-1]



def attr(element, name: str, default='') -> str:
    """Read plain or XMI namespaced attributes."""
    for key in (
        name,
        '{http://www.omg.org/spec/XMI/20131001}' + name,
        '{http://www.omg.org/XMI}' + name,
    ):
        value = element.get(key)
        if value is not None:
            return value
    return default



def parse_spec_xml(filename: str | Path) -> tuple[list[Node], list[Edge]]:
    """Read flat graph XML or MagicDraw XMI and return normalized nodes/edges.

    Explicitly represented XML relationships only. No LLM-inferred links.
    Dangling references are skipped; no implicit placeholder nodes.
    """
    root = ET.parse(filename).getroot()

    if local_name(root.tag) == 'graph':
        nodes = []
        for e in root.findall('./node'):
            nid = e.get('id')
            if nid:
                nodes.append(Node(nid, e.get('type') or 'Generic', {
                    'name': e.get('name') or 'Unnamed',
                    'documentation': e.get('documentation') or '',
                }))

        ids = {n.id for n in nodes}
        edges = []

        for e in root.findall('./edge'):
            s, t = e.get('source'), e.get('target')
            if s in ids and t in ids:
                edges.append(
                    Edge(s, t, e.get('type') or e.get('label') or 'relatedTo')
                )

        return nodes, edges

    if local_name(root.tag) != 'XMI':
        raise ValueError(f'Unsupported XML root: {root.tag}')

    # Restrict extraction to the UML model,
    # not MagicDraw diagram metadata/extensions.
    models = [e for e in root if e.tag == UML + 'Model']

    if not models:
        raise ValueError('No UML Model element found in XMI')

    nodes_by_id = {}
    edges = []
    parent_by_child = {}

    for model in models:
        for parent in model.iter():
            for child in parent:
                parent_by_child[id(child)] = parent

        for e in model.iter():
            nid = attr(e, 'id')
            tag = local_name(e.tag)
            typ = e.get(XMI + 'type') or tag

            if nid and tag not in {
                'memberEnd', 'generalization', 'ownedEnd'
            }:
                # Include modeled elements,
                # avoiding non-semantic XMI subelements.
                if tag in {
                    'Model',
                    'packagedElement',
                    'ownedAttribute',
                    'ownedOperation',
                    'ownedConnector',
                    'nestedClassifier',
                    'ownedRule',
                    'ownedBehavior',
                    'node' 
                }:
                    nodes_by_id[nid] = Node(
                        nid,
                        typ.removeprefix('uml:'),
                        {
                            'name': e.get('name') or f'{typ}:{nid[-8:]}',
                            'documentation': e.get('documentation') or '',
                        }
                    )

    ids = set(nodes_by_id)

    for model in models:
        for parent in model.iter():
            parent_id = attr(parent, 'id')

            for child in parent:
                child_id = attr(child, 'id')
                child_tag = local_name(child.tag)

                if (
                    parent_id in ids
                    and child_id in ids
                    and child_tag in {
                        'packagedElement',
                        'ownedAttribute',
                        'ownedOperation',
                        'ownedConnector',
                        'nestedClassifier',
                        'ownedRule',
                        'ownedBehavior'
                    }
                ):
                    edges.append(
                        Edge(parent_id, child_id, 'contains')
                    )

            if parent_id in ids:
                for key, relation in [
                    ('type', 'typedBy'),
                    ('client', 'dependsOn'),
                    ('supplier', 'supplies'),
                    ('general', 'generalizes')
                ]:
                    ref = parent.get(key)

                    if ref in ids:
                        edges.append(
                            Edge(parent_id, ref, relation)
                        )

                for child in parent:
                    if local_name(child.tag) == 'generalization':
                        target = child.get('general')

                        if target in ids:
                            edges.append(
                                Edge(parent_id, target, 'generalizes')
                            )

            # Association memberEnd references point to properties.
            # Their type attributes refer to modeled element IDs.
            if (
                local_name(parent.tag) == 'packagedElement'
                and attr(parent, 'type') == 'uml:Association'
            ):
                end_ids = [
                    attr(c, 'idref')
                    for c in parent
                    if local_name(c.tag) == 'memberEnd'
                ]

                resolved_types = []

                for end_id in end_ids:
                    for child in model.iter():
                        if (
                            attr(child, 'id') == end_id
                            and child.get('type') in ids
                        ):
                            resolved_types.append(
                                child.get('type')
                            )
                            break

                if (
                    len(resolved_types) == 2
                    and resolved_types[0] != resolved_types[1]
                ):
                    edges.append(
                        Edge(
                            resolved_types[0],
                            resolved_types[1],
                            'association',
                            {
                                'association_id': attr(parent, 'id')
                            }
                        )
                    )
    


    # Extract UML activity flow relationships.
    for model in models:
        for element in model.iter():
            if local_name(element.tag) != 'edge':
                continue

            edge_type = attr(element, 'type')

            if edge_type not in {'uml:ControlFlow', 'uml:ObjectFlow'}:
                continue

            source = element.get('source')
            target = element.get('target')

            if source in ids and target in ids:
                edges.append(
                    Edge(
                        source,
                        target,
                        edge_type.removeprefix('uml:'),
                        {'xmi_id': attr(element, 'id')}
                    )
                )

    # Extract UML UseCase include and extend relationships.
    for model in models:
        for parent in model.iter():
            parent_id = attr(parent, 'id')

            if parent_id not in ids:
                continue

            for child in parent:
                child_tag = local_name(child.tag)

                if child_tag == 'include':
                    target = child.get('addition')

                    if target in ids:
                        edges.append(
                            Edge(parent_id, target, 'includes')
                        )

                elif child_tag == 'extend':
                    target = child.get('extendedCase')

                    if target in ids:
                        edges.append(
                            Edge(parent_id, target, 'extends')
                        )


    # De-duplicate edges while preserving relation types.
    seen = set()
    result_edges = []

    for edge in edges:
        key = (edge.src, edge.dst, edge.type)

        if key not in seen:
            seen.add(key)
            result_edges.append(edge)

    return list(nodes_by_id.values()), result_edges


def build_graph(nodes: list[Node], edges: list[Edge]) -> nx.DiGraph:
    """Downstream contract: node name/type/documentation/label,
    edge relation.
    """
    graph = nx.DiGraph()

    for n in nodes:
        graph.add_node(
            n.id,
            name=n.attrs.get('name', 'Unnamed'),
            type=n.type,
            documentation=n.attrs.get('documentation', ''),
            label=n.attrs.get('name', 'Unnamed')
        )

    for e in edges:
        if e.src in graph and e.dst in graph:
            # DiGraph stores one edge per ordered pair.
            # Annotate multiple relations.
            if graph.has_edge(e.src, e.dst):
                prior = graph[e.src][e.dst]['relation'].split('|')

                if e.type not in prior:
                    graph[e.src][e.dst]['relation'] += '|' + e.type
            else:
                graph.add_edge(
                    e.src,
                    e.dst,
                    relation=e.type
                )

    return graph


def main():
    parser = argparse.ArgumentParser(
        description='Convert graph XML or MagicDraw XMI to NetworkX'
    )

    parser.add_argument('xml_file')
    parser.add_argument(
        '--out',
        default='spec_graph.graphml'
    )

    args = parser.parse_args()

    nodes, edges = parse_spec_xml(args.xml_file)
    graph = build_graph(nodes, edges)

    nx.write_graphml(graph, args.out)

    print(
        f'{Path(args.xml_file).name}: '
        f'{graph.number_of_nodes()} nodes, '
        f'{graph.number_of_edges()} directed graph edges; '
        f'saved {args.out}'
    )


if __name__ == '__main__':
    main()
