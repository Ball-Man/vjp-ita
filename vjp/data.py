"""Data loading and manipulation."""
import os
from typing import List, Sequence
import xml.etree.ElementTree as ET

import importlib_resources as resources
import pandas as pd
import networkx as nx
import re

OTHER_OUTCOMES_RESOURCES = 'vjp.dataset.OtherOutcomes'
SECOND_INSTANCE_REJECT_RESOURCES = 'vjp.dataset.Reject.SecondInstance'
FIRST_INSTANCE_REJECT_RESOURCES = 'vjp.dataset.Reject.FirstInstance'
SECOND_INSTANCE_UPHOLD_RESOURCES = 'vjp.dataset.Uphold.SecondInstance'
FIRST_INSTANCE_UPHOLD_RESOURCES = 'vjp.dataset.Uphold.FirstInstance'

LINK_SEPARATOR = r'|'
EDGE_RELATIONS = {'O', 'D', 'PRO', 'SUP', 'ATT', 'CON', 'REPH'}
"""Tag attributes that can be used as relations in a graph."""


def load_instance_raw(file: os.PathLike) -> ET.Element:
    """Load and return an XML instance tree, with no cleanup."""
    root =  ET.parse(file).getroot()
    root.set('source_file', file)
    return root


def load_directory(directory: resources.abc.Traversable) -> List[ET.Element]:
    """Load all instances from a given directory in a list.

    Does not explore subdirectories.
    """
    instances = []
    for file in directory.iterdir():
        if file.is_file():
            instances.append(load_instance_raw(file))

    return instances


def load_second_instance() -> List[ET.Element]:
    """Load all second instance samples from the default dataset."""
    return (
        load_directory(resources.files(SECOND_INSTANCE_UPHOLD_RESOURCES))
        + load_directory(resources.files(SECOND_INSTANCE_REJECT_RESOURCES))
    )

def findall(instances: Sequence[ET.Element], query: str) -> List[ET.Element]:
    """Execute an XPath query on all given instances.

    Return: a flattened output list of all the results and a mapping
    from the results to the corresponding queried elements.
    """
    mapping = {}
    for instance in instances:
        for result in instance.findall(query):
            mapping[result] = instance

    return list(mapping.keys()), mapping


def filter_other_outcomes(instances: List[ET.Element]) -> List[ET.Element]:
    """Return instances with at least an admissible outcome.

    Data is copied.
    """
    decisions, decision_mapping = findall(instances,
                                          ".//courtdec[@G='2']/dec")
    upheld, _ = findall(decisions, ".[@E='1']")
    rejected, _ = findall(decisions, ".[@E='0']")

    decisions = upheld + rejected
    return list(set(decision_mapping[decision] for decision in decisions))


def extract_link(element: ET.Element, key: str) -> List[str]:
    """Extract a list of linked element ids from an element's argument.

    Links are considered to be separated by a pipe
    (:attr:`LINK_SEPARATOR`).
    """
    return element.attrib[key].split(LINK_SEPARATOR)


def extract_link_elements(document: ET.Element, element: ET.Element,
                          key: str, proc=2) -> List[ET.Element]:
    """Extract a list of linked elements from an element's argument.

    Uses :func:`extract_link` to get the ids and retrieves them from
    the given document.

    ``proc`` specifies the grade (``G`` attribute) to consider. If
    set to ``None``, no filtering by grade is performed. Defaults to
    ``2``.
    """
    links = extract_link(element, key)

    link_elements = []
    for link_id in links:
        # IDs shall be unique
        query = f".//*[@G='{proc}']/*[@ID='{link_id}']"
        if proc is None:
            query = f".//*[@ID='{link_id}']"

        element = document.find(query)
        if element is not None:
            link_elements.append(element)

    return link_elements


def build_tag_triples(document: ET.Element,
                      relations: Sequence[str] = EDGE_RELATIONS
                      ) -> pd.DataFrame:
    """Build triples from an XML document, suitable to build a graph.

    Given a set of string names of attributes to explore, each attribute
    becomes a relation in a triple where the referenced tag IDs are the
    targets and the owner's ID is the source.
    Tags that are not implied in any relation are ignored.
    """
    # Perform a DFS starting, starting from all sources having the
    # desired attribute
    fringe = sum(
        (document.findall(f'.//*[@{relation}]') for relation in relations),
        start=[])

    triples = set()

    while fringe:
        source = fringe.pop()

        # For each relation defined by the element
        for relation in EDGE_RELATIONS.intersection(source.attrib):
            targets = extract_link_elements(document, source, relation)

            # Generate a triple for each target of said relation
            # If a duplicate is found, drop current element, it's an
            # infinite loop
            for target in targets:
                triple = (source.get('ID'), target.get('ID'), relation)
                if triple in triples:
                    break

                triples.add(triple)
                fringe.append(target)

    return pd.DataFrame(triples, columns=['source', 'target', 'edge'])


def tagid_in_sequence(tagid: str, tag_names: Sequence[str]) -> int:
    """Return whether the given tag ID is of one of the given types.

    The index of the related name in the sequence is returned (``-1``
    if not found).
    """
    for i, prefix in enumerate(tag_names):
        if tagid.lower().startswith(prefix):
            return i

    return -1

def get_node_sub_text(node_element: ET.Element) -> str:
    ret = "".join([get_node_sub_text(child) for child in list(node_element) if child.text is not None])
    if node_element.text is not None:
        return (re.sub('\s+', ' ', node_element.text) + " " + ret).strip()
    return ret

def dataframe_from_graphs(
        graphs: Sequence[nx.Graph],
        samples: Sequence[ET.Element],
        tag_names: Sequence[str] = ('req', 'arg', 'claim'),
        use_child_text_tag_names: Sequence[str] = ('mot', 'dec'),
        join_token: str = ' ') -> pd.DataFrame:
    """Given a sequence of graphs and corresponding samples, build dataframe.

    The resulting dataframe is constructed by looking at the connected
    components of each graph. From each connected component one or more
    samples can be generated, based on the number of requests in said
    component. All records are concatenated together.

    Columns: one per considered tag type (found in the connected
    component). The ``fact`` tag is always included, even though it is
    not found in any connected component (global knowledge).
    A final column for the label (0 -> rejected, 1 -> uphold).
    The data for each cell is given by the concatenation of all the text
    contained by each tag, divided by tag type (column) and connected
    component (row).
    """
    assert len(graphs) == len(samples), (
        'Graph and sample lists must have the same amount of elements')

    df_list = []
    for document_index, (graph, document) in enumerate(zip(graphs, samples)):
        for component in tuple(nx.connected_components(graph.to_undirected())):
            # Extract decisional tags
            dec_ids = tuple(filter(
                lambda id_: tagid_in_sequence(id_, ('dec',)) == 0,
                component))

            # Decisions represent labels, skip if none is found
            if not dec_ids:
                continue

            # Also skip if the label is not 0 or 1
            # NB: How to handle multiple decisions in one connected component?
            label = -1
            dec_element = document.find(f".//*[@ID='{dec_ids[0]}']")
            try:
                label = int(dec_element.get('E'))
                if label not in (0, 1):
                    raise ValueError
            except ValueError:
                continue

            # Build a sequence per tag type
            concat_lists = [[] for _ in tag_names]

            node_indeces = map(lambda id_: tagid_in_sequence(id_, tag_names),
                               component)

            for node, index in zip(component, node_indeces):
                node_element = document.find(f".//*[@ID='{node}']")
                if index >= 0:
                    if any(node.lower().startswith(tag)
                           for tag in use_child_text_tag_names):
                        concat_lists[index].append(
                            get_node_sub_text(node_element))
                    elif node_element.text is not None:
                        concat_lists[index].append(
                            re.sub(r'\s+', ' ', node_element.text).strip())

            # Add fact column
            fact_element = document.find(".//fact")
            fact = ''
            if fact_element is not None:
                fact = fact_element.text

            req_prefix_index = tag_names.index('req')
            for req_text in concat_lists[req_prefix_index]:
                df_list.append([document_index, fact, *map(join_token.join,
                                                           concat_lists),
                                label])
                df_list[-1][1 + req_prefix_index] = req_text

    return pd.DataFrame(df_list, columns=['document_index', 'fact', *tag_names,
                                          'label'])


def sort_documents(documents: Sequence[ET.Element]) -> Sequence[ET.Element]:
    return sorted(documents, key=lambda x: x.get('source_file'))
