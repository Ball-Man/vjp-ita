"""Data loading and manipulation."""
import os
from typing import List, Sequence
import xml.etree.ElementTree as ET

import importlib_resources as resources
import pandas as pd

OTHER_OUTCOMES_RESOURCES = 'vjp.dataset.OtherOutcomes'
SECOND_INSTANCE_REJECT_RESOURCES = 'vjp.dataset.Reject.SecondInstance'
FIRST_INSTANCE_REJECT_RESOURCES = 'vjp.dataset.Reject.FirstInstance'
SECOND_INSTANCE_UPHOLD_RESOURCES = 'vjp.dataset.Uphold.SecondInstance'
FIRST_INSTANCE_UPHOLD_RESOURCES = 'vjp.dataset.Uphold.FirstInstance'

LINK_SEPARATOR = r'|'
EDGE_RELATIONS = {'O', 'D', 'PRO', 'SUP', 'ATT'}
"""Tag attributes that can be used as relations in a graph."""


def load_instance_raw(file: os.PathLike) -> ET.Element:
    """Load and return an XML instance tree, with no cleanup."""
    return ET.parse(file).getroot()


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
