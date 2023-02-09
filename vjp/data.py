"""Data loading and manipulation."""
import os
from typing import List, Sequence
import xml.etree.ElementTree as ET

import importlib_resources as resources

OTHER_OUTCOMES_RESOURCES = 'vjp.dataset.OtherOutcomes'
SECOND_INSTANCE_REJECT_RESOURCES = 'vjp.dataset.Reject.SecondInstance'
FIRST_INSTANCE_REJECT_RESOURCES = 'vjp.dataset.Reject.FirstInstance'
SECOND_INSTANCE_UPHOLD_RESOURCES = 'vjp.dataset.Uphold.SecondInstance'
FIRST_INSTANCE_UPHOLD_RESOURCES = 'vjp.dataset.Uphold.FirstInstance'

LINK_SEPARATOR = r'|'


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
    """
    links = extract_link(element, key)

    link_elements = []
    for link_id in links:
        # IDs shall be unique
        element = document.find(f".//partreq[@G='{proc}']/*[@ID='{link_id}']")
        if element is not None:
            link_elements.append(element)

    return link_elements
