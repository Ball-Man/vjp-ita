"""Data loading and manipulation."""
import os
from typing import List
import xml.etree.ElementTree as ET

import importlib_resources as resources


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
