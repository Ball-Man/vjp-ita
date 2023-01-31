"""Data loading and manipulation."""
import os
import xml.etree.ElementTree as ET

import importlib_resources as resources


def load_instance_raw(file: os.PathLike) -> ET.Element:
    """Load and return an XML instance tree, with no cleanup."""
    return ET.parse(file).getroot()
