import argparse

import json
from .xml_tool import UIXMLTree

def compressed_xml_android(xml, type="plain_text", from_file=False, version="v1"):
    xml_parser = UIXMLTree()
    if not from_file:
        xml_str = xml
    else:
        with open(xml, 'r', encoding='utf-8') as f:
            xml_str = f.read()

    compressed_xml, element_location = xml_parser.process(xml_str, level=2, str_type=type)
        
    if isinstance(compressed_xml, tuple):
        compressed_xml = compressed_xml[0]

    if type == "plain_text":
        compressed_xml = compressed_xml.strip()
    # except Exception as e:
    #     compressed_xml = None
    #     print(f"XML compressed failure: {e}")
    return compressed_xml, element_location
