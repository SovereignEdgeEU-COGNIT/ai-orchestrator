import os
import xml.etree.ElementTree as ET
import json

# Function to convert XML element to JSON dictionary
def xml_to_json(element):
    if len(element) == 0:
        return element.text if element.text else "none"
    result = {}
    for child in element:
        child_data = xml_to_json(child)
        if child.tag in result:
            if isinstance(result[child.tag], list):
                result[child.tag].append(child_data)
            else:
                result[child.tag] = [result[child.tag], child_data]
        else:
            result[child.tag] = child_data
    return result

# Define the directory containing XML files
xml_directory = r"/home/zhou/project/VM_Orchestration-main/OpenNebula-datasets/OpenNebula-datasets/vmindividualshow"

# Define the directory to save JSON files
json_directory = r"/home/zhou/repo/VM_Orchestration/fake_input"


# Loop through XML files in the directory
for xml_filename in os.listdir(xml_directory):
    if xml_filename.endswith(".json"):
        xml_path = os.path.join(xml_directory, xml_filename)

        # Parse XML
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Convert XML to JSON
        json_data = xml_to_json(root)
        #print(json_data["MONITORING"]["DISK_SIZE"])

        # Create JSON file path
        json_filename = os.path.splitext(xml_filename)[0] + ".json"
        json_path = os.path.join(json_directory, json_filename)

        # Apply fixed template structure
        fake_input = { 
    "$schema": "http://json-schema.org/2020‐12/schema#",
    "type": "object",
    "properties": {
        "SERVERLESS_RUNTIME": {
            "type": "object",
            "properties": {
                "VM": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "VM_ID": int(json_data["ID"]),
                            "SERVICE_ID": {
                              "type": "integer"
                            },
                            "STATUS": {
                                "type": "string"
                            },
                            "HOSTS": {
                                "type": "array",
                                "properties": {
                                    "HOST_ID": {
                                        "type": "integer"
                                    }
                                }
                            },
                            "REQUIREMENTS": {
                                "type": "object",
                                "properties": {
                                    "CPUS": float(json_data["MONITORING"]["CPU"]),
                                    "FLOPS": {
                                        "type": "integer"
                                    },
                                    "MEMORY": int(json_data["MONITORING"]["MEMORY"]),
                                    "DISK_SIZE": {
                                        "type": "integer"
                                    },
                                    "IOPS" : {
                                        "type": "integer"
                                    },
                                    "LATENCY" : {
                                        "type": "integer"
                                    },
                                    "BANDWIDTH": { 
                                        "type": "integer"
                                    },
                                    "ENERGY": {
                                        "type": "integer"
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

}

        # Write JSON data to file
        with open(json_path, "w") as json_file:
            json.dump(fake_input, json_file, indent=4)


