#
# (C) Copyright Cloudlab URV 2024
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import logging
import requests

logger = logging.getLogger(__name__)


class OneGateError(Exception):
    """General exception for OneGate-related errors."""

    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


class OneGateClient:
    def __init__(self):
        self.endpoint = os.getenv(
            "ONEGATE_ENDPOINT", self.get_config("ONEGATE_ENDPOINT")
        )
        self.token = self.get_config("TOKENTXT")
        self.vm_id = self.get_config("VMID")

    @staticmethod
    def get_config(param, filepath="/var/run/one-context/one_env"):
        with open(filepath, "r") as file:
            for line in file:
                line = line.strip()
                if line.startswith("export "):
                    line = line[len("export ") :]

                if line.startswith(f"{param}="):
                    return line.split("=", 1)[1].strip().strip("'\"")
        return None

    def get(self, path):
        """
        Make a GET request to OneGate API and return the JSON response.
        """
        url = f"{self.endpoint}/{path}"
        headers = {"X-ONEGATE-TOKEN": self.token, "X-ONEGATE-VMID": self.vm_id}
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            status_code = e.response.status_code if e.response else None
            raise OneGateError(f"GET request to {url} failed: {e}", status_code)
        except ValueError as e:
            raise OneGateError(f"Failed to parse JSON response: {e}")

    def scale(self, cardinality, role="worker"):
        """
        Make a PUT request to OneGate API.
        """
        url = f"{self.endpoint}/service/role/{role}"
        headers = {
            "X-ONEGATE-TOKEN": self.token,
            "X-ONEGATE-VMID": self.vm_id,
            "Content-Type": "application/json",
        }
        data = {"cardinality": cardinality}
        try:
            response = requests.put(url, headers=headers, json=data)
            response.raise_for_status()
            logger.info(f"Completed scaling role {role} to cardinality {cardinality}!")
        except requests.exceptions.RequestException as e:
            status_code = e.response.status_code if e.response else None
            raise OneGateError(f"PUT request to {url} failed: {e}", status_code)
