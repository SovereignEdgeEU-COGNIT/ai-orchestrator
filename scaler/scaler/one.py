# Copyright 2002-2025, OpenNebula Project, OpenNebula Systems
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging
from .gate import OneGateClient
from .config import ServiceState
import time

logger = logging.getLogger(__name__)


class OpenNebulaBackend:
    """
    A wrap-up around OpenNebula backend.
    """

    def __init__(self):
        """
        Initializes the scaler.
        """
        logger.debug("Initializing OneGate python client for scaling")
        self.client = OneGateClient()

    def get_role_cardinality(self, role_name: str) -> int:
        """
        Gets the current cardinality of a given role.
        Returns 0 if the role is not found.
        """
        service_info = self.client.get("service")
        roles = service_info.get("SERVICE", {}).get("roles", [])
        for role in roles:
            if role.get("name", "").lower() == role_name.lower():
                return int(role.get("cardinality"))

        logger.warning(f"Role '{role_name}' not found in service.")
        return 0

    def scale_role(self, role_name: str, cardinality: int, timeout: int = 300, interval: int = 5):
        """
        Scales a service role to a specific cardinality.
        """
        service_info = self.client.get("service")
        service_state = service_info.get("SERVICE", {}).get("state")
        current_cardinality = self.get_role_cardinality(role_name)

        logger.info(
            f"Service role '{role_name}' is at cardinality {current_cardinality}"
        )

        if current_cardinality == cardinality:
            logger.info(
                f"Service role '{role_name}' is already at cardinality {cardinality}"
            )
            return
    
        # Waiting until the service is running
        while service_state != ServiceState.RUNNING.value:
            logger.info(
                f"Service is not in 'RUNNING' state (current: {service_state}) and cannot be scaled. Waiting for {interval} seconds. Max timeout: {timeout} seconds"
            )
            time.sleep(interval)
            service_info = self.client.get("service")
            service_state = service_info.get("SERVICE", {}).get("state")
            if timeout > 0:
                timeout -= interval
            else:
                logger.error(f"Service is not in 'RUNNING' state after {timeout} seconds. Exiting...")
                return

        logger.info(
            f"Scaling role '{role_name}' from {current_cardinality} to {cardinality} nodes"
        )
        self.client.scale(cardinality, role_name)
