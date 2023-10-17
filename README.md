# ai-orchestrator
AI orchestrator interacts with the cloud-edge manager in order to get the metrics related to resource usage and employ them to build learning models that provide an initial deployment plan according to the device requirements.

User guide:

**Installation**

The necessary python libraries and environment settings are listed below.
Python libraries:
fastapi==0.103.1
requests==2.25.1
uvicorn==0.23.2
Docker: Version 24.0.6

Building the docker container.

Building: docker build -t python/vm_placement .
Execution: docker run -p 4567:4567 --rm -it python/vm_placement

Run system state recorder

https://github.com/SovereignEdgeEU-COGNIT/ai-orchestrator.git
cd ai-orchestrator/src/system-state-recorder/bin
export ROCKET_ADDRESS=0.0.0.0
./staterec http://localhost:4567
