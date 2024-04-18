# Introduction

![AI-O Architecture](..\AI-OArchitecturev2.jpg)

## Algorithms:

### Interference-Aware Scheduling
* Let $H={h_1,h_2,…,h_n}$ be the set of hosts
* Let $R={r_1,r_2,…,r_m }$ be the set of resources (CPU, memory, etc)
* Let $U_{h_{i}r_{j}}$ be the utilization of resource $r_{j}$  at host $h_{i}$ 
* Let $V_{r_{j}}$ be the resource demand of the VM for resource $r_{j}$
* Let $D_{h_{i}}$ be the distance measure for host $h_{i}$ 
We are trying to minimize:
$$D_{max}=max⁡(D_{h_{1}},D_{h_{2}},…,D_{h_{n}})$$
To calculate $D_{h_{i}}$ we use:

$$D_{h_{i}} = \sum_{j=1}^m|(U_{h_{i}r_{j}}+V_{r_j} ) - \overline{U_{r_{j}}}| $$

Where $\overline{U_{r_{j}}}$ denotes the average utilization of resource $r_j$ across all hosts.

### Calculating the VM resource demand
To calculate $D_{h_{i}}$ we need $V_{r_{j}}$, this requires an understanding of the resource consumption of each VM. However, having the user specify the utilization manually is inconvenient and potentially error-prone. Thus, we are instead inferring it from the historical resource usage of the VMs. As the raw time series data of a VM is high entropy  we use the classes of classifiers to provide a lower entropy representation of a VM. For example, calculating a VM’s distance to each of the classes gives us a proxy for its resource utilization, assuming the classes represent distinct types of resource utilization. If we have three classes $(c_1,c_2,c_3)$ for the classifier we get that $R={d(c_1),d(c_2),d(c_3)}$ where the function $d$ measure the distance to the center of the cluster.

Similarly, the intermediate representation of the Auto-Encoder (AE) should represent the resource utilization of the VM. These are not directly representative of the typical resource metrics such as CPU and memory, but we are exploring methods to relate them to these metrics. For example, by looking at the change in resource utilization at the host when deploying VMs with known vectors one can estimate the effect of each vector index relative to the host resources, thus giving us scalars from VM vectors to traditional metrics such as CPU. If we have the traditional metrics $R_{trad}={r_1,r_2,…,r_m }$ and the classifier metrics $R_{class}={d(c_1),d(c_2),d(c_3)}$ then we hope to be able to estimate the scalars $S_i={s_{1r_i},s_{2r_i},s_{3r_i}}$ to go from $R_{class}$ to $R_{trad}$ by $r_i=s_{1r_i}*c_1+ s_{2r_i}*c_2+ s_{3r_i}*c_3$.

### Green-energy adjustment
Green-energy adjustment of $D_{h_{i}}$:
* Let $g(h_i)$ give the green energy percentage of the host
* Let $s_{green}$ be the green-energy scalar
$$D_{green_{h_i}}= D_{h_{i}}+D_{h_{i}}*(1-g(h_i))* s_{green}$$
This could be extended to any arbitrary cost function F as such, allowing for smarter scheduling in the future:
$$D_{green_{h_i}}= D_{h_{i}}+F(g(h_i),D_{h_{i}},...)$$

## APIs

### Schedule VM
**Method:** POST\
**URL:** /api/classifier\
**JSON:**
```json
{
  "ID":1234,
  "HOST_IDS":[1,2,3]
}
```
**Model options:**\
ID is the VM id\
Host ids are the valid hosts for the VM\
**Return:** 200 successful and:
```json
{
  "ID":1234,
  "HOST_ID":3
}
```
or 400 malformed request

### Set Classifier
**Method:** POST\
**URL:** /api/classifier\
**JSON:**
```json
{
  "model_name":"see_options_below"
}
```
**Model options:**\
| Name | Description                           |
|--------|---------------------------------------|
| RandomClassifier | Randomly generates the resource utilization for the VM. |
| DLIR    | Calculates the VM resource utilization from the intermediate representation of the AutoEncoder trained for IDEC. |
| DLClassifier    | Calculates the VM resource utilization from the distance to classes using IDEC. |
| ClassicalClassifier    | Calculates the VM resource utilization from the distance to classes using MC2PCA. |\
**Return:** 200 successful, 400 malformed request

### Set Scheduler
**Method:** POST\
**URL:** /api/scheduler\
**JSON:**
```json
{
  "model_name":"see_options_below"
}
```
**Model options:**\
| Name | Description                           |
|--------|---------------------------------------|
| RandomScheduler | Schedules the VM on a random host in the valid host list. |
| InteferenceAwareScheduler | Schedules VM using Interference-Aware Scheduling. Selecting the optimal host by minimizing interference on hosts through distributing VM's of similar resource profiles on hosts |\
**Return:** 200 successful, 400 malformed request

### Set Energy Scalar
**Method:** POST\
**URL:** /api/iaEnergyScalar\
**JSON:**\
```json
{
  "energy_cost_scalar":1
}
```
**Model options:**\
Any integer to set the $s_{green}$ to\
**Return:** 200 successful, 400 malformed request


# Building
The AI-O can either be ran locally or as a docker container. Container can be pulled from [here_LINK_MISSING]() or built. 
To build the container run:

```bash
cd ./MLServer
docker build -t placeholder/MLServer .
```

To run the server locally:
```bash
pip install -r ./requirements.txt
python ./server.py
```

# Running

The MLServer requires the following env. variables to function properly, it expects to be able to contact both Oned and the Timeseries DB.

```bash
export ENVSERVER_DB_HOST="addr"     # Address to the Timescale DB
export ENVSERVER_DB_USER="postgres" # User credentials for the Timescale DB
export ENVSERVER_DB_PORT="5432"     # Port for the Timescale DB
export ENVSERVER_DB_PASSWORD="pass" # User credentials for the Timescale DB
export ONED_PASS="pass"             # User credentials for OneD
export ONED_ADDR="addr"             # Address to OneD
export ML_MODEL_PORT="50090"        # Port for the MLServer
```

Running the docker container:
```bash
docker run --env-file /path/to/env-file placeholder/MLServer
```
