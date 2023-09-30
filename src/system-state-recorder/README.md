# Introduction
The **System State Recorder** provides contextual information obtained from the OpenNeubula system. Is used by the AI Orchestrator to make placement decisions. 

# Installation
```console
export ROCKET_ADDRESS=0.0.0.0
export PROMETHEUS_URL="http://localhost:9090"
cargo run
```
Note that there is also an .env file checked into the repo.

## Building
```console
cargo build --release
```
The binary can find here: ./target/release/staterec

# Usage 
## Get current placement layout
```console
curl http://localhost:8000
```
```json
[
   {
      "hostid":"0",
      "vmids":[
         
      ],
      "state":{
         "renewable_energy":false
      }
   },
   {
      "hostid":"4",
      "vmids":[
         "12",
         "7"
      ],
      "state":{
         "renewable_energy":false
      }
   },
   {
      "hostid":"5",
      "vmids":[
         
      ],
      "state":{
         "renewable_energy":false
      }
   },
   {
      "hostid":"3",
      "vmids":[
         "6"
      ],
      "state":{
         "renewable_energy":false
      }
   },
   {
      "hostid":"2",
      "vmids":[
         "5"
      ],
      "state":{
         "renewable_energy":false
      }
   }
]
```

## Get information about a host
```console
curl http://localhost:8000/hosts/2
```
```json
{"state":{"renewable_energy":false},"total_mem_bytes":16785711104,"usage_mem_bytes":805306368,"cpu_total":1600,"cpu_usage":100,"powerstate":2,"vms":"2"}
```

## Set a host to use renewable energy
```console
curl -X PUT "http://localhost:8000/set?hostid=2&renewable=true"
```

If we now get host info, the renewable_energy attribute will be set to true.

# Simulator
## Starting simulator
The URL is the location where the AI Orchestrator runs.
```console
./staterec --sim http://localhost:5678   
```

or alternatively:
```console
cargo run -- --sim http://localhost:4567   
```

## Adding a host
To add a host with id=1 and 8GB RAM (1073741824 bytes) and 16 CPU cores.
```console
curl -X POST "http://localhost:8000/addhost/?hostid=1&mem=8073741824&cpu=16000"
```

## Deploy a VM
To add a VM with id=1 to host with id=1. Allocate 1GB RAM (1073741824 bytes) and 2 CPU cores. 
```console
curl -X POST "http://localhost:8000/placevm?vmid=13&mem=1073741824&cpu=2000"
```

## Delete a VM
Delete VM with id 11.
```console
curl -X DELETE "http://localhost:8000/vms/11
```

# Testbed
## Accessing Grafana
```console
ssh -v -N -L 3000:10.10.10.2:3000 root@194.28.122.112
```

Then open browser: http://localhost:3000.

## Accessing Prometheus
```console
ssh -v -N -L 9090:10.10.10.2:9090 root@194.28.122.112
```

Prometheus is available at http://localhost:9090.

## Accessing System State Recorder
```console
ssh -v -N -L 8000:10.10.10.3:8000 root@194.28.122.112
```

To access it from all hosts at the network.
```console
ssh -v -N -L 0.0.0.0:8000:10.10.10.3:8000 root@194.28.122.112
```

The State Recorder is available at http://localhost:8000.

## Logging in to the scheduler VM
```console
ssh -J root@194.28.122.112 root@10.10.10.3
```
