[![Go](https://github.com/SovereignEdgeEU-COGNIT/ai-orchestrator-env/actions/workflows/go.yml/badge.svg)](https://github.com/SovereignEdgeEU-COGNIT/ai-orchestrator-env/actions/workflows/go.yml)

![AI-O architecture](..\AI-OArchitecturev2.jpg)

# Introduction
This repo provides a virtual environment, functioning as a *digital twin* of a cloud-edge infrastructure, specifically designed to enable AI agents to automate and optimize IT operations. The environment enables AI agents to both operate and learn to manage IT infrastructures. 

The virtual environment serves multiple purposes:

* **Data collection:** Connect to a real infrastructure to gather real-time data, essential for training AI agents.
* **Simulation:** Utilizing the collected data, AI agents can be trained within the virtual environment, allowing them to learn and adapt to various situations without impacting the real infrastructure.
* **Deployment:** Post-training, by connecting to a real infrastrucure, the AI agents can be deployed to leverage their learned strategies and to automate and enhance IT operations.

The environment can be used to generate *state and action spaces*. 
* The state-space represents configurations (e.g. hosts, VMs, CPU load etc) that an AI agent might encounter in the environment. Each state is a unique snapshot of the environment at a given time. 
* The action space defines the set of all possible actions (e.g. place or scale a VM) that the AI agent can take at any given state. 

# Build
See Makefile for further options

```bash
make build # Builds the project, requires go installed
# Built to ./bin/envcli
```

```bash
make container # Builds the project and creates container (modify BUILD_IMAGE for alternative name)
```

# Testing
Requires setting the following env variables for testing the integrations

```bash
export PROMETHEUS_HOST="IP"
export PROMETHEUS_PORT="PORT"
```

**Core testing:**
```bash
go test ./pkg/core
```

**Database testing:**
```bash
go test ./pkg/database
```

**Emulator connector integration**
Set PROMETHEUS env variables as above for emulator
```bash
go test ./pkg/emulator
```

**OpenNebula connector integration**
Set PROMETHEUS env variables as above for OpenNebula
```bash
go test ./pkg/opennebula
```

**DB Manager**
```bash
go test ./pkg/server
```


# Getting started
To use the CLI, you most source the following environmental variables:

```console
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export LC_CTYPE=UTF-8
export TZ=Europe/Stockholm
export ENVSERVER_VERBOSE="false"
export ENVSERVER_HOST="localhost"
export ENVSERVER_PORT="50080"
export ENVSERVER_TLS="false"
export ENVSERVER_DB_HOST="localhost"
export ENVSERVER_DB_USER="postgres"
export ENVSERVER_DB_PORT="50070"
export ENVSERVER_DB_PASSWORD="rFcLGNkgsNtksg6Pgtn9CumL4xXBQ7"
```

Or simply:
```console
source .env
```

## Starting an Envserver
First start a TimescaleDB instance.

```console
docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=rFcLGNkgsNtksg6Pgtn9CumL4xXBQ7 --restart unless-stopped timescale/timescaledb:latest-pg16
```

```console
envserver start -v
```

## Docker-compose
```console
dockercompose up
```

To remove all data, type:
```console
docker-compose down --volumes
```

## Initialize TimescaleDB database
```console
envserver database create
```

The database can be droped by this command:
```console
envserver database drop
```

# CLI

## Adding a host
```console
envcli hosts add --hostid "hostid1" --totalcpu 1200 --totalmem 16785711104
```

## Listing hosts
```console
envcli hosts ls 
```

```console
╭─────────┬─────────┬───────────┬─────────────┬───────────┬───────────┬─────╮
│ HOSTID  │ STATEID │ TOTAL CPU │ TOTAL MEM   │ USAGE CPU │ USAGE MEM │ VMS │
├─────────┼─────────┼───────────┼─────────────┼───────────┼───────────┼─────┤
│ hostid1 │ 1       │ 1200      │ 16785711104 │ 0         │ 0         │ 0   │
│ hostid2 │ 2       │ 1200      │ 16785711104 │ 0         │ 0         │ 0   │
│ hostid3 │ 3       │ 1200      │ 16785711104 │ 0         │ 0         │ 0   │
╰─────────┴─────────┴───────────┴─────────────┴───────────┴───────────┴─────╯
```

## Adding a VM 
```console
envcli vms add --vmid "vmid1" --totalcpu 1200 --totalmem 16785711104
```

## Listing VMs
```console
envcli vms ls 
```

```console
╭───────┬─────────┬──────────┬────────┬──────────────┬───────────┬─────────────┬───────────┬───────────╮
│ VMID  │ STATEID │ DEPLOYED │ HOSTID │ HOST STATEID │ TOTAL CPU │ TOTAL MEM   │ USAGE CPU │ USAGE MEM │
├───────┼─────────┼──────────┼────────┼──────────────┼───────────┼─────────────┼───────────┼───────────┤
│ vmid1 │ 1       │ false    │        │ 0            │ 1200      │ 16785711104 │ 0         │ 0         │
│ vmid2 │ 2       │ false    │        │ 0            │ 1200      │ 16785711104 │ 0         │ 0         │
│ vmid3 │ 3       │ false    │        │ 0            │ 1200      │ 16785711104 │ 0         │ 0         │
╰───────┴─────────┴──────────┴────────┴──────────────┴───────────┴─────────────┴───────────┴───────────╯
```

## Bind a VM to a host
```console
envcli vms bind --vmid vmid1 --hostid hostid3
```

We can now see the *vmid1* VM is deployed on *hostid2*. 
```console
envcli vms ls 
```

```console
╭───────┬─────────┬──────────┬─────────┬──────────────┬───────────┬─────────────┬───────────┬───────────╮
│ VMID  │ STATEID │ DEPLOYED │ HOSTID  │ HOST STATEID │ TOTAL CPU │ TOTAL MEM   │ USAGE CPU │ USAGE MEM │
├───────┼─────────┼──────────┼─────────┼──────────────┼───────────┼─────────────┼───────────┼───────────┤
│ vmid1 │ 1       │ true     │ hostid3 │ 3            │ 1200      │ 16785711104 │ 0         │ 0         │
│ vmid2 │ 2       │ false    │         │ 0            │ 1200      │ 16785711104 │ 0         │ 0         │
│ vmid3 │ 3       │ false    │         │ 0            │ 1200      │ 16785711104 │ 0         │ 0         │
╰───────┴─────────┴──────────┴─────────┴──────────────┴───────────┴─────────────┴───────────┴───────────╯
```

And *hostid3* has one VM.

```console
envcli hosts ls 
```

```console
╭─────────┬─────────┬───────────┬─────────────┬───────────┬───────────┬─────╮
│ HOSTID  │ STATEID │ TOTAL CPU │ TOTAL MEM   │ USAGE CPU │ USAGE MEM │ VMS │
├─────────┼─────────┼───────────┼─────────────┼───────────┼───────────┼─────┤
│ hostid1 │ 1       │ 1200      │ 16785711104 │ 0         │ 0         │ 0   │
│ hostid2 │ 2       │ 1200      │ 16785711104 │ 0         │ 0         │ 0   │
│ hostid3 │ 3       │ 1200      │ 16785711104 │ 0         │ 0         │ 1   │
╰─────────┴─────────┴───────────┴─────────────┴───────────┴───────────┴─────╯
```

## Report host metrics
```console
envcli hosts report --hostid hostid1 --cpu 701 --mem 12345678 
```

```console
envcli hosts ls
```

```console
╭─────────┬─────────┬───────────┬─────────────┬───────────┬───────────┬─────╮
│ HOSTID  │ STATEID │ TOTAL CPU │ TOTAL MEM   │ USAGE CPU │ USAGE MEM │ VMS │
├─────────┼─────────┼───────────┼─────────────┼───────────┼───────────┼─────┤
│ hostid1 │ 1       │ 1200      │ 16785711104 │ 701       │ 12345678  │ 0   │
│ hostid2 │ 2       │ 1200      │ 16785711104 │ 0         │ 0         │ 0   │
│ hostid3 │ 3       │ 1200      │ 16785711104 │ 0         │ 0         │ 1   │
╰─────────┴─────────┴───────────┴─────────────┴───────────┴───────────┴─────╯
```

## Report VM metrics
```console
envcli vms report --vmid vmid1 --cpu 701 --mem 12345678 
```

```console
╭───────┬─────────┬──────────┬─────────┬──────────────┬───────────┬─────────────┬───────────┬───────────╮
│ VMID  │ STATEID │ DEPLOYED │ HOSTID  │ HOST STATEID │ TOTAL CPU │ TOTAL MEM   │ USAGE CPU │ USAGE MEM │
├───────┼─────────┼──────────┼─────────┼──────────────┼───────────┼─────────────┼───────────┼───────────┤
│ vmid1 │ 1       │ true     │ hostid3 │ 3            │ 1200      │ 16785711104 │ 701       │ 12345678  │
│ vmid2 │ 2       │ false    │         │ 0            │ 1200      │ 16785711104 │ 0         │ 0         │
│ vmid3 │ 3       │ false    │         │ 0            │ 1200      │ 16785711104 │ 0         │ 0         │
╰───────┴─────────┴──────────┴─────────┴──────────────┴───────────┴─────────────┴───────────┴───────────╯
```

# REST API

## Hosts 

| Method | Endpoint       | Description                           |
|--------|----------------|---------------------------------------|
| POST   | /hosts         | Add a new host                        |
| GET    | /hosts/:id     | Retrieve a specific host by its ID    |
| GET    | /hosts         | Retrieve all hosts                    |
| DELETE | /hosts/:id     | Remove a specific host by its ID      |

## Virtual Machines (VMs)

| Method | Endpoint         | Description                           |
|--------|------------------|---------------------------------------|
| POST   | /vms             | Add a new virtual machine (VM)        |
| GET    | /vms/:id         | Retrieve a specific VM by its ID      |
| GET    | /vms             | Retrieve all VMs                      |
| PUT    | /vms/:id/:hostid | Bind a VM to a host by their IDs      |
| DELETE | /vms/:id/:hostid | Unbind a VM from a host               |
| DELETE | /vms/:id         | Remove a specific VM by its ID        |

## Metrics

| Method | Endpoint       | Description                           |
|--------|----------------|---------------------------------------|
| POST   | /metrics       | Add new metrics                       |
| GET    | /metrics       | Retrieve all metrics                  |


# Emulator testbed

```console
envcli connectors emulator start
```
