# Interoperability
The OpenNebula scheduler will send this request to the AI Orchestrator when a VM is going to be deployed. It is expecting 
a server running at port 8000

```json
{
  "VMS": [
    {
      "CAPACITY": {
        "CPU": 1.0,
        "DISK_SIZE": 2252,
        "MEMORY": 786432
      },
      "HOST_IDS": [
        0,
        2,
        3,
        4
      ],
      "ID": 7,
      "STATE": "PENDING",
      "USER_TEMPLATE": {
        "LOGO": "images/logos/ubuntu.png",
        "LXD_SECURITY_PRIVILEGED": "true",
        "SCHED_REQUIREMENTS": "ID=\"0\" | ID=\"2\" | ID=\"3\" | ID=\"4\""
      }
    }
  ]
}
```

The server is expecting this JSON response:

```json
{
  "VMS": [
    {
      "ID": 7,
      "HOST_ID": 4
    }
  ]
}
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
ssh -v -N -L 0.0.0.0:9090:10.10.10.2:9090 root@194.28.122.112
```

Prometheus is available at http://localhost:9090.

## Accessing OpenNebula Connector
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
