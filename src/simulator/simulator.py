import time
from universe import *
from location import *
from edgecluster import *
from server import *
from clock import *
from process import *
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from prometheus_client import generate_latest, REGISTRY, Gauge, CONTENT_TYPE_LATEST
import uvicorn
import threading

def sleep_ms(milliseconds):
    event = threading.Event()
    threading.Timer(milliseconds / 1000, event.set).start()
    event.wait()

def start_sim():
    clock = Clock()
    univ = Universe()
    
    loc = Location("umea")
    univ.add_location(loc)
    edgecluster = EdgeCluster("umu_cs")
    loc.add_edgecluster(edgecluster)
    for i in range(5):
        server = Server("edge_server_node_"+str(i), clock, cores=1)
        edgecluster.add_server(server)
    
    loc = Location("lulea")
    univ.add_location(loc)
    edgecluster = EdgeCluster("rise")
    loc.add_edgecluster(edgecluster)
    for i in range(5):
        server = Server("edge_server_node_"+str(i), clock, cores=1)
        edgecluster.add_server(server)
        
    while True:
        for l in univ.locations:
            for e in l.edgeclusters:
                cpuload_gauge = e.cpuload_gauge
                processes_gauge = e.processes_gauge
                
                e.energymix.change_state()
                state = e.energymix.state
                mix = e.energymix.calc()
                e.energymix_gauge.labels(energytype="green").set(mix[0])
                e.energymix_gauge.labels(energytype="brown").set(mix[1])
                e.energymix_state_gauge.set(state)
    
                for server in edgecluster.servers:
                    a, b = 4, 7 
                    s = beta.rvs(a, b, size=1)[0]
                    if s > 0.80:
                        process = Process(exectime=random.randint(1, 1000)*1e7, category=[])
                        server.launch(process)
                        print("starting process <{}>".format(process.pid))
      
                    server.execute()

                    cpuload_gauge.labels(server=server.name).set(server.cpuload)
                    processes_gauge.labels(server=server.name).set(server.processes)
       
                clock.tick_ms()
                sleep_ms(1)

app = FastAPI()

@app.get("/metrics", tags=["Monitoring"], response_class=PlainTextResponse)
def metrics():
    """Metrics endpoint for Prometheus."""
    return PlainTextResponse(generate_latest(REGISTRY).decode("utf-8"), headers={"Content-Type": CONTENT_TYPE_LATEST})

def main():
    t = threading.Thread(target=start_sim)
    t.start()
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
