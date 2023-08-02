


# RestAPI for SovereignEdge.COGNIT
# By Paul Townend

version = "0.1"


# To install / use
#
#     pip install "fastapi[all]"
#     pip install "uvicorn[standard]"
#     cd to <folder with RestAPI.py inside it>
#     uvicorn RestAPI:theService --reload



# Two services - one is monitoring so just periodically gets JSON push from ON?   Or grabs from ON
# Second service gets VM placement request and parses and sends on to AI?  Quicker to go straight to AI orch.


from fastapi import Request, FastAPI, BackgroundTasks
theService = FastAPI()


###### Version information, accessible via GET request

@theService.get("/version")
async def about():
    return {"COGNIT Monitoring service - Version " + str(version)}



###### Main service:   Receive periodic updates (host_pool.json) from OpenNebula.  The JSON is transmitted as a string for now.
###### The service returns an "ACK" upon receipt of Json, and then processes the JSON in the background asynchronously 

def processJSON(theJson: str):
    print("Processing JSON")
    # validate the JSON and reject if necessary
    # parse JSON
    # store into memory object (pandas dataframe)
    # periodically store to disk
 

@theService.post("/CognitMonitor")
async def post_body(jsonPayload: Request, background_tasks: BackgroundTasks):
    background_tasks.add_task(processJSON, jsonPayload)
    return {"ACK"}









