## Run on local machine  
python3 main.py

## Run on docker container  
1. Build the docker container  
   sudo docker build -t python/ai-orchestrator .  
2. Start the docker container  
   sudo docker run -p 4567:4567 --rm -it python/ai-orchestrator  
3. Use the system recorder to send the request to simulator    

## Notice(Before building the docker container)
In VM_info.py, line 47  
url = "http://192.168.113.128:8000/hosts/{}".format(int(host_id))  
Use the ip address in your own machine to instead The IP address in above code.
