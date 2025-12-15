import requests
import base64
import cv2
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

print('Running...')

# Image path
img_path = "../frame_sample/3.jpg"
image = cv2.imread(img_path)

# Read and encode image to Base64
with open(img_path, "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

# Kubernetes Deployment URL (Replace <node-ip> and <nodeport>)
deployment_url = "http://129.192.71.168:32710/"  # Use the NodePort for the normal deployment

# Define payload
payload = {
    "object_of_interest": "red",
    "frame": encoded_image
}

# Number of requests to send
num_requests = 50  # Change this value to control the load

# Function to send a request and measure response time
def send_request(req_id):
    start_time = time.time()
    response = requests.post(deployment_url, json=payload, timeout=100)  # Timeout 100 seconds
    end_time = time.time()
    response_time = end_time - start_time
    return req_id, response.json(), response_time

# Use ThreadPoolExecutor for parallel requests
with ThreadPoolExecutor(max_workers=1) as executor:  # Adjust max_workers based on system resources
    futures = {executor.submit(send_request, i): i for i in range(num_requests)}

    for future in as_completed(futures):
        req_id, response_data, response_time = future.result()
        print(f"Request {req_id} completed in {response_time:.4f} seconds")
        print(f"Response: {response_data}")

print("All requests completed.")
