import requests
import base64
import cv2
import time
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
# Define the OpenFaaS function URL
#whisk_url = "https://129.192.71.168:31001/api/v1/namespaces/guest/actions/object-detection?blocking=true&result=true"
whisk_url = "https://130.239.48.225:31001/api/v1/namespaces/guest/actions/object-detection?blocking=true&result=true"

#Authentication
auth_key = "23bc46b1-71f6-4ed5-8c54-816aa4f8c502:123zO3xZCLrMN6v2BKK1dXYFpXlPkccOFqm12CdAsMgRU4VrNZ9lyGVCGuMDGIwP"

header = {
    "Authorization": f"Basic {base64.b64encode(auth_key.encode()).decode()}",
    "Content-Type": "application/json"
}

print("Running....")
img_path = "../frame_sample/3.jpg"
image = cv2.imread(img_path)
# Read an image and encode it to Base64
with open(img_path, "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

object_of_interest = 'red'

# Prepare request payload
payload = {
    "object_of_interest": object_of_interest,
    "frame": encoded_image
}
print('sending request')
# Send POST request
start_time = time.time()
response = requests.post(whisk_url, json=payload, headers = header, verify= False)
end_time = time.time()  # Stop the timer
response_time = end_time - start_time

print(f'get result in {response_time} seconds')
# Print response
print(response.json())

#Draw result to image

response_data = response.json()
center_point = response_data.get("center_point")
bounding_box = response_data.get("bounding_box")


# Define colors for different objects
color_map = {
        "yellow": (0, 255, 255),  # Yellow in BGR format
        "green": (0, 255, 0),     # Green
        "red": (0, 0, 255),       # Red
        "blue": (255, 0, 0),      # Blue
        "orange": (0, 165, 255),  # Orange
        "default": (255, 255, 255)  # White (fallback color)
    }

# If you need x and y separately:
if center_point and bounding_box:
    x, y = center_point
    #print(f"Object detected at: x={x}, y={y}")

    # Get the color based on object label, default to white if not found
    interest_color = color_map.get(object_of_interest.lower(), color_map["default"])
    # Draw an arrow pointing to the object
    # Define colors
    point_color = (255, 0, 0)  # Blue (not red, yellow, or green)
    arrow_color = (255, 0, 0)  # Same as point color
    text_color = (255, 0, 0)  # Same as point color
    arrow_start = (x - 50, y - 50)  # Start above and to the left
    cv2.arrowedLine(image, arrow_start, (x,y), arrow_color, thickness=2, tipLength=0.3)

    # Put text "HERE" near the arrow start
    cv2.putText(image, "HERE", (arrow_start[0] - 10, arrow_start[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
    cv2.circle(image, (x,y), radius=5, color=interest_color, thickness=-1)  # Red dot

    # Draw rectangle
    x_min = bounding_box["x_min"]
    y_min = bounding_box["y_min"]
    x_max = bounding_box["x_max"]
    y_max = bounding_box["y_max"]
    #cv2.rectangle(image, (x_min, y_min), (x_max, y_max), interest_color, thickness=1)
    # Show the image
    cv2.imshow("Detected Object", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No object location found in response.")

