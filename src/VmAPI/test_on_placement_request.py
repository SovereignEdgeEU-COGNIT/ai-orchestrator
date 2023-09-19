
import requests
import json

url = "http://localhost:4567"
headers = {
    "Content-Type": "application/json",
}
request =  {
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

response = requests.post(url, headers=headers, data=json.dumps(request))

# Expected output is:
# {
#   "VMS": [
#     {
#       "ID": 7,
#       "HOST_ID": 4
#     }
#   ]
# }

json_data = response.json()
print(json_data)


