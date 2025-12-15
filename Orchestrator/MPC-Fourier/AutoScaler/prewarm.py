import requests
import json
import base64
from concurrent.futures import ThreadPoolExecutor
import subprocess

whisk_url = "https://XXX.XX.XXXX:XXXXX/api/v1/namespaces/{ns}/actions/{func}?blocking=true&result=true"

# Authentication
auth_key = "FILL YOUR AUTHENTICATION"

def send_request(full_url, headers, payload, index, count):
    response = requests.post(
        full_url,
        headers=headers,
        data=json.dumps(payload),
        verify=False
    )
    print(f"[{index+1}/{count}] Prewarm response: {response.status_code} {response.text}")

def prewarm_container(function_name="object-detection", namespace="guest", count=100):
    full_url = whisk_url.format(ns=namespace, func=function_name)
    encoded_auth = base64.b64encode(auth_key.encode()).decode()

    headers = {
        "Authorization": f"Basic {encoded_auth}",
        "Content-Type": "application/json"
    }

    payload = {
        "forcePrewarm": True
    }

    with ThreadPoolExecutor(max_workers=count) as executor:
        for i in range(count):
            executor.submit(send_request, full_url, headers, payload, i, count)

#count: number containers to invoke/cold start/prewarm
#parallelism: how many to send at the same time
def prewarm_with_shell(count=20, function_name = 'object-detection'):
    parallelism = count
    cmd = f"seq {count} | xargs -P{parallelism} -I{{}} wsk -i action invoke {function_name} --param forcePrewarm true "
    try:
        result = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result.stdout.decode())
    except subprocess.CalledProcessError as e:
        print(":x: Command failed:")
        print(e.stderr.decode())


