import requests

def test_post_api_place():
    # The URL to your Flask application
    url = 'http://localhost:50090/api/place'
    headers = {'Content-Type': 'application/json'}
    # Example data payload
    payload = {
        "ID": 1234,
        "HOST_IDS": [7]
    }
    # POST request to /api/place
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 200:
        if response.json().get('HOST_ID') == 7:
            print("POST /api/place response: Success")
            return
    print ("POST /api/place response: Failed")
    print("POST /api/place response:", response.status_code, response.json())

def test_post_api_place_invalid_host():
    # The URL to your Flask application
    url = 'http://localhost:50090/api/place'
    headers = {'Content-Type': 'application/json'}
    # Example data payload
    payload = {
        "ID": 1234,
        "HOST_IDS": [7000]
    }
    # POST request to /api/place
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 200:
        if response.json().get('HOST_ID') == -1:
            print("POST /api/place invalid host response: Success")
            return
    print ("POST /api/place invalid host response: Failed")
    print("POST /api/place invalid host response:", response.status_code, response.json())
    
    

def test_post_api_place_invalid_request():
    # The URL to your Flask application
    url = 'http://localhost:50090/api/place'
    headers = {'Content-Type': 'application/json'}
    # Example data payload
    payload = {
    }
    # POST request to /api/place
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 400:
        print("POST /api/place invalid request response: Success")
        return
    print ("POST /api/place invalid request response: Failed")
    print("POST /api/place invalid request response:", response.status_code, response.json())

def test_post_api_scheduler():
    url = 'http://localhost:50090/api/scheduler'
    headers = {'Content-Type': 'application/json'}
    payload = {"model_name": "InteferenceAwareScheduler"}
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        if response.json().get('status') == 'success':
            print("POST /api/scheduler response: Success")
            return
    print ("POST /api/scheduler response: Failed")
    print("POST /api/scheduler response:", response.status_code, response.json())
    

def test_post_api_scheduler_invalid():
    url = 'http://localhost:50090/api/scheduler'
    headers = {'Content-Type': 'application/json'}
    payload = {"model_name": "NotAScheduler"}
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 400:
        print("POST /api/scheduler invalid response: Success")
        return
    print ("POST /api/scheduler response: Failed")
    print("POST /api/scheduler response:", response.status_code, response.json())

def test_post_api_classifier():
    url = 'http://localhost:50090/api/classifier'
    headers = {'Content-Type': 'application/json'}
    payload = {"model_name": "DLClassifier"}
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        if response.json().get('status') == 'success':
            print("POST /api/classifier response: Success")
            return
    print ("POST /api/classifier response: Failed")
    print("POST /api/classifier response:", response.status_code, response.json())
    
def test_post_api_classifier_invalid():
    url = 'http://localhost:50090/api/classifier'
    headers = {'Content-Type': 'application/json'}
    payload = {"model_name": "NotAClassifier"}
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 400:
        print("POST /api/classifier invalid response: Success")
        return
    print ("POST /api/classifier response: Failed")
    print("POST /api/classifier response:", response.status_code, response.json())
    
def test_post_api_energy():
    url = 'http://localhost:50090/api/iaEnergyScalar'
    headers = {'Content-Type': 'application/json'}
    payload = {"energy_cost_scalar": 1.5}  # Adjust as necessary
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        if response.json().get('status') == 'success':
            print("POST /api/iaEnergyScalar response: Success")
            return
    print("POST /api/iaEnergyScalar response:", response.status_code, response.json())


def main():
    test_post_api_place()
    test_post_api_place_invalid_host()
    test_post_api_place_invalid_request()
    test_post_api_scheduler()
    test_post_api_scheduler_invalid()
    test_post_api_classifier()
    test_post_api_classifier_invalid()
    test_post_api_energy()

if __name__ == "__main__":
    main()
