#from ModelManager import ModelManager, ModelTypes
from typing import List
from ClassifierInterface import ClassifierInterface
from ClassicalClassifier.ClassicalClassifier import ClassicalClassifier
from DLClassifier.DLClassifier import DLClassifier, DLClassifierType
from RandomScheduler import RandomScheduler
from SchedulerInterface import SchedulerInterface
from OnedConnector import OnedConnector
from RandomClassifier import RandomClassifier
from IAScheduler import InteferenceAwareScheduler
from DBConnector import DBClient
from flask import Flask, request, jsonify
from time import sleep
import sys
import os

from typing_extensions import TypedDict

class MyAppExtensions(TypedDict):
    #random_classifier: RandomClassifier
    # You can add other custom attributes here
        
    scheduler: SchedulerInterface
    dbClient: DBClient
    onedClient: OnedConnector
    ia_scheduler: InteferenceAwareScheduler
    random_scheduler: RandomScheduler
    classifier: ClassifierInterface
    dl_ir: DLClassifier
    dl_classifier: DLClassifier
    classical_classifier: ClassicalClassifier
    random_classifier: RandomClassifier


class CustomFlask(Flask):
    def __init__(self, extensions: MyAppExtensions, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extensions: MyAppExtensions = extensions
#modelManager: ModelManager


def create_app(__name__) -> CustomFlask:
    dbClient = DBClient(1, 10)
    onedClient = OnedConnector()
    
    random_classifier = RandomClassifier()
    random_classifier.initialize()
    print("Random Classifier initialized")
    
    dl_ir = DLClassifier(dbClient, onedClient, DLClassifierType.IR)
    dl_ir.initialize()
    print("DL IR initialized")
    
    dl_classifier = DLClassifier(dbClient, onedClient, DLClassifierType.CLASSIFIER)
    dl_classifier.initialize()
    print("DL Classifier initialized")
    
    classical_classifier = ClassicalClassifier(dbClient, onedClient)
    classical_classifier.initialize()
    print("Classical Classifier initialized")
    
    classifier = dl_ir
    
    ia_scheduler = InteferenceAwareScheduler(dbClient, onedClient, classifier)
    ia_scheduler.initialize()
    print("IA Scheduler initialized")
    scheduler = ia_scheduler
    
    random_scheduler = RandomScheduler()
    random_scheduler.initialize()
    print("Random Scheduler initialized")
    
    # Populate extensions after initialization
    extensions = MyAppExtensions(
        scheduler=scheduler,
        dbClient=dbClient,
        onedClient=onedClient,
        ia_scheduler=ia_scheduler,
        random_scheduler=random_scheduler,
        classifier=classifier,
        dl_ir=dl_ir,
        dl_classifier=dl_classifier,
        classical_classifier=classical_classifier,
        random_classifier=random_classifier,
    )

    return CustomFlask(extensions, __name__)

#? Lazy solution for waiting for the DB to have hosts loaded before initiating the schedulers
#sleep(10)

app = create_app(__name__)

def simulate_placement_req():
    arr1 = [0, 2, 3, 4, 7, 8]
    arr2 = [7, 8]
    
    
    for i in [1536, 1526, 1527, 1528, 1529, 1530, 1531, 1532]: 
        if i % 2 == 0:
            print("Optimal host:", app.extensions['scheduler'].predict(i, arr2))
        else:
            print("Optimal host:", app.extensions['scheduler'].predict(i, arr1))


# Define a route to receive VM data
@app.route('/api/place', methods=['POST'])
def receive_vm_data():
    vm_req = request.json  # Parse JSON data from request
    # Validate or use the data as needed
    #print("Received VM data:", vm_req, file=sys.stderr)

    # Example: you might want to pass this data to some ML function
    # result = some_ml_function(data)
    
    if vm_req is None or 'ID' not in vm_req is None or 'HOST_IDS' not in vm_req:
        return jsonify({"error": "Invalid request, missing 'ID' or 'HOST_IDS'."}), 400
    
    vm_id: int = vm_req["ID"]
    potential_hosts: List[int] = vm_req["HOST_IDS"]
    
    print("Potential hosts for VM", vm_id, ":",potential_hosts, file=sys.stdout)
    
    host_id = app.extensions['scheduler'].predict(vm_id, potential_hosts)
    
    print("Optimal host:", host_id, file=sys.stdout)

    # For now, we just send back a confirmation response
    return jsonify({"ID": vm_id, "HOST_ID": host_id}), 200

"""
@app.route('/api/model', methods=['POST'])
def set_model():
    data = request.json
    model_type = data.get('model_type')
    model_name = data.get('model_name')
    
    try:
        modelManager.set_model(model_type, model_name)
        return jsonify({"status": "success", "message": "Model set"}), 200
    except ValueError as e:
        return jsonify({"status": "error", "message": str(e)}), 400
"""

@app.route('/api/scheduler', methods=['POST'])
def set_scheduler():
    data = request.json
    #model_type = data.get('model_type')
    
    if data is None or 'model_name' not in data: 
        return jsonify({"status": "error", "message": "missing data"}), 400
    
    model_name = data.get('model_name')
    
    if model_name == InteferenceAwareScheduler.get_name():
        app.extensions['scheduler'] = app.extensions['ia_scheduler']
        #return jsonify({"status": "success", "message": "Model set"}), 200
    elif model_name == RandomScheduler.get_name():
        app.extensions['scheduler'] = app.extensions['random_scheduler']
    else:
        return jsonify({"status": "error", "message": "no such scheduler model"}), 400
    
    #simulate_placement_req()
    return jsonify({"status": "success", "message": "Model set"}), 200


@app.route('/api/classifier', methods=['POST'])
def set_classifier():
    data = request.json
    #model_type = data.get('model_type')
    
    if data is None or 'model_name' not in data: 
        return jsonify({"status": "error", "message": "missing data"}), 400
    
    model_name = data.get('model_name')
    
    if model_name == DLClassifier.get_name(DLClassifierType.CLASSIFIER):
        app.extensions['classifier'] = app.extensions['dl_classifier']
    elif model_name == DLClassifier.get_name(DLClassifierType.IR):
        app.extensions['classifier'] = app.extensions['dl_ir']
    elif model_name == ClassicalClassifier.get_name():
        app.extensions['classifier'] = app.extensions['classical_classifier']
    elif model_name == RandomClassifier.get_name():
        app.extensions['classifier'] = app.extensions['random_classifier']
    else:
        return jsonify({"status": "error", "message": "no such scheduler model"}), 400

    app.extensions['scheduler'].set_classifier(app.extensions['classifier'])
    
    #simulate_placement_req()
    return jsonify({"status": "success", "message": "Model set"}), 200
    

@app.route('/api/iaEnergyScalar', methods=['POST'])
def set_green_energy_scalar():
    data = request.json
    #model_type = data.get('model_type')
    
    if data is None or 'energy_cost_scalar' not in data: 
        return jsonify({"status": "error", "message": "missing data"}), 400
    
    energy_scalar = data.get('energy_cost_scalar')
    app.extensions['ia_scheduler'].set_green_energy_scalar(energy_scalar)
    return jsonify({"status": "success", "message": "Green Energy Scalar set"}), 200


if __name__ == '__main__':
    port = int(os.getenv('ML_MODEL_PORT', 50090))
    #modelManager = ModelManager()
    
    #init_models(app)
    
    #modelManager.add_model(ModelTypes.SCHEDULER, ia_scheduler)
    #modelManager.add_model(ModelTypes.CLASSIFIER, randomClassifier)
    #modelManager.set_model(ModelTypes.SCHEDULER, ia_scheduler.get_name())
    
    #simulate_placement_req()

    app.run(debug=True, port=port, host='0.0.0.0')