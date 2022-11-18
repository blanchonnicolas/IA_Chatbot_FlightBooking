from pprint import pprint

from azure.cognitiveservices.language.luis.authoring import LUISAuthoringClient
from azure.cognitiveservices.language.luis.authoring.models import ApplicationCreateObject
from azure.cognitiveservices.language.luis.runtime import LUISRuntimeClient  #https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/cognitiveservices/azure-cognitiveservices-language-luis
from msrest.authentication import CognitiveServicesCredentials
from functools import reduce

import os, pathlib, json, time, uuid, requests, datetime, pickle
#import http.client, urllib.request, urllib.parse, urllib.error, base64

def get_local_file_secrets():
    try:
        from local_secrets import luisauthoringKey, luisauthoringEndpoint, luispredictionKey, luispredictionEndpoint, luisappId
        return luisauthoringKey, luisauthoringEndpoint, luispredictionKey, luispredictionEndpoint, luisappId
    except FileNotFoundError:
        print("local_secrets.py file not found")
        exit(0)


class Luis_cloning_Authoring:
    """Cloning_Authoring.
    This will Clone an existing LUIS Flight Booking application version, as requested by project FlyMe.
    """
    def __init__(self):
        self.client = None
        self.appName = None
        self.versionId = None
        self.app_id = None

    def authentify(self, authoringKey, authoringEndpoint):
        #Authentify the client 
        self.authoringKey = authoringKey
        self.authoringEndpoint = authoringEndpoint
        self.client = LUISAuthoringClient(self.authoringEndpoint, CognitiveServicesCredentials(self.authoringKey))
        self.headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Ocp-Apim-Subscription-Key": self.authoringKey,
        }
    
    def get_app_version(self, luisappId):
        self.app_id = luisappId
        print("\nGet App versionId through API request")
        self.base_author_url = f"{self.authoringEndpoint}/luis/api/v2.0/apps/{self.app_id}/versions"
        self.response_app_versions = requests.get(f"{self.base_author_url}", headers=self.headers)
        if self.response_app_versions:
            self.versionId = self.response_app_versions.json()[0]["version"]
            print("Current LUIS app versionId {}".format(self.versionId))
        else:
            print("Error: Status Code", self.response_app_versions.status_code)

    def clone_app_version(self, luisappId):
        self.app_id = luisappId
        print("\nClone App version through API request")
        version_increment = round((float(self.versionId) + 0.1), 1)
        self.new_versionId = str(version_increment) 
        body_json = {
            "version": self.new_versionId
            }
        print("New LUIS app versionId {}".format(self.new_versionId))
        self.response_app_clone = requests.post(f"{self.base_author_url}/{self.versionId}/clone", headers=self.headers, json=body_json)
        if self.response_app_clone:
            self.versionId = self.response_app_clone.text.replace('"', '')
            print("Cloned LUIS app versionId {}".format(self.versionId))
        else:
            print("Error: Status Code", self.response_app_clone.status_code)


    def create_FlyMe_intent(self): #Loop necessarily on 3 intents
        # create intents - 3 intents expected in the frame of FlyMe application
        for intent in ["Book flight", "Greeting"]:
            self.client.model.add_intent(self.app_id, self.versionId, intent)
            print("Created LUIS intent {}".format(intent))
    
    def create_FlyMe_ml_entity(self, entity_name): #Loop necessarily on 5 entities
        # create entities - 5 ML entities expected
        # Add entity to app
        self.modelId_Entity = self.client.model.add_entity(self.app_id, self.versionId, name=entity_name)
        print("Created LUIS entity {}".format(entity_name))
        # Get entity and subentities - 5 ML entities
        self.modelObject = self.client.model.get_entity(self.app_id, self.versionId, self.modelId_Entity)
        print("LUIS Get entity {}".format(self.modelId_Entity))
        

    def create_FlyMe_prebuilt_entity(self):
        # Add Prebuilt entity
        self.modelId_Entity = self.client.model.add_prebuilt(self.app_id, self.versionId, prebuilt_extractor_names=["datetimeV2"])
        print("Created LUIS prebuild entity datetimeV2")
        return self.modelId_Entity

    def load_json(self, json_folder_path = None, json_file_name = None):
        #Load testing data from json
        if json_folder_path is None:
            src_path = os.path.abspath(os.path.join("../"))
            json_path = os.path.join(src_path, "dataset", json_file_name)
        else:
            json_path = os.path.join(json_folder_path, json_file_name)
        
        try:
            print(f"Json loading process from file : {json_path}")
            with open(json_path) as json_file:
                print("Json loading process completed")
                return json.load(json_file)
        except FileNotFoundError as e:
            print(e)
            exit(0)

    def batch_training_with_labeled_utterances(self): #Loop on batches
        #Create batch of 100 utterances, to train our LUIS model (Maximum batch size)
        #https://westus.dev.cognitive.microsoft.com/docs/services/luis-programmatic-apis-v3-0-preview/operations/5890b47c39e2bb052c5b9c09
        training_json_file = self.load_json(None, "train.json")
        print("\nWe'll start feeding you model with labeled utterance")
        for i in range(0, len(training_json_file), 100): # Lenght total of 1232 for training data len(training_json_file)
            j = (i + 100)
            self.client.examples.batch(self.app_id, self.versionId, training_json_file[i+1:j])
            print(f"Feeding process running on batch range from {i+1} to {j}")
    
    def training_status(self):
        #Sends a training API POST request for a version of a specified LUIS app.
        self.client.train.train_version(self.app_id, self.versionId)
        waiting = True
        while waiting:
            #Submit a GET request to get training status.
            info = self.client.train.get_status(self.app_id, self.versionId)
            # get_status returns a list of training statuses, one for each model. Loop through them and make sure all are done.
            waiting = any(map(lambda x: 'Queued' == x.details.status or 'InProgress' == x.details.status, info))
            if waiting:
                print ("Waiting for training to complete...")
                time.sleep(10)
            else: 
                print ("Training phase Completed")
                waiting = False

    def publish_app(self):
        # Mark the app as public so we can query it using any prediction endpoint.
        # Note: For production scenarios, you should instead assign the app to your own LUIS prediction endpoint. See:
        # https://docs.microsoft.com/en-gb/azure/cognitive-services/luis/luis-how-to-azure-subscription#assign-a-resource-to-an-app
        self.client.apps.update_settings(self.app_id, is_public=True)
        responseEndpointInfo = self.client.apps.publish(self.app_id, self.versionId, is_staging=False)
        print(f"Application published : {responseEndpointInfo}")

    def batch_testing_with_labeld_utterances(self, valid_json_file):
        self.environment = "Production"
        self.clientRuntime = LUISRuntimeClient(predictionEndpoint, CognitiveServicesCredentials(predictionKey))
        self.base_test_url = f"{predictionEndpoint}/luis/v3.0-preview/apps/{self.app_id}/slots/{self.environment}/evaluations" #If looking in Production
        #self.base_test_url = f"{predictionEndpoint}/luis/v3.0-preview/apps/{self.app_id}/versions/{self.versionId}/evaluations" #If looking by VersionId
        self.headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Ocp-Apim-Subscription-Key": predictionKey,
        }
        validation_json = {
            "LabeledTestSetUtterances": self.load_json(None, "valid.json")
        }
        print(f"\nTesting phase starting using LUIS Batch API requests")
        self.response_batch_test = requests.post(self.base_test_url, headers=self.headers, json=validation_json)
        
        if self.response_batch_test:
            operationId = self.response_batch_test.json()["operationId"]
            print(f"operationId is = {operationId}")
            return operationId
        else:
            print("Error: Status Code", self.response_batch_test.status_code)
            return False
    
    def batch_testing_status(self, operation_id):
        #Sends a testing API GET request for a operation_id of a batch testing request.
        #   <YOUR-PREDICTION-ENDPOINT>/luis/v3.0-preview/apps/<YOUR-APP-ID>/slots/<YOUR-SLOT-NAME>/evaluations/<YOUR-OPERATION-ID>/status   #Get status of batch testing
        print ("\nTesting phase status check")
        self.operation_id = operation_id
        if self.operation_id:
            self_status_url = f"{self.base_test_url}/{self.operation_id}/status"
            waiting = True
            while waiting:
                #Submit a GET request to get training status.
                self.response_batch_status =  requests.get(self_status_url, headers=self.headers)
                if self.response_batch_status.json()["status"] == "succeeded":
                    print ("Testing phase Completed")
                    waiting = False
                else:
                    print ("Waiting for testing to complete...")
                    time.sleep(10)
        else:
            print ("Testing status error with operation_id")
    
    def batch_testing_results(self):
        print("\nResults analysis phase starting")
        self.response_batch_results = requests.get(f"{self.base_test_url}/{self.operation_id}/result", headers=self.headers)

        if self.response_batch_results:
            print("--> JSON:")
            src_path = os.path.abspath(os.getcwd())
            luis_results_json_path = os.path.join(src_path, "results", "luis_batch_testing_results.json")
            with open(luis_results_json_path, 'w') as file:
                json.dump(self.response_batch_results.json(), file, indent=1)
            print("--> Intent Overall results:")
            for i in range(len(self.response_batch_results.json()["intentModelsStats"])):
                print(self.response_batch_results.json()["intentModelsStats"][i])
            print("--> Entities Overall results:")
            for i in range(len(self.response_batch_results.json()["entityModelsStats"])):
                print(self.response_batch_results.json()["entityModelsStats"][i])
        else:
            print("Error: Status Code", self.response_batch_results.status_code)
       

if __name__ == "__main__":

    luis_cloning_authoring = Luis_cloning_Authoring()

    authoringKey, authoringEndpoint, predictionKey, predictionEndpoint, luisappId = get_local_file_secrets()

    #--------------- AUTHORING ---------------
    luis_cloning_authoring.authentify(authoringKey, authoringEndpoint)
    luis_cloning_authoring.get_app_version(luisappId)
    luis_cloning_authoring.clone_app_version(luisappId)
    # luis_cloning_authoring.create_FlyMe_app()
    # luis_cloning_authoring.create_FlyMe_intent()
    # luis_cloning_authoring.create_FlyMe_prebuilt_entity()
    # for entity in ["From", "To", "budget", "str_date", "end_date"]:
    #     modelId_Entity = luis_cloning_authoring.create_FlyMe_ml_entity(entity)  
    
    # #--------------- TRAINING ---------------
    # luis_cloning_authoring.batch_training_with_labeled_utterances()
    # luis_cloning_authoring.training_status()
    luis_cloning_authoring.publish_app()

    # #--------------- TESTING ---------------
    valid_json_file = luis_cloning_authoring.load_json(None, "valid.json")
    operation_id = luis_cloning_authoring.batch_testing_with_labeld_utterances(valid_json_file)
    luis_cloning_authoring.batch_testing_status(operation_id)
    luis_cloning_authoring.batch_testing_results()






