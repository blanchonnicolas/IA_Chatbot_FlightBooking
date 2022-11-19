from azure.cognitiveservices.language.luis.authoring import LUISAuthoringClient
from azure.cognitiveservices.language.luis.runtime import LUISRuntimeClient  #https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/cognitiveservices/azure-cognitiveservices-language-luis
from msrest.authentication import CognitiveServicesCredentials
import os, json, time, uuid, requests, pytest

#from local_secrets import luisauthoringKey, luisauthoringEndpoint, luispredictionKey, luispredictionEndpoint, luisappId
luisauthoringKey = os.environ.get("LUISAUTHORINGKEY")
luisauthoringEndpoint = os.environ.get("LUISAUTHORINGENDPOINT")
luisappId = os.environ.get("LUIS_APP_ID")

runtimeCredentials = CognitiveServicesCredentials(luisauthoringKey)
clientRuntime = LUISRuntimeClient(endpoint=luisauthoringEndpoint, credentials=runtimeCredentials)


#############################################
##### OUT OF PROJECT CONTEXT UNIT TEST ######
#############################################
def hello(name):
    return 'Hello ' + name
def test_hello():
    assert hello('Nicolas') == 'Hello Nicolas'

#############################################
################## INTENT ###################
#############################################
def test_get_intent_greeting():
    user_text = "Hi"
    bot_response = clientRuntime.prediction.resolve(luisappId, user_text) #Method resolve replaced get_slot_prediction (not present in LUISRUNTIMECLIENT library)
    bot_intent = bot_response.top_scoring_intent.intent
    assert bot_intent == "Greeting"

def test_get_intent_book_flight():
    user_text = "I want to go from Paris to Fortaleza"
    bot_response = clientRuntime.prediction.resolve(luisappId, user_text) #Method resolve replaced get_slot_prediction (not present in LUISRUNTIMECLIENT library)
    bot_intent = bot_response.top_scoring_intent.intent
    assert bot_intent == "Book flight"

def test_score_intent_greeting():
    user_text = "Hi"
    bot_response = clientRuntime.prediction.resolve(luisappId, user_text) #Method resolve replaced get_slot_prediction (not present in LUISRUNTIMECLIENT library)
    bot_intent_score = int(bot_response.top_scoring_intent.score*100)
    assert bot_intent_score >= 80

def test_score_intent_book_flight():
    user_text = "I want to go from Paris to Fortaleza"
    bot_response = clientRuntime.prediction.resolve(luisappId, user_text) #Method resolve replaced get_slot_prediction (not present in LUISRUNTIMECLIENT library)
    bot_intent_score = int(bot_response.top_scoring_intent.score*100)
    assert bot_intent_score >= 80


#############################################
################# ENTITIES ##################
#############################################

def test_get_origin_entity_book_flight():
    user_text = "I want to go from Paris"
    bot_response = clientRuntime.prediction.resolve(luisappId, user_text) #Method resolve replaced get_slot_prediction (not present in LUISRUNTIMECLIENT library)
    bot_origin_entity = bot_response.entities[0].entity
    assert bot_origin_entity == "paris"

def test_get_destination_entity_book_flight():
    user_text = "I want to go to Fortaleza"
    bot_response = clientRuntime.prediction.resolve(luisappId, user_text) #Method resolve replaced get_slot_prediction (not present in LUISRUNTIMECLIENT library)
    bot_destination_entity = bot_response.entities[0].entity
    assert bot_destination_entity == "fortaleza"

def test_get_origin_entity_score_book_flight():
    user_text = "I want to go from Paris"
    bot_response = clientRuntime.prediction.resolve(luisappId, user_text) #Method resolve replaced get_slot_prediction (not present in LUISRUNTIMECLIENT library)
    bot_origin_entity_score = bot_response.entities[0].additional_properties['score']*100
    assert bot_origin_entity_score >= 80

def test_get_destination_entity_score_book_flight():
    user_text = "I want to go to Fortaleza"
    bot_response = clientRuntime.prediction.resolve(luisappId, user_text) #Method resolve replaced get_slot_prediction (not present in LUISRUNTIMECLIENT library)
    bot_destination_entity_score = bot_response.entities[0].additional_properties['score']*100
    assert bot_destination_entity_score >= 80
