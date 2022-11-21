# Cognitive Services: LUIS Runtime

This repository completes the FlyMe_BOT_MVP with:
- LUIS Authoring script to build, train, publish and test your LUIS model
- LUIS Cloning Authoring script to clone existing LUIS model, increment version, train, publish and test your LUIS model
- LUIS Testing script to test existing LUIS model

## Prerequisites

The minimum prerequisites to run this sample are:
* A [LUIS.ai account](https://www.luis.ai/) where to upload the sample's LUIS model.

The first step is to get your Authoring Key. Go to the home page, [www.luis.ai](https://www.luis.ai/), and log in. After creating your LUIS account, a starter key, also known as a authoring key, is created automatically for LUIS account. To find the authoring key, click on the account name in the upper-right navigation bar to open [Account Settings](https://www.luis.ai/user/settings), which displays the Authoring Key.

![Get the programmatic key](images/programmatic-key.png)

Set the `LUIS_SUBSCRIPTION_KEY` environment variable to this authoring key to continue.

Detail your information secrets, that will be retreived by scripts.

#### Where to find the Application ID and Subscription Key

You'll need these two values to configure the LuisDialog through the LuisModel attribute:

1. Application ID

    You can find the App ID in the LUIS application's settings.

    ![App Settings](images/prereqs-appid.png)

2. Subscription Key and Endpoint

    Click on the Publish App link from the top of the LUIS application dashboard. Once your app is published, copy the Endpoint and Key String from *Starter_Key* from the Endpoints table on the Publish App page.

    ![Programmatic API Key](images/prereqs-apikey.png)


### LUIS Application Overview

Create or import an application: home page [www.luis.ai](https://www.luis.ai/)

![Import an Existing Application](images/prereqs-import.png)

Train your model using dataset [train.json](https://github.com/blanchonnicolas/IA_Project10_Openclassrooms_Chatbot/blob/main/dataset/train.json) file.
- [Training a Model](https://docs.microsoft.com/en-us/azure/cognitive-services/luis/train-test) 
- [Publishing a Model](https://docs.microsoft.com/en-us/azure/cognitive-services/luis/publishapp).


## To try this sample on your own
- Follow the resource creation steps for LUIS (as detailed in [Azure SDK sample 21])
- Clone the repository
- Activate your desired virtual environment
- In the terminal, type `pip install -r requirements.txt`
- In a terminal, navigate to `\luis` folder
- Set your own azure and github secrets
- Run your bot with `python luis_authoring.py`

### Code Highlights

Scripts are detailed through python Class, and you must comment/uncomment the desired part at the end of each script.

We invoke the LUIS Runtime API to analyze user input and obtain intent and entities.

````python
from azure.cognitiveservices.language.luis.runtime import LUISRuntimeClient
from msrest.authentication import CognitiveServicesCredentials

// Create client with SubscriptionKey and Endpoint
client = LUISRuntimeClient(
    'https://westus.api.cognitive.microsoft.com',             # Change "westus" to your region if necessary
    CognitiveServicesCredentials("[LUIS_SUBSCRIPTION_KEY]"),  # Put your LUIS Subscription key
)

// Predict
luis_result = client.prediction.resolve(
    "[LUIS_APPLICATION_ID]",                                  # Put your LUIS Application ID
    "Text to Predict or User input"
)
````

The LuisResult object contains the possible detected intents and entities that could be extracted from the input.
Library [azure-cognitiveservices-language-luis](http://pypi.python.org/pypi/azure-cognitiveservices-language-luis)

### Outcome

You will see the following when running the application:

![Sample Outcome](images/outcome.png)

### More Information

To get more information about how to get started in Bot Builder for .NET and Conversations please review the following resources:
* [Language Understanding Intelligent Service](https://azure.microsoft.com/en-us/services/cognitive-services/language-understanding-intelligent-service/)
* [LUIS.ai](https://www.luis.ai)
* [LUIS Docs](https://docs.microsoft.com/en-us/azure/cognitive-services/luis/home)
* [LUIS Runtime API v2 - Specification](https://github.com/Azure/azure-rest-api-specs/tree/master/specification/cognitiveservices/data-plane/LUIS/Runtime)
