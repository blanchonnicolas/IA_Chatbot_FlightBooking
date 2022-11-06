#!/usr/bin/env python
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Configuration for the bot."""

import os


class DefaultConfig:
    """Configuration for the bot."""

    PORT = 3978
    APP_ID = os.environ.get("MicrosoftAppId", "")
    APP_PASSWORD = os.environ.get("MicrosoftAppPassword", "")
    LUIS_APP_ID = os.environ.get("LuisAppId", "7113906d-43c9-473b-b314-34e95b96c6c0") #d94580fa-613f-4633-b8fd-4e786bb61163
    LUIS_API_KEY = os.environ.get("LuisAPIKey", "7705533016a54152b086d5dccf81ee6a") #"7705533016a54152b086d5dccf81ee6a")
    # LUIS endpoint host name, ie "westus.api.cognitive.microsoft.com"
    LUIS_API_HOST_NAME = os.environ.get("LuisAPIHostName", "chatbot-iap10openclassrooms.cognitiveservices.azure.com")
    APPINSIGHTS_INSTRUMENTATION_KEY = os.environ.get(
        "AppInsightsInstrumentationKey", "85c2e05f-72ea-48b7-a6a7-5e465b6df4f1"
    )
