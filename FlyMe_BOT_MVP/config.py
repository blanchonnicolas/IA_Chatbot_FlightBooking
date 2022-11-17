#!/usr/bin/env python
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Configuration for the bot."""

import os


class DefaultConfig:
    """Configuration for the bot."""

    PORT = 8000
    APP_ID = os.environ.get("MicrosoftAppId", "") #da1f543d-2d9e-4866-8817-a89b7e304618
    APP_PASSWORD = os.environ.get("MicrosoftAppPassword", "") #PCS8Q~7UWo1iJqm-0Cir.xa2NNajW.-i6ATmtcdc
    LUIS_APP_ID = os.environ.get("LUIS_APP_ID", "") #7113906d-43c9-473b-b314-34e95b96c6c0 #d94580fa-613f-4633-b8fd-4e786bb61163
    LUIS_API_KEY = os.environ.get("LUIS_API_KEY", "") #7705533016a54152b086d5dccf81ee6a #"7705533016a54152b086d5dccf81ee6a")
    # LUIS endpoint host name, ie "westus.api.cognitive.microsoft.com"
    LUIS_API_HOST_NAME = os.environ.get("LUIS_API_HOST_NAME", "") #chatbot-iap10openclassrooms.cognitiveservices.azure.com
    APPINSIGHTS_INSTRUMENTATION_KEY = os.environ.get("APPINSIGHTS_INSTRUMENTATION_KEY", "") #85c2e05f-72ea-48b7-a6a7-5e465b6df4f1 #7080142b-dd63-4b1c-a752-341088f263b8
