# Project "FlyMe Chatbot"

<p align="center">
	<img src="https://github.com/blanchonnicolas/IA_Project10_Openclassrooms_Chatbot/files/FlyMe_image.png" style="width:400px;">
</p>
Context: Develop a Chatbot using LUIS API from Azure
Repository of OpenClassrooms project 10' [AI Engineer path](https://openclassrooms.com/fr/paths/188)

## FlyMe

Our role is to develop a chatbot for flight booking purpose:
 - BOT Framework and Azure Language understanding (LUIS)
 - We expect to achieve a first MVP for FlyMe employees.

The project is using below dataset : [Frames dataset](https://www.microsoft.com/en-us/research/project/frames-dataset/download/).

## How to use

For Setup:
- Python and VSCode with Jupyter extension to read notebooks.
- Environment settings available below (availaible in [requirements.txt](https://github.com/blanchonnicolas/IA_Project10_Openclassrooms_Chatbot/blob/main/requirements.txt))

This repository is part of a 3-repos project :
- Main repo : [Notebook for project exploration](https://github.com/blanchonnicolas/IA_Project10_Openclassrooms_Chatbot) : **this repo**
- ChatBot repo : [CoreBot with application insights](https://github.com/blanchonnicolas/IA_Project10_Openclassrooms_Chatbot/tree/main/FlyMe_BOT_MVP)
- Luis repo : [CoreBot with application insights](https://github.com/blanchonnicolas/IA_Project10_Openclassrooms_Chatbot/tree/main/FlyMe_BOT_MVP)


## More details here :

-   [Presentation](https://github.com/blanchonnicolas/IA_Project10_Openclassrooms_Chatbot)

-   [Vidéo]()

-   [Dataset](https://github.com/blanchonnicolas/IA_Project10_Openclassrooms_Chatbot/tree/main/dataset)

## Main Repo content

-   [Notebook : Exploratory Data Analysis](https://github.com/blanchonnicolas/IA_Project10_Openclassrooms_Chatbot/blob/main/EDA.ipynb)
    - Data Analysis: Visualisation, Modification 
	- Generation of LUIS compatible JSON files
    - Intent and Entity settings
    - Output Json Files
    - Luis performance results analysis

## ChatBot Repo content

-   [CoreBot with application insights](https://github.com/blanchonnicolas/IA_Project10_Openclassrooms_Chatbot/tree/main/FlyMe_BOT_MVP)
    - [Bot Framework v4 ](https://dev.botframework.com/)
    - [Deployment workflow](https://github.com/blanchonnicolas/IA_Project10_Openclassrooms_Chatbot/blob/main/.github/workflows/main_flyme-bot-webapp.yml)
    - [Application Monitoring Insights](https://learn.microsoft.com/fr-fr/azure/azure-monitor/app/azure-web-apps-net-core?tabs=Windows%2Clinux)

## Luis Repo content

-   [Language Understanding Intelligent Service](https://github.com/blanchonnicolas/IA_Project10_Openclassrooms_Chatbot/tree/main/luis)
    - [Luis AI ](https://www.luis.ai/)
	- [Luis Doc](https://learn.microsoft.com/en-us/azure/cognitive-services/luis/what-is-luis)
    - [Luis Unit Testing](https://github.com/blanchonnicolas/IA_Project10_Openclassrooms_Chatbot/tree/main/unit_test)
    - [Luis results](https://github.com/blanchonnicolas/IA_Project10_Openclassrooms_Chatbot/tree/main/luis/results)


## Specifities that you must pay attention to
- Library for BOT modified compared to official SDK pip (4.15 requested instead of 4.14)
- requirements version for emoji==1.7 and aiohttp==3.6.2 and pytest==6.2.3
- requirements library asyncio removed due to compatibility issues
- publish-profile is unique to any Webapp resource, it should never be communicated outside of the project workflow from GitHub to Azure
- GitHub secrets and Azure environment variables are key sensitive (Pay attention to scripts calling them)
- PORT in config.py set to 8000, to reach URL as deployed
- init_func in app.py should match with Azure starting command : *python -m aiohttp.web -H 0.0.0.0 -P 8000 app:init_func*
- scripts using *resolve method* to reach endpoint (instead of *get_slot_production* recommended in azure doc)


