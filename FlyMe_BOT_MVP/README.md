# CoreBot with Application Insights

We started our project CoreBot from [Azure SDK sample 21](https://github.com/microsoft/BotBuilder-Samples/tree/main/samples/python/21.corebot-app-insights)

Bot Framework v4 core bot sample.

This bot has been created using [Bot Framework](https://dev.botframework.com), it shows how to:

- Use [LUIS](https://www.luis.ai) to implement core AI capabilities
- Implement a multi-turn conversation using Dialogs
- Handle user interruptions for such things as `Help` or `Cancel`
- Prompt for and validate requests for information from the user
- Use [Application Insights](https://docs.microsoft.com/azure/azure-monitor/app/cloudservices) to monitor your bot

## Prerequisites

This sample **requires** prerequisites in order to run.

### Overview

This bot uses [LUIS](https://www.luis.ai), an AI based cognitive service, to implement language understanding
and [Application Insights](https://docs.microsoft.com/azure/azure-monitor/app/cloudservices), an extensible Application Performance Management (APM) service for web developers on multiple platforms.

This BOT relies on **LUIS language model** cognitive services as developped in sub-repo [luis](https://github.com/blanchonnicolas/IA_Project10_Openclassrooms_Chatbot/tree/main/FlyMe_BOT_MVP)


**Application Insights resource** enable to log activity and user/chabot information through `app.py` telemetry client (TELEMETRY_LOGGER_MIDDLEWARE).
Instrumentation key and luis endpoint are detailed in Github secrets and Azure environment configuration. They are retreived throu the `config.py` file.

![Azure Insights event](https://github.com/blanchonnicolas/IA_Project10_Openclassrooms_Chatbot/blob/47a80bc2b3f77b65178766f70b5ff079afae588b/files/Insights-events.png)

### To try this sample on your own
- Follow the resource creation steps for LUIS (as detailed in [Azure SDK sample 21])
- Clone the repository
- In a terminal, navigate to `\FlyMe_BOT_MVP` folder
- Activate your desired virtual environment
- In the terminal, type `pip install -r requirements.txt`
- Set your own azure and github secrets
- Run your bot with `python app.py`

> PLEASE, PAY ATTENTION TO DIFFERENCE USING LIBRAIRIES 4.15 (official sdk pip download still on 4.14)

## Testing the bot using Bot Framework Emulator

### Interactive Testing
[Bot Framework Emulator](https://github.com/microsoft/botframework-emulator) is a desktop application that allows bot developers to test and debug their bots on localhost or running remotely through a tunnel.

- Install the latest Bot Framework Emulator from [here](https://github.com/Microsoft/BotFramework-Emulator/releases)
- Launch Bot Framework Emulator
- File -> Open Bot
- Enter a Bot URL of `http://localhost:3978/api/messages` for local test
or
- Enter a Cloud based URL for remote test

![BOT Emulator](https://github.com/blanchonnicolas/IA_Project10_Openclassrooms_Chatbot/blob/21873f199b7bafb68ea677ec41317b5aad0e959b/files/BOT_Emulator.png)
> PLEASE, PAY ATTENTION Ngrok version 2.3.40

### Unit Testing

We've create [pytest](https://docs.pytest.org/) scenarii to run during deployment phase.
The pytest script is available in repository sub-folder [unit_test](https://github.com/blanchonnicolas/IA_Project10_Openclassrooms_Chatbot/tree/main/unit_test)

![Git Hub Continuous Integration - Unit Test](https://github.com/blanchonnicolas/IA_Project10_Openclassrooms_Chatbot/blob/21873f199b7bafb68ea677ec41317b5aad0e959b/files/GitHub_Unit_Testing.png)

We've followed the GitHub Continuous Integration workflow, as described [here](https://docs.github.com/en/actions/automating-builds-and-tests/building-and-testing-python).

> PLEASE, PAY ATTENTION pytest version 6.2.3

## Deploy the bot to Azure

To learn more about deploying a bot to Azure, see [Deploy your bot to Azure](https://aka.ms/azuredeployment) for a complete list of deployment instructions.

The workflow has been created following turorials:
- [Video here](https://www.youtube.com/watch?v=eLMYd4LGAu8&list=PL-PgMmMmma8DItgH7hO7oJHG8mHm8-7iA&index=6)
- [Azure Doc](https://learn.microsoft.com/fr-fr/azure/app-service/deploy-continuous-deployment?tabs=github)

File is available in [.github/workflows folder](https://github.com/blanchonnicolas/IA_Project10_Openclassrooms_Chatbot/blob/main/.github/workflows/main_flyme-bot-webapp.yml)

Any update in our GitHub repository will trigger a new deployment of BOT resources.
> This is using unique publish-profile set in GitHub secrets.

## Monitor the bot to Azure

We have the possibility to display transactions and events through azure portal (insights resource)

![Azure Insights transaction](https://github.com/blanchonnicolas/IA_Project10_Openclassrooms_Chatbot/blob/21873f199b7bafb68ea677ec41317b5aad0e959b/files/Insights-transaction_search.png)

![Azure Insights alerts](https://github.com/blanchonnicolas/IA_Project10_Openclassrooms_Chatbot/blob/21873f199b7bafb68ea677ec41317b5aad0e959b/files/Insights-events.png)

We have defined customized alerts, for application usage, based on track_trace in booking_dialog.py file:

    ````python
    self.telemetry_client.track_trace(name="Flight Booking process completed", properties=book_flight_user_request)
                return await step_context.end_dialog()
    ````
![Azure Insights transaction search Trace](https://github.com/blanchonnicolas/IA_Project10_Openclassrooms_Chatbot/blob/21873f199b7bafb68ea677ec41317b5aad0e959b/files/Insights-transaction_search_trace.png)

This will allow to raise error, when user rejects final BOT proposal more than 3 times (in less than 30minutes).


## Further reading

- [Bot Framework Documentation](https://docs.botframework.com)
- [Bot Basics](https://docs.microsoft.com/azure/bot-service/bot-builder-basics?view=azure-bot-service-4.0)
- [Dialogs](https://docs.microsoft.com/en-us/azure/bot-service/bot-builder-concept-dialog?view=azure-bot-service-4.0)
- [Gathering Input Using Prompts](https://docs.microsoft.com/en-us/azure/bot-service/bot-builder-prompts?view=azure-bot-service-4.0&tabs=csharp)
- [Activity processing](https://docs.microsoft.com/en-us/azure/bot-service/bot-builder-concept-activity-processing?view=azure-bot-service-4.0)
- [Azure Bot Service Introduction](https://docs.microsoft.com/azure/bot-service/bot-service-overview-introduction?view=azure-bot-service-4.0)
- [Azure Bot Service Documentation](https://docs.microsoft.com/azure/bot-service/?view=azure-bot-service-4.0)
- [Azure CLI](https://docs.microsoft.com/cli/azure/?view=azure-cli-latest)
- [Azure Portal](https://portal.azure.com)
- [Language Understanding using LUIS](https://docs.microsoft.com/en-us/azure/cognitive-services/luis/)
- [Channels and Bot Connector Service](https://docs.microsoft.com/en-us/azure/bot-service/bot-concepts?view=azure-bot-service-4.0)
- [Application insights Overview](https://docs.microsoft.com/azure/azure-monitor/app/app-insights-overview)
- [Getting Started with Application Insights](https://github.com/Microsoft/ApplicationInsights-aspnetcore/wiki/Getting-Started-with-Application-Insights-for-ASP.NET-Core)
- [Filtering and preprocessing telemetry in the Application Insights SDK](https://docs.microsoft.com/azure/azure-monitor/app/api-filtering-sampling)
