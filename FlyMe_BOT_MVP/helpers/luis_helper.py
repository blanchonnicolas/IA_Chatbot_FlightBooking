# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from enum import Enum
from typing import Dict
from botbuilder.ai.luis import LuisRecognizer
from botbuilder.core import IntentScore, TopIntent, TurnContext
from datetime import date, datetime
from booking_details import BookingDetails



class Intent(Enum):
    BOOK_FLIGHT = "Book_flight"
    #CANCEL = "Cancel"
    GREETING = "Greeting"
    NONE_INTENT = "None"


def top_intent(intents: Dict[Intent, dict]) -> TopIntent:
    max_intent = Intent.NONE_INTENT
    max_value = 0.0

    for intent, value in intents:
        intent_score = IntentScore(value)
        if intent_score.score > max_value:
            max_intent, max_value = intent, intent_score.score

    return TopIntent(max_intent, max_value)


class LuisHelper:
    @staticmethod
    async def execute_luis_query(
        luis_recognizer: LuisRecognizer, turn_context: TurnContext
    ) -> (Intent, object):
        """
        Returns an object with preformatted LUIS results for the bot's dialogs to consume.
        """
        result = None
        intent = None

        try:
            recognizer_result = await luis_recognizer.recognize(turn_context)

            intent = (
                sorted(
                    recognizer_result.intents,
                    key=recognizer_result.intents.get,
                    reverse=True,
                )[:1][0]
                if recognizer_result.intents
                else None
            )

            if intent == Intent.BOOK_FLIGHT.value:
                result = BookingDetails()

                # We need to get the result from the LUIS JSON which at every level returns an array.
                #Entity To Destination
                to_entities = recognizer_result.entities.get("$instance", {}).get("To", [])
                if len(to_entities) > 0:
                    result.destination = to_entities[0]["text"].capitalize()

                #Entity From Origin
                from_entities = recognizer_result.entities.get("$instance", {}).get("From", [])
                if len(from_entities) > 0:
                    result.origin = from_entities[0]["text"].capitalize()

                #Entity budget
                budget_entities = recognizer_result.entities.get("$instance", {}).get("budget", [])
                if len(budget_entities) > 0:
                    result.budget = budget_entities[0]["text"]
               
                #Entity travel_date
                date_entities = recognizer_result.entities.get("datetime", [])
                #str_date_entities = recognizer_result.entities.get("str_date", [])
                str_date_entities = recognizer_result.entities.get("$instance", {}).get("str_date", [])
                #end_date_entities = recognizer_result.entities.get("end_date", [])
                end_date_entities = recognizer_result.entities.get("$instance", {}).get("end_date", [])
                
                if date_entities:
                    if len(date_entities)==1: # 1 datetime is retreived in timex
                        timex = date_entities[0]["timex"]
                        if date_entities[0]['type'] == 'daterange': #Option to split daterange for the flights str_date and end_date
                            datetime_range = timex[0].strip('(').strip(')').split(',')
                            result.str_date = datetime_range[0]
                            result.end_date = datetime_range[1]
                            
                        elif date_entities[0]['type'] == 'date':
                            result.str_date = timex[0].split("T")[0]
                    
                    elif len(date_entities)==2: # 2 datetimes are retreived in timex
                        timex1 = date_entities[0]["timex"]
                        timex2 = date_entities[1]["timex"]
                        if (date_entities[0]['type'] == 'date' and date_entities[1]['type'] == 'date'):
                            if timex1[0] <= timex2[0]:
                                result.str_date = timex1[0].split("T")[0]
                                result.end_date = timex2[0].split("T")[0]
                            else:
                                result.str_date = timex2[0].split("T")[0]
                                result.end_date = timex1[0].split("T")[0]
                        
                        # if date_entities[0]['type'] == 'duration': #1 datetime is duration type
                        #     result.str_date = timex1[0].split("T")[0]
                        #     result.end_date = timex2[0].split("T")[0] + timex1[0].split("T")[0]
                        # elif if date_entities[1]['type'] == 'duration':
                        #     result.str_date = timex1[0].split("T")[0]
                        #     result.end_date = timex2[0].split("T")[0] + timex1[0].split("T")[0]

                 
                # #el
                # if ((len(str_date_entities) > 0) or (len(end_date_entities) > 0)):
                #     if (len(str_date_entities) > 0) and (result.str_date == recognizer_result.entities["str_date"]):
                        
                #         if recognizer_result.entities.get("str_date")[0]:
                #             result.str_date = get_timex(str_date_entities[0]["text"])
                #     if (len(end_date_entities) > 0):
                #         if recognizer_result.entities.get("end_date")[0]:
                #             result.str_date = get_timex(end_date_entities[0]["text"])

                else:
                    result.str_date = None
                    result.end_date = None


        except Exception as exception:
            print(exception)

        return intent, result
