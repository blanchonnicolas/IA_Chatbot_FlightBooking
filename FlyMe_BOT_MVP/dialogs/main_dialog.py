# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from botbuilder.dialogs import (
    ComponentDialog,
    WaterfallDialog,
    WaterfallStepContext,
    DialogTurnResult,
)
from botbuilder.dialogs.prompts import ConfirmPrompt, TextPrompt, PromptOptions
from botbuilder.core import (
    MessageFactory,
    TurnContext,
    BotTelemetryClient,
    NullTelemetryClient,
)
from botbuilder.schema import InputHints
from datatypes_date_time.timex import Timex
from booking_details import BookingDetails
from flight_booking_recognizer import FlightBookingRecognizer
from helpers.luis_helper import LuisHelper, Intent
from .booking_dialog import BookingDialog



class MainDialog(ComponentDialog):
    def __init__(
        self,
        luis_recognizer: FlightBookingRecognizer,
        booking_dialog: BookingDialog,
        booking_details: BookingDetails = None,
        telemetry_client: BotTelemetryClient = None,
    ):
        super(MainDialog, self).__init__(MainDialog.__name__)
        self.telemetry_client = telemetry_client or NullTelemetryClient()

        text_prompt = TextPrompt(TextPrompt.__name__)
        text_prompt.telemetry_client = self.telemetry_client

        booking_dialog.telemetry_client = self.telemetry_client

        wf_dialog = WaterfallDialog(
            "WFDialog", [self.intro_step, self.act_step, self.final_step]
        )
        wf_dialog.telemetry_client = self.telemetry_client

        self._luis_recognizer = luis_recognizer
        self._booking_dialog_id = booking_dialog.id

        if booking_details is None:
            self._booking_details = BookingDetails()
        else:
            self._booking_details = booking_details

        self.add_dialog(text_prompt)
        self.add_dialog(booking_dialog)
        self.add_dialog(wf_dialog)

        self.initial_dialog_id = "WFDialog"

    async def intro_step(self, step_context: WaterfallStepContext) -> DialogTurnResult:
        if not self._luis_recognizer.is_configured:
            await step_context.context.send_activity(
                MessageFactory.text(
                    "NOTE: LUIS is not configured. To enable all capabilities, add 'LuisAppId', 'LuisAPIKey' and "
                    "'LuisAPIHostName' to the appsettings.json file.",
                    input_hint=InputHints.ignoring_input,
                )
            )

            return await step_context.next(None)
        message_text = (
            str(step_context.options)
            if step_context.options
            else "Welcome to FlyMe Chatbot, How can I help you to book your next trip ?"
        )
        prompt_message = MessageFactory.text(message_text, message_text, InputHints.expecting_input)

        return await step_context.prompt(TextPrompt.__name__, PromptOptions(prompt=prompt_message))

    async def act_step(self, step_context: WaterfallStepContext) -> DialogTurnResult:
        if not self._luis_recognizer.is_configured:
            # LUIS is not configured, we just run the BookingDialog path with an empty BookingDetailsInstance.
            return await step_context.begin_dialog(self._booking_dialog_id, self._booking_details) 

        # Call LUIS and gather any potential booking details. (Note the TurnContext has the response to the prompt.)
        intent, luis_result = await LuisHelper.execute_luis_query(self._luis_recognizer, step_context.context)

        if intent == Intent.BOOK_FLIGHT.value and luis_result:
            # Run the BookingDialog giving it whatever details we have from the LUIS call.
            return await step_context.begin_dialog(self._booking_dialog_id, luis_result)

        elif intent == Intent.GREETING.value:
            greeting_text = "My best regards 😀 !"
            return await step_context.replace_dialog(self.id, greeting_text)

        else:
            didnt_understand_text = ("Sorry, I didn't get that. Please try asking in a different way")
            didnt_understand_message = MessageFactory.text(didnt_understand_text, didnt_understand_text, InputHints.ignoring_input)
            await step_context.context.send_activity(didnt_understand_message)
            
            self.telemetry_client.track_trace(name="Flight Booking process failed by user misunderstanding", properties={"step_context_index":str(step_context.index), "step_context_result":step_context.result})
            #return await step_context.end_dialog()
            
            if type(self.telemetry_client) != NullTelemetryClient:
                potential_bug_detected_text = ("Please, Call your admin to deep dive into LUIS traces (Microsoft Insights)")
                potential_bug_detected_message = MessageFactory.text(potential_bug_detected_text, potential_bug_detected_text, InputHints.ignoring_input)
                await step_context.context.send_activity(potential_bug_detected_message)

        return await step_context.next(None)

    async def final_step(self, step_context: WaterfallStepContext) -> DialogTurnResult:
        # If the child dialog ("BookingDialog") was cancelled or the user failed to confirm,
        # the Result here will be null.
        if step_context.result is None:
            msg_txt = (f"I apologizes for the misunderstanding, please retry your flight booking request ")
            message = MessageFactory.text(msg_txt, msg_txt, InputHints.ignoring_input)
            await step_context.context.send_activity(message)
        else:
            result = step_context.result
            # If the call to the booking service was successful tell the user.
            # str_date_property = Timex(result.str_date)
            # end_date_property = Timex(result.end_date)
            # travel_date_msg = str_date_property.to_natural_language(datetime.now())
            msg_txt = (
                        f"I have you booked a flight from {result.origin} to {result.destination} "
                        f"leaving on {result.str_date}, and coming back on {result.end_date} "
                        f"for a very cheap price of $ {result.budget} "
                        # f"str_date_msg is {str_date_msg}"
                        # f"end_date_msg is {end_date_msg}"
                    )
            message = MessageFactory.text(msg_txt, msg_txt, InputHints.ignoring_input)
            await step_context.context.send_activity(message)

        prompt_message = "What else can I do for you?"
        return await step_context.replace_dialog(self.id, prompt_message)

    # @staticmethod


    # async def _show_warning_for_unsupported_dates(
    #     context: TurnContext, luis_result: BookingDetails
    # ) -> None:
    #     """
    #     Shows a warning if the reformat are not in fulfiling Timex rules.
    #     """
    #     if luis_result.unsupported_airports:
    #         message_text = (f"Sorry but the following airports are not supported:"
    #             f" {', '.join(luis_result.unsupported_airports)}")
    #         message = MessageFactory.text(message_text, message_text, InputHints.ignoring_input)
    #         await context.send_activity(message)
