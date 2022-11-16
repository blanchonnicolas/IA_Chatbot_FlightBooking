# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Flight booking dialog."""
from datetime import date, datetime
from datatypes_date_time.timex import Timex

from botbuilder.dialogs import WaterfallDialog, WaterfallStepContext, DialogTurnResult
from botbuilder.dialogs.prompts import ConfirmPrompt, TextPrompt, PromptOptions, DateTimePrompt
from botbuilder.core import MessageFactory, BotTelemetryClient, NullTelemetryClient
from .cancel_and_help_dialog import CancelAndHelpDialog
from .str_date_resolver_dialog import StrDateResolverDialog
from .end_date_resolver_dialog import EndDateResolverDialog


class BookingDialog(CancelAndHelpDialog):
    """Flight booking implementation."""

    def __init__(self, dialog_id: str = None, telemetry_client: BotTelemetryClient = NullTelemetryClient(),):
        super(BookingDialog, self).__init__(dialog_id or BookingDialog.__name__, telemetry_client)
        self.telemetry_client = telemetry_client
        text_prompt = TextPrompt(TextPrompt.__name__)
        date_prompt = DateTimePrompt(DateTimePrompt.__name__)
        text_prompt.telemetry_client = telemetry_client

        waterfall_dialog = WaterfallDialog(WaterfallDialog.__name__,
            [
                self.destination_step,
                self.origin_step,
                self.budget_step,
                self.str_date_step,
                self.end_date_step,
                self.period_validity,
                self.trip_duration,
                self.confirm_step,
                self.final_step,
            ],
        )
        waterfall_dialog.telemetry_client = telemetry_client

        self.add_dialog(TextPrompt(TextPrompt.__name__))
        self.add_dialog(ConfirmPrompt(ConfirmPrompt.__name__))
        self.add_dialog(StrDateResolverDialog(StrDateResolverDialog.__name__, self.telemetry_client))
        #self.add_dialog(DateTimePrompt(DateTimePrompt.__name__))
        self.add_dialog(EndDateResolverDialog(EndDateResolverDialog.__name__, self.telemetry_client))
        self.add_dialog(waterfall_dialog)
        self.initial_dialog_id = WaterfallDialog.__name__

    async def destination_step(self, step_context: WaterfallStepContext) -> DialogTurnResult:
        """Prompt for destination."""
        booking_details = step_context.options
        if booking_details.destination is None:
            return await step_context.prompt(TextPrompt.__name__, PromptOptions(prompt=MessageFactory.text("To what city would you like to travel?")),)  # pylint: disable=line-too-long,bad-continuation
        return await step_context.next(booking_details.destination)

    async def origin_step(self, step_context: WaterfallStepContext) -> DialogTurnResult:
        """Prompt for origin city."""
        booking_details = step_context.options  
        booking_details.destination = step_context.result # Capture the response to the previous step's prompt
        if booking_details.origin is None:
            return await step_context.prompt(TextPrompt.__name__, PromptOptions(prompt=MessageFactory.text("From what city will you be travelling?")),)  # pylint: disable=line-too-long,bad-continuation
        return await step_context.next(booking_details.origin)

    async def budget_step(self, step_context: WaterfallStepContext) -> DialogTurnResult:
        """Prompt for Budget."""
        booking_details = step_context.options  
        booking_details.origin = step_context.result # Capture the response to the previous step's prompt
        if booking_details.budget is None:
            return await step_context.prompt(TextPrompt.__name__, PromptOptions(prompt=MessageFactory.text("What is your estimated budget (Currency = USD $)?")),)  # pylint: disable=line-too-long,bad-continuation
        return await step_context.next(booking_details.budget)

    async def str_date_step(self, step_context: WaterfallStepContext) -> DialogTurnResult:
        """Prompt for travel start date.
        This will use the STR_DATE_RESOLVER_DIALOG."""
        error_text = None
        booking_details = step_context.options 
        booking_details.budget = step_context.result # Capture the response to the previous step's prompt

        if not booking_details.str_date or self.is_ambiguous(booking_details.str_date):
            return await step_context.begin_dialog(StrDateResolverDialog.__name__, booking_details.str_date)
        elif self.is_valid_date(datetime.now().date(), booking_details.str_date) is False:
            error_text = f"Your departure date {booking_details.str_date} cannot be earlier than today's date {datetime.now().date()}"
            error_message = MessageFactory.text(error_text, error_text)
            await step_context.context.send_activity(error_message)
            booking_details.str_date = None
            return await step_context.begin_dialog(StrDateResolverDialog.__name__, booking_details.str_date)  # pylint: disable=line-too-long
        else:
            return await step_context.next(booking_details.str_date)


    async def end_date_step(self, step_context: WaterfallStepContext) -> DialogTurnResult:
        """Prompt for travel start date.
        This will use the END_DATE_RESOLVER_DIALOG."""
        error_text = None
        booking_details = step_context.options
        booking_details.str_date = step_context.result # Capture the results of the previous step

        if not booking_details.end_date or self.is_ambiguous(booking_details.end_date):
            return await step_context.begin_dialog(EndDateResolverDialog.__name__, booking_details.end_date)
        elif self.is_valid_date(datetime.now().date(), booking_details.end_date) is False:
            error_text = f"Your return date {booking_details.end_date} cannot be earlier than today's date {datetime.now().date()}"
            error_message = MessageFactory.text(error_text, error_text)
            await step_context.context.send_activity(error_message)
            booking_details.end_date = None
            return await step_context.begin_dialog(EndDateResolverDialog.__name__, booking_details.end_date)
        else:
            return await step_context.next(booking_details.end_date)

  
    async def period_validity(self, step_context: WaterfallStepContext) -> DialogTurnResult:
        """Prompt for period validity verification."""
        booking_details = step_context.options
        booking_details.end_date = step_context.result # Capture the response to the previous step's prompt
        if self.is_valid_period(booking_details.str_date, booking_details.end_date) is False:
            error_text = f"Your return date {booking_details.end_date} cannot be earlier than departure date {booking_details.str_date}"
            error_message = MessageFactory.text(error_text, error_text)
            await step_context.context.send_activity(error_message)
            booking_details.end_date = None
            return await step_context.begin_dialog(EndDateResolverDialog.__name__, booking_details.end_date)
        else:
            return await step_context.next(booking_details.end_date)

    async def trip_duration(self, step_context: WaterfallStepContext) -> DialogTurnResult:
        """Prompt for trip_duration computation."""
        booking_details = step_context.options
        booking_details.end_date = step_context.result # Capture the response to the previous step's prompt
        trip_duration = self.compute_trip_duration(booking_details.str_date, booking_details.end_date)
        return await step_context.next(trip_duration)


    async def confirm_step(self, step_context: WaterfallStepContext) -> DialogTurnResult:
        """Confirm the information the user has provided."""
        booking_details = step_context.options
        # Capture the results of the previous step
        booking_details.trip_duration = step_context.result
        # trip_duration = self.compute_trip_duration(booking_details.str_date, booking_details.end_date)
        
        msg = (
            f"Please confirm, I have you traveling to: { booking_details.destination } "
            f" taking off from: { booking_details.origin } "
            f" leaving on: { booking_details.str_date } "
            f" coming back on: { booking_details.end_date } "
            f" for a total trip duration of : { booking_details.trip_duration } days "
            f" within a budget of : ${ booking_details.budget } "
        )
        # Offer a YES/NO prompt.
        return await step_context.prompt(ConfirmPrompt.__name__, PromptOptions(prompt=MessageFactory.text(msg)))

    async def final_step(self, step_context: WaterfallStepContext) -> DialogTurnResult:
        """Complete the interaction and end the dialog."""
        booking_details = step_context.options
        book_flight_user_request = {}
        book_flight_user_request["To"] = booking_details.destination
        book_flight_user_request["From"] = booking_details.origin
        book_flight_user_request["budget"] = booking_details.budget
        book_flight_user_request["str_date"] = booking_details.str_date
        book_flight_user_request["end_date"] = booking_details.end_date
        book_flight_user_request["trip_duration"] = booking_details.trip_duration
        
        #Using Trace activity : https://learn.microsoft.com/fr-fr/azure/bot-service/using-trace-activities?view=azure-bot-service-4.0&tabs=csharp
        if step_context.result:
            self.telemetry_client.track_trace(name="Flight Booking process completed", properties=book_flight_user_request)                        
            return await step_context.end_dialog(booking_details)
        else:
            self.telemetry_client.track_trace(name="Flight Booking process aborted by user", properties=book_flight_user_request)
            return await step_context.end_dialog()


    def is_ambiguous(self, timex: str) -> bool:
        """Ensure time is correct."""
        timex_property = Timex(timex)
        return "definite" not in timex_property.types
    
    def is_valid_date(self, timex1: str, timex2: str) -> bool:
        d1 = timex1
        d2 = datetime.strptime(timex2, "%Y-%m-%d").date()
        if d1 > d2:
            return False
        else:
            return True
    
    def is_valid_period(self, timex1: str, timex2: str) -> bool:
        d1 = datetime.strptime(timex1, "%Y-%m-%d").date()
        d2 = datetime.strptime(timex2, "%Y-%m-%d").date()
        if d1 > d2:
            return False
        else:
            return True

    def compute_trip_duration(self, timex1: str, timex2: str) -> bool:
        d1 = datetime.strptime(timex1, "%Y-%m-%d").date()
        d2 = datetime.strptime(timex2, "%Y-%m-%d").date()
        trip_duration = (d2 - d1).days
        return trip_duration