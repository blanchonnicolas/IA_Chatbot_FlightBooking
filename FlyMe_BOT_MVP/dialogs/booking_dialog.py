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
        booking_details = step_context.options 
        booking_details.budget = step_context.result # Capture the response to the previous step's prompt

        if not booking_details.str_date or self.is_ambiguous(booking_details.str_date):
            return await step_context.begin_dialog(StrDateResolverDialog.__name__, booking_details.str_date)  # pylint: disable=line-too-long
        #     return await step_context.prompt(TextPrompt.__name__, PromptOptions(prompt=MessageFactory.text("On what date would you like to leave ?")),)
        # elif self.is_ambiguous(booking_details.str_date):
        #     return await step_context.begin_dialog(StrDateResolverDialog.__name__, booking_details.str_date)
        return await step_context.next(booking_details.str_date)


    async def end_date_step(self, step_context: WaterfallStepContext) -> DialogTurnResult:
        """Prompt for travel start date.
        This will use the END_DATE_RESOLVER_DIALOG."""
        booking_details = step_context.options
        booking_details.str_date = step_context.result # Capture the results of the previous step
        if not booking_details.end_date or self.is_ambiguous(booking_details.end_date):
            return await step_context.begin_dialog(EndDateResolverDialog.__name__, booking_details.end_date)  # pylint: disable=line-too-long
        #     return await step_context.prompt(TextPrompt.__name__, PromptOptions(prompt=MessageFactory.text("On what date would you like to be back?")),)
        # elif self.is_ambiguous(booking_details.end_date):
        #     return await step_context.begin_dialog(StrDateResolverDialog.__name__, booking_details.end_date)
        return await step_context.next(booking_details.end_date)

    async def confirm_step(self, step_context: WaterfallStepContext) -> DialogTurnResult:
        """Confirm the information the user has provided."""
        booking_details = step_context.options

        # Capture the results of the previous step
        booking_details.end_date = step_context.result

        d1 = datetime.strptime(booking_details.str_date, "%Y-%m-%d")
        d2 = datetime.strptime(booking_details.end_date, "%Y-%m-%d")
        trip_duration = (d2 - d1).days
        msg = (
            f"Please confirm, I have you traveling to: { booking_details.destination } "
            f" taking off from: { booking_details.origin } "
            f" leaving on: { booking_details.str_date } "
            f" coming back on: { booking_details.end_date } "
            f" for a total trip duration of : { trip_duration } days "
            f" within a budget of : ${ booking_details.budget } "
        )
        # Offer a YES/NO prompt.
        return await step_context.prompt(ConfirmPrompt.__name__, PromptOptions(prompt=MessageFactory.text(msg)))

    async def final_step(self, step_context: WaterfallStepContext) -> DialogTurnResult:
        """Complete the interaction and end the dialog."""
        if step_context.result:
            booking_details = step_context.options
            booking_properties = {}
            booking_properties["To"] = booking_details.destination
            booking_properties["From"] = booking_details.origin
            booking_properties["budget"] = booking_details.budget
            booking_properties["str_date"] = booking_details.str_date
            booking_properties["end_date"] = booking_details.end_date

            self.telemetry_client.track_trace(name="Flight Booking process completed", properties=booking_properties)                        
            return await step_context.end_dialog(booking_details)
        else:
            self.telemetry_client.track_trace(name="Flight Booking process aborted by user", properties=booking_properties)
            return await step_context.end_dialog()


    def is_ambiguous(self, timex: str) -> bool:
        """Ensure time is correct."""
        timex_property = Timex(timex)
        return "definite" not in timex_property.types
    