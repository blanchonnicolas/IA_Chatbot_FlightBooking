# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


class BookingDetails:
    def __init__(
        self,
        destination: str = None,
        origin: str = None,
        budget: str = None,
        str_date: str = None,
        end_date: str = None,
        travel_date: str = None,
        # unsupported_airports=None, #To be removed
    ):
        # if unsupported_airports is None:
        #     unsupported_airports = []
        self.destination = destination
        self.origin = origin
        self.budget = budget
        self.str_date = str_date
        self.end_date = end_date
        self.travel_date = travel_date
        # self.unsupported_airports = unsupported_airports #To be removed
