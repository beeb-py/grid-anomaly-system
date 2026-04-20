# ingestion/entsoe_client.py

import os
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


class EntsoeClient:
    BASE_URL = "https://web-api.tp.entsoe.eu/api"

    def __init__(self):
        self.api_key = os.getenv("ENTSOE_API_KEY")
        if not self.api_key:
            raise ValueError("ENTSOE_API_KEY not found in environment variables")

    def _format_time(self, dt: datetime) -> str:
        return dt.strftime("%Y%m%d%H%M")

    def _request(self, params: dict) -> str:
        params["securityToken"] = self.api_key

        response = requests.get(self.BASE_URL, params=params)

        if response.status_code != 200:
            raise Exception(f"ENTSO-E API error: {response.status_code} | {response.text}")

        return response.text

    def fetch_load_actual(self, zone: str, start: datetime, end: datetime) -> str:
        params = {
            "documentType": "A65",
            "processType": "A16",  # actual load
            "outBiddingZone_Domain": zone,
            "periodStart": self._format_time(start),
            "periodEnd": self._format_time(end),
        }
        return self._request(params)

    def fetch_load_forecast(self, zone: str, start: datetime, end: datetime) -> str:
        params = {
            "documentType": "A65",
            "processType": "A01",  # forecast
            "outBiddingZone_Domain": zone,
            "periodStart": self._format_time(start),
            "periodEnd": self._format_time(end),
        }
        return self._request(params)

    def fetch_generation(self, zone: str, start: datetime, end: datetime) -> str:
        params = {
            "documentType": "A75",
            "outBiddingZone_Domain": zone,
            "periodStart": self._format_time(start),
            "periodEnd": self._format_time(end),
        }
        return self._request(params)