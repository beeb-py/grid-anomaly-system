# processing/parser.py

import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta


def _parse_resolution(resolution: str) -> timedelta:
    # Example: PT15M, PT60M
    if resolution == "PT15M":
        return timedelta(minutes=15)
    elif resolution == "PT60M":
        return timedelta(hours=1)
    else:
        raise ValueError(f"Unsupported resolution: {resolution}")


def parse_entsoe_xml(xml_string: str) -> pd.DataFrame:
    root = ET.fromstring(xml_string)

    records = []

    # Iterate over TimeSeries
    for ts in root.findall(".//{*}TimeSeries"):

        for period in ts.findall(".//{*}Period"):
            start_str = period.find(".//{*}timeInterval/{*}start").text
            resolution = period.find(".//{*}resolution").text

            start_time = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
            delta = _parse_resolution(resolution)

            for point in period.findall(".//{*}Point"):
                position = int(point.find("{*}position").text)
                value = float(point.find("{*}quantity").text)

                timestamp = start_time + (position - 1) * delta

                records.append({
                    "timestamp": timestamp,
                    "value": value
                })

    df = pd.DataFrame(records)

    if df.empty:
        raise ValueError("Parsed DataFrame is empty — check XML structure")

    df = df.sort_values("timestamp").reset_index(drop=True)

    return df