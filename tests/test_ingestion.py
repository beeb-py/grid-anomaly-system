# tests/test_ingestion.py

from datetime import datetime
from ingestion.entsoe_client import EntsoeClient
from processing.parser import parse_entsoe_xml

client = EntsoeClient()

xml_data = client.fetch_load_actual(
    zone="10Y1001A1001A83F",  # Germany
    start=datetime(2024, 1, 1),
    end=datetime(2024, 1, 2)
)

df = parse_entsoe_xml(xml_data)

print(df.head())
print(df.tail())
print(df.shape)