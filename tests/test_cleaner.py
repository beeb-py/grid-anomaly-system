# tests/test_cleaner.py

from datetime import datetime
from ingestion.entsoe_client import EntsoeClient
from processing.parser import parse_entsoe_xml
from processing.cleaner import clean_time_series

client = EntsoeClient()

xml_data = client.fetch_load_actual(
    zone="10Y1001A1001A83F",
    start=datetime(2024, 1, 1),
    end=datetime(2024, 1, 2)
)

df_raw = parse_entsoe_xml(xml_data)
df_clean = clean_time_series(df_raw)

print(df_clean.head())
print(df_clean.tail())
print(df_clean.shape)