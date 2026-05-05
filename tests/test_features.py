# tests/test_features.py

from datetime import datetime
from ingestion.entsoe_client import EntsoeClient
from processing.parser import parse_entsoe_xml
from processing.cleaner import clean_time_series
from features.build_features import build_features

client = EntsoeClient()

xml_data = client.fetch_load_actual(
    zone="10Y1001A1001A83F",
    start=datetime(2024, 1, 1),
    end=datetime(2024, 1, 2)
)

df_raw = parse_entsoe_xml(xml_data)
df_clean = clean_time_series(df_raw)
df_features = build_features(df_clean)

print(df_features.head())
print(df_features.columns)
print(df_features.shape)