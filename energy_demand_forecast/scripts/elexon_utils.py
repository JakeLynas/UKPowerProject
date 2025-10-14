from datetime import date, datetime, timedelta, time, timezone
import pandas as pd

def iso_utc(dt: datetime) -> str:
    # Always emit explicit Z-UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

def each_day(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)
