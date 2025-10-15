from enum import Enum

class Query(str, Enum):
    """Store queries"""

    PREDICTED_DATA = "bmrs/api/v1/forecast/indicated/day-ahead/history"
    """
    Day and day ahead predicted demand, supply, margin, inbalance
    updated every 30 mins.
    Data is split into zones so group over publish and start times for UK level.
    Add /stream for live data
    """

    PREDICTED_DEMAND = "bmrs/api/v1/forecast/demand/day-ahead/history"
    """
    Day and day ahead predicted demand updated every 30 mins.
    Data is split into zones so group over publish and start times for UK level.
    Add /stream for live data
    """

    ACTUAL_DEMAND = "bmrs/api/v1/demand/outturn"
    """
    30 min updated values for avtual demand.
    Returns the national demand and the nation demand + station transformer load, pumped storage demand, and interconnectors
    I want the later
    """

    ACTUAL_SUPPLY = "bmrs/api/v1/datasets/FUELHH"
    """
    30 min updated values for avtual supply.
    Data is split into fuel types so group over publish and start times for UK level.
    Add /stream for live data
    """

class DataTypes(str, Enum):
    FORECAST = "forecast"
    ACTUAL = "actual"