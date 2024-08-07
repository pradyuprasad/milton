from .models import Series
import requests
from .database_ops import insert_series
from .config import config, APIKeyNotFoundError
try:
    FRED_API_KEY = config.get_api_key('FRED_API_KEY')
except APIKeyNotFoundError as e:
    print(e)
    raise e

def store_series_in_DB(series_fred_id: str) -> bool:
    try:
        response = requests.get(f'https://api.stlouisfed.org/fred/series?series_id={series_fred_id}&api_key={FRED_API_KEY}&file_type=json')
        if response.status_code == 200:
            data = response.json()['seriess'][0]
            series = Series(
                fred_id=data['id'],
                title=data['title'],
                observation_start=data['observation_start'],
                observation_end=data['observation_end'],
                frequency=data['frequency'],
                frequency_short=data['frequency_short'],
                units=data['units'],
                units_short=data['units_short'],
                seasonal_adjustment=data['seasonal_adjustment'],
                seasonal_adjustment_short=data['seasonal_adjustment_short'],
                last_updated=data['last_updated'],
                popularity=data['popularity'],
                notes=data.get('notes')
            )
            return insert_series(series)
        else:
            print(f"Failed to fetch data for series_id {series_fred_id}: HTTP {response.status_code}")
            return False
    except requests.RequestException as e:
        print(f"Request exception: {e}")
        return False

