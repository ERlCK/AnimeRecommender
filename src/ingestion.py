from curl_cffi import requests
from datetime import date
import pandas as pd

class DataAcquirer:
    def __init__(self, seasons_list_url: str, seasons_options: list, base_dataset_date: int):
        self.seasons_url = seasons_list_url
        self.seasons_options = seasons_options
        self.base_dataset_date = base_dataset_date
    
    def get_current_year(self) -> int:
        current_year = date.today().year
        return current_year

    def get_years_since_base_date(self, current_year: int, base_dataset_date: int) -> list:
        all_years = []

        while base_dataset_date<current_year:
            base_dataset_date = base_dataset_date + 1
            all_years.append(base_dataset_date)

        return all_years

    def get_single_seasons_list(self, year: int, season: str) -> dict | None: 
        try:
            response = requests.get(
                f"{self.seasons_url}/{year}/{season}", 
                impersonate="chrome")

            return response.json()

        except Exception as e:
            print(f"Error: {e}")
            return None

    def get_all_seasons_list(self, all_years: list[int], seasons_options: list[str]) -> list[dict]:
        all_seasons = []

        for year in all_years:
            for season in seasons_options:
                season_data = self.get_single_seasons_list(year, season)

                if season_data is not None:
                    all_seasons.append(season_data)
        
        return all_seasons

    def create_new_data_df(self, new_data: list[dict], output_path: str) -> pd.DataFrame: 
        anime_list = []

        for season_data in new_data:
            anime_list.extend(season_data.get("data", []))

        new_data_df = pd.json_normalize(anime_list)
        new_data_df.to_csv(output_path, index=False)
        return new_data_df

if __name__ == "__main__":
    seasons_options = ["winter", "spring", "summer", "fall"]
    seasons_list_url = "https://api.jikan.moe/v4/seasons"
    base_dataset_date = 2025
    output_path = "data/new_data.csv"

    data_acquirer = DataAcquirer(seasons_list_url, seasons_options, base_dataset_date)
    
    current_year = data_acquirer.get_current_year()
    all_years = data_acquirer.get_years_since_base_date(current_year, base_dataset_date)
    new_data = data_acquirer.get_all_seasons_list(all_years, seasons_options)
    new_data_df = data_acquirer.create_new_data_df(new_data, output_path)

