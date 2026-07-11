import pandas as pd
from dataset_io import load_dataset, save_dataset
from dataset_cleaning import DataHandler


MAPPING_JIKAN_TO_OLD = {
    "mal_id": "myanimelist_id",
    "title": "title",
    "synopsis": "description",
    "images.jpg.image_url": "image",
    "type": "Type",
    "episodes": "Episodes",
    "status": "Status",
    "season": "Released_Season",
    "year": "Released_Year",
    "source": "Source",
    "genres": "Genres",
    "themes": "Themes",
    "studios": "Studios",
    "producers": "Producers",
    "demographics": "Demographic",
    "duration": "Duration",
    "rating": "Rating",
    "score": "Score",
    "rank": "Ranked",
    "popularity": "Popularity",
    "members": "Members",
    "favorites": "Favorites",
    "url": "source_url",
}


class DatasetUpdater:
    def __init__(self, MAPPING_JIKAN_TO_OLD: dict, new_data: pd.DataFrame):
        self.MAPPING_JIKAN_TO_OLD = MAPPING_JIKAN_TO_OLD
        self.new_data = new_data

    def fixing_new_data(self, new_data: pd.DataFrame) -> pd.DataFrame:
        columns_to_keep = []

        for column in new_data.columns:
            if column in self.MAPPING_JIKAN_TO_OLD:
                columns_to_keep.append(column)

        new_data = new_data[columns_to_keep]
        new_data = new_data.rename(columns=self.MAPPING_JIKAN_TO_OLD)
        self.new_data = new_data
        return new_data
    
    def merging_datasets(self, new_data_cleaned: pd.DataFrame, mal_anime_data: pd.DataFrame) -> pd.DataFrame:
        merged_data = pd.concat([mal_anime_data, new_data_cleaned], ignore_index=True)
        merged_data = merged_data.drop_duplicates(subset=["myanimelist_id"], keep="last")

        return merged_data

if __name__ == "__main__":
    RAW_NEW_DATA_PATH = "data/new_data.csv"
    FIXED_NEW_DATA_PATH = "data/new_data_fixed.csv"
    CLEANED_NEW_DATA_PATH = "data/new_data_cleaned.csv"
    BASE_DATA_PATH = "data/mal_anime.csv"
    MERGED_DATA_PATH = "data/merged_datasets.csv"

    new_data = load_dataset(RAW_NEW_DATA_PATH)
    dataset_updater = DatasetUpdater(MAPPING_JIKAN_TO_OLD, new_data)
    new_data_fixed = dataset_updater.fixing_new_data(new_data)
    save_dataset(new_data_fixed, FIXED_NEW_DATA_PATH)

    
    data_handler = DataHandler(FIXED_NEW_DATA_PATH, CLEANED_NEW_DATA_PATH)
    new_data_cleaned = data_handler.run_cleaning_pipeline()

    mal_anime_data = load_dataset(BASE_DATA_PATH)
    merged_datasets = dataset_updater.merging_datasets(new_data_cleaned, mal_anime_data)
    save_dataset(merged_datasets, MERGED_DATA_PATH)
