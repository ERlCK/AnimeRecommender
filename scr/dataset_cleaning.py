import pandas as pd

class DataHandler:
    def __init__(self, dataset_path: str, output_path: str):
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.df = pd.read_csv(dataset_path)

    def checkingDataset(self) -> None:
        print(f"Dataset columns:\n{self.df.columns}\n")
        print(f"Dataset info:\n{self.df.info()}\n")
        print(f"Duplicated lines: {self.df.duplicated().sum()}\n")
        print(f"Null qtt:\n{self.df.isna().sum()}\n")
        print(f"Dataset describe:\n{self.df.describe()}\n")

    def clean_episodes_column(self) -> pd.Series:
        return pd.to_numeric(self.df['Episodes'], errors='coerce')

    def clean_duration_column(self) -> pd.Series:
        duration = self.df['Duration'].astype('string')

        hours = duration.str.extract(r"(\d+)\s*hr", expand=False)
        minutes = duration.str.extract(r"(\d+)\s*min", expand=False)
        seconds = duration.str.extract(r"(\d+)\s*sec", expand=False)

        total_minutes = (
            pd.to_numeric(hours, errors='coerce').fillna(0) * 60
            + pd.to_numeric(minutes, errors='coerce').fillna(0)
            + pd.to_numeric(seconds, errors='coerce').fillna(0) / 60
        )
        has_duration = hours.notna() | minutes.notna() | seconds.notna()

        return total_minutes.where(has_duration).clip(lower=1).round().astype("Int64")

    def clean_ranked_column(self) -> pd.Series:
        ranked = self.df['Ranked'].astype('string').str.replace("#", "", regex=False)
        return pd.to_numeric(ranked, errors='coerce')

    def clean_popularity_column(self) -> pd.Series:
        popularity = self.df['Popularity'].astype('string').str.replace("#", "", regex=False)
        return pd.to_numeric(popularity, errors='coerce')

    def clean_members_columns(self) -> pd.Series:
        members = self.df['Members'].astype('string').str.replace(",", "", regex=False)
        return pd.to_numeric(members, errors='coerce')

    def clean_favorites_columns(self)-> pd.Series:
        favorites = self.df['Favorites'].astype('string').str.replace(",", "", regex=False)
        return pd.to_numeric(favorites, errors='coerce')

    def saving_dataset(self) -> None:
        self.df.to_csv(self.output_path, index=False)

    def run_cleaning_pipeline(self) -> pd.DataFrame:
        print("Running cleaning dataset pipeline...")

        self.df['Episodes'] = self.clean_episodes_column()
        self.df['Duration'] = self.clean_duration_column()
        self.df['Ranked'] = self.clean_ranked_column()
        self.df['Popularity'] = self.clean_popularity_column()
        self.df['Members'] = self.clean_members_columns()
        self.df['Favorites'] = self.clean_favorites_columns()
        self.saving_dataset()

        print("Dataset cleaned.")
        return self.df
         
if __name__ == "__main__":
    pd.set_option('display.max_columns', None)

    DATASET_PATH = "data/mal_anime.csv"
    OUTPUT_PATH = "data/cleaned_mal_anime.csv"
    dataHandler = DataHandler(DATASET_PATH, OUTPUT_PATH)

    #dataHandler.checkingDataset()
    cleaned_df = dataHandler.run_cleaning_pipeline()
    dataHandler.checkingDataset()
