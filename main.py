from pathlib import Path
import sys
import logging



SCR_DIR = Path(__file__).resolve().parent / "scr"
sys.path.append(str(SCR_DIR))

from scr.dataset_cleaning import DataHandler
from scr.dataset_io import load_dataset, save_dataset
from scr.dataset_update import DatasetUpdater, MAPPING_JIKAN_TO_OLD
from scr.ingestion import DataAcquirer


def main() -> None:
    logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logging.info("Starting pipeline...")
    seasons_options = ["winter", "spring", "summer", "fall"]
    seasons_list_url = "https://api.jikan.moe/v4/seasons"
    base_dataset_date = 2025

    raw_new_data_path = "data/new_data.csv"
    fixed_new_data_path = "data/new_data_fixed.csv"
    cleaned_new_data_path = "data/new_data_cleaned.csv"
    base_data_path = "data/cleaned_mal_anime.csv"
    merged_data_path = "data/merged_datasets.csv"

    logging.info("Fetching new anime data...")
    data_acquirer = DataAcquirer(seasons_list_url, seasons_options, base_dataset_date)
    current_year = data_acquirer.get_current_year()
    all_years = data_acquirer.get_years_since_base_date(current_year, base_dataset_date)
    new_data = data_acquirer.get_all_seasons_list(all_years, seasons_options)
    data_acquirer.create_new_data_df(new_data, raw_new_data_path)
    logging.info("New anime data obtained. ")

    raw_new_data = load_dataset(raw_new_data_path)
    dataset_updater = DatasetUpdater(MAPPING_JIKAN_TO_OLD, raw_new_data)

    new_data_fixed = dataset_updater.fixing_new_data(raw_new_data)
    save_dataset(new_data_fixed, fixed_new_data_path)

    data_handler = DataHandler(fixed_new_data_path, cleaned_new_data_path)
    new_data_cleaned = data_handler.run_cleaning_pipeline()
    logging.info("New anime dataset fixed and cleaned.")

    logging.info("Merging new data with old data...")
    base_data = load_dataset(base_data_path)
    merged_data = dataset_updater.merging_datasets(new_data_cleaned, base_data)
    save_dataset(merged_data, merged_data_path)
    logging.info("Data merged.")
    logging.info("Pipeline finished.")

if __name__ == "__main__":
    main()
