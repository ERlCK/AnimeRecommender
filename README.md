# AnimeRecommender

AnimeRecommender is a work-in-progress recommendation system for anime discovery.

The project combines dataset cleaning, Jikan API ingestion, semantic embeddings, and a local ChromaDB vector database to recommend anime based on natural language queries.

## Project Status

This repository is currently under active development.

The core data pipeline is already working:

- Clean and normalize an existing MyAnimeList anime dataset
- Fetch newer seasonal anime data from the Jikan API
- Align the new API data with the original dataset schema
- Merge old and new anime records
- Generate semantic embeddings with Sentence Transformers
- Store embeddings in ChromaDB
- Query the vector database with natural language prompts

## Current Goal

The short-term goal is to build a recommendation workflow where a user can describe what they want to watch, for example:

```text
    An anime about vikings and a history of revenge.
```

The system then searches for semantically similar anime and returns recommendations with useful metadata such as title, score, status, release year, type, and genres.

## Tech Stack

- Python
- pandas
- Jikan API
- curl_cffi
- Sentence Transformers
- ChromaDB
- PyTorch / CUDA for GPU-accelerated embedding generation

## Repository Structure

```text
AnimeRecommender/
|-- data/                  # Local datasets, ignored by Git
|-- src/
|   |-- dataset_cleaning.py # Cleaning and normalization logic
|   |-- dataset_io.py       # Dataset loading/saving helpers
|   |-- dataset_update.py   # Schema alignment and dataset merging
|   |-- embeddings.py       # Embedding generation and ChromaDB queries
|   |-- evaluation.py       # Retrieval quality evaluation script
|   |-- ingestion.py        # Jikan API data acquisition
|   `-- recommender.py      # Query and recommendation logic
|-- main.py                 # Pipeline orchestration
|-- .gitignore
`-- README.md
```

## Pipeline Overview

1. Load the base anime dataset.
2. Clean columns such as duration, episodes, rank, popularity, members, and favorites.
3. Fetch newer anime data from Jikan.
4. Normalize the Jikan response to match the base dataset schema.
5. Merge both datasets and remove duplicated anime by `myanimelist_id`.
6. Build embedding text from relevant columns.
7. Generate embeddings using `sentence-transformers/all-mpnet-base-v2`.
8. Store the vectors and metadata in ChromaDB.
9. Query ChromaDB to retrieve semantically similar anime.

## Dataset

The base dataset used during development was downloaded from Kaggle:

[MyAnimeList 2025](https://www.kaggle.com/datasets/syahrulapriansyah2/myanimelist-2025)

The dataset files are not included in this repository because local CSV files are ignored by Git. To run the full pipeline, download the dataset from Kaggle and place the cleaned base file at:

```text
data/cleaned_mal_anime.csv
```

The project then fetches newer seasonal anime data from the Jikan API and merges it with this local base dataset.

## How to Run

This project is still a work in progress, so setup is currently manual.

1. Clone the repository:

```bash
git clone https://github.com/ERlCK/AnimeRecommender.git
cd AnimeRecommender
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

3.Install the main dependencies:

```bash
pip install -r requirements.txt
```

If you want GPU acceleration for embeddings, install the PyTorch version that matches your CUDA setup from the official PyTorch instructions.

4. Add the local dataset file.

Download the base dataset from [MyAnimeList 2025 on Kaggle](https://www.kaggle.com/datasets/syahrulapriansyah2/myanimelist-2025) and place the cleaned CSV at:

```text
data/cleaned_mal_anime.csv
```

5. Run the full pipeline:

```bash
python main.py
```

The pipeline will fetch new anime data, clean and merge datasets, generate embeddings, and update the local ChromaDB database.

6. Evaluate recommendation quality:

```bash
python src/evaluation.py
```

The evaluation script runs hand-labeled queries and reports Top-1 accuracy, Top-5 recall, Top-10 recall, and MRR.

## Example Query Prototype

The current prototype can query the local ChromaDB collection using the same embedding model used to index the anime dataset.

Example query:

```text
An anime about vikings and a history of revenge.
```

The result output is formatted with:

- Title
- Score
- Status
- Release year
- Type
- Genres
- Vector distance

## Next Steps

- Improve recommendation quality with filtering by score, status, and release year
- Integrate MyAnimeList user profile data
- Avoid recommending anime already watched by the user
- Add a cleaner CLI or simple web interface
- Add setup instructions and dependency management
- Add tests for dataset cleaning and update logic

## Notes

Generated datasets, embedding files, and the local ChromaDB database are intentionally ignored by Git because they can be large and are environment-specific.

This README was drafted with AI assistance and reviewed/edited by me.
