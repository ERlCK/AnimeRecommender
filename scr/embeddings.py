from dataset_io import load_dataset
from sentence_transformers import SentenceTransformer
import pandas as pd
import chromadb
import time

COLUMNS_TO_KEEP = [
    "myanimelist_id",
    "title",
    "description",
    "Type",
    "Genres",
    "Themes",
    "Demographic",
    "Rating",
    "Status",
    "Score",
    "Released_Year",
]

BATCH_SIZE = 500


class DataEmbedding:
    def __init__(self, BASE_DATA_PATH: str, BASE_DATA_TO_EMBED_PATH: str, OUTPUT_EMBEDDINGS_PATH: str, model: str, base_dataset: pd.DataFrame, collection_name: str):
        self.BASE_DATA_PATH = BASE_DATA_PATH
        self.BASE_DATA_TO_EMBED_PATH = BASE_DATA_TO_EMBED_PATH
        self.OUTPUT_EMBEDDINGS_PATH = OUTPUT_EMBEDDINGS_PATH
        self.model = model
        self.base_dataset = base_dataset
        self.embedder = None
        self.client = None
        self.collection_name = collection_name

    def create_file_to_embed(self, base_dataset: pd.DataFrame, BASE_DATA_TO_EMBED_PATH: str):
        columns_to_keep = []
        for column in base_dataset.columns:
            if column in COLUMNS_TO_KEEP:
                columns_to_keep.append(column)

        dataset_to_embed = base_dataset[columns_to_keep]
        dataset_to_embed.to_csv(BASE_DATA_TO_EMBED_PATH, index=False)
    
    def load_embedding_model(self):
        self.embedder = SentenceTransformer(self.model, device="cuda")

    def embed_data(self, BASE_DATA_TO_EMBED_PATH: str, OUTPUT_EMBEDDINGS_PATH: str) -> pd.DataFrame:
        dataset_to_embed = load_dataset(BASE_DATA_TO_EMBED_PATH)
        #dataset_to_embed = dataset_to_embed.head(500) #TESTING
        self.load_embedding_model()

        dataset_to_embed["embedding_text"] = (
            "Title: " + dataset_to_embed["title"].fillna("") + "\n" +
            "Type: " + dataset_to_embed["Type"].fillna("") + "\n" +
            "Genres: " + dataset_to_embed["Genres"].fillna("") + "\n" +
            "Themes: " + dataset_to_embed["Themes"].fillna("") + "\n" +
            "Demographic: " + dataset_to_embed["Demographic"].fillna("") + "\n" +
            "Rating: " + dataset_to_embed["Rating"].fillna("") + "\n" +
            "Description: " + dataset_to_embed["description"].fillna("")
        )

        start_time = time.perf_counter()
        embedding_texts = dataset_to_embed["embedding_text"].tolist()
        embeddings = []

        for start_index in range(0, len(embedding_texts), BATCH_SIZE):
            end_index = start_index + BATCH_SIZE
            batch_texts = embedding_texts[start_index:end_index]
            batch_embeddings = self.embedder.encode(batch_texts)
            embeddings.extend(batch_embeddings.tolist())

            completed = min(end_index, len(embedding_texts))
            progress = (completed / len(embedding_texts)) * 100
            print(f"Embeddings progress: {completed}/{len(embedding_texts)} ({progress:.2f}%)")

        elapsed_time = time.perf_counter() - start_time
        print(f"Generated {len(dataset_to_embed)} embeddings in {elapsed_time:.2f} seconds")

        dataset_to_embed["embedding"] = embeddings
        dataset_to_embed.to_pickle(OUTPUT_EMBEDDINGS_PATH)

        return dataset_to_embed

    def inicialize_chromadb(self, collection_name: str, OUTPUT_EMBEDDINGS_PATH: str):
        embedded_dataset = pd.read_pickle(OUTPUT_EMBEDDINGS_PATH)

        self.client = chromadb.PersistentClient("./chroma_db")
        collection = self.client.get_or_create_collection(name=collection_name)

        ids = embedded_dataset["myanimelist_id"].astype(str).tolist()
        embeddings = embedded_dataset["embedding"].tolist()
        documents = embedded_dataset["embedding_text"].tolist()
        metadatas = embedded_dataset[["myanimelist_id", "title", "Status", "Score", "Released_Year", "Type", "Genres"]].fillna("").to_dict("records")

        for start_index in range(0, len(ids), BATCH_SIZE):
            end_index = start_index + BATCH_SIZE
            collection.upsert(
                ids=ids[start_index:end_index],
                embeddings=embeddings[start_index:end_index],
                documents=documents[start_index:end_index],
                metadatas=metadatas[start_index:end_index],
            )

            completed = min(end_index, len(ids))
            progress = (completed / len(ids)) * 100
            print(f"ChromaDB progress: {completed}/{len(ids)} ({progress:.2f}%)")

if __name__ == "__main__":
    BASE_DATA_PATH = "data/merged_datasets.csv"
    BASE_DATA_TO_EMBED_PATH = "data/dataset_to_embed.csv"
    OUTPUT_EMBEDDINGS_PATH = "data/embedded_dataset.pkl"
    model = "sentence-transformers/all-mpnet-base-v2"
    base_dataset = load_dataset(BASE_DATA_PATH)
    collection_name = "dataset_embeddings"

    data_embedding = DataEmbedding(BASE_DATA_PATH, BASE_DATA_TO_EMBED_PATH, OUTPUT_EMBEDDINGS_PATH, model, base_dataset, collection_name)

    data_embedding.create_file_to_embed(base_dataset, BASE_DATA_TO_EMBED_PATH)
    data_embedding.embed_data(BASE_DATA_TO_EMBED_PATH, OUTPUT_EMBEDDINGS_PATH)
    data_embedding.inicialize_chromadb(collection_name, OUTPUT_EMBEDDINGS_PATH)
