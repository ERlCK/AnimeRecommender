from sentence_transformers import SentenceTransformer
import chromadb


class AnimeRecommender:
    def __init__(self, collection_name: str, model: str, chromadb_path: str = "./chroma_db"):
        self.collection_name = collection_name
        self.model = model
        self.chromadb_path = chromadb_path
        self.client = chromadb.PersistentClient(chromadb_path)
        self.collection = self.client.get_collection(name=collection_name)
        self.embedder = SentenceTransformer(model, device="cuda")

    def _build_chromadb_filter(self, status: str | None = None, anime_type: str | None = None) -> dict | None:
        filters = []

        if status is not None:
            filters.append({"Status": status})

        if anime_type is not None:
            filters.append({"Type": anime_type})

        if len(filters) == 1:
            return filters[0]

        if len(filters) > 1:
            return {"$and": filters}

        return None

    def _passes_post_filters(
        self,
        metadata: dict,
        distance: float,
        min_score: float | None = None,
        genre: str | None = None,
        themes: str | None = None,
        max_distance: float | None = None,
    ) -> bool:
        if max_distance is not None and distance > max_distance:
            return False

        if min_score is not None:
            try:
                score = float(metadata["Score"])
            except (KeyError, TypeError, ValueError):
                return False

            if score < min_score:
                return False

        if genre is not None:
            genres = str(metadata.get("Genres", "")).lower()
            if genre.lower() not in genres:
                return False

        if themes is not None:
            anime_themes = str(metadata.get("Themes", "")).lower()
            if themes.lower() not in anime_themes:
                return False

        return True

    def recommend(
        self,
        query: str,
        n_results: int = 5, #Final number of recommendations to show
        status: str | None = "Finished Airing",
        min_score: float | None = None,
        anime_type: str | None = None, #"TV", "Movie", "OVA", "ONA", "Special", "Music"
        genre: str | None = None,
        themes: str | None = None,
        max_distance: float | None = None,
        fetch_results: int = 15, #Number of ChromaDB results fetched before post-filtering (filtered by score, for example)
    ) -> list[dict]:
        query_embedding = self.embedder.encode([query]).tolist()
        chromadb_filter = self._build_chromadb_filter(status=status, anime_type=anime_type)

        anime_results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=fetch_results,
            where=chromadb_filter,
        )

        recommendations = []
        metadatas = anime_results["metadatas"][0]
        distances = anime_results["distances"][0]

        for metadata, distance in zip(metadatas, distances):
            if not self._passes_post_filters(
                metadata,
                distance,
                min_score=min_score,
                genre=genre,
                themes=themes,
                max_distance=max_distance,
            ):
                continue

            recommendation = dict(metadata)
            recommendation["Distance"] = distance
            recommendations.append(recommendation)

            if len(recommendations) == n_results:
                break

        return recommendations

    def show_recommendations(self, recommendations: list[dict], query: str) -> None:
        print(f"\nSearch results for the query:\n{query}")

        if not recommendations:
            print("\nNo recommendations found with the selected filters.")
            return

        for index, recommendation in enumerate(recommendations, start=1):
            print(f"\n{index}. {recommendation['title']}")
            print(f"   Score: {recommendation['Score']}")
            print(f"   Status: {recommendation['Status']}")
            print(f"   Year: {recommendation['Released_Year']}")
            print(f"   Type: {recommendation['Type']}")
            print(f"   Genres: {recommendation['Genres']}")
            print(f"   Themes: {recommendation.get('Themes', '')}")
            print(f"   Distance: {recommendation['Distance']:.4f}")


if __name__ == "__main__":
    collection_name = "dataset_embeddings"
    model = "sentence-transformers/all-mpnet-base-v2"
    query = "An anime about vikings and a history of revenge."

    recommender = AnimeRecommender(collection_name, model)
    recommendations = recommender.recommend(
        query=query,
        n_results=5,
        status="Finished Airing",
        min_score=7.0,
        max_distance=0.85,
    )
    recommender.show_recommendations(recommendations, query)
