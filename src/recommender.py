import re
import unicodedata

from sentence_transformers import SentenceTransformer
import chromadb

SIMILARITY_PATTERNS = [
    r"similar to ['\"]?(.+?)['\"]?$",
    r"like ['\"]?(.+?)['\"]?$",
    r"in the style of ['\"]?(.+?)['\"]?$",
]

ITEM_TO_ITEM_FETCH_RESULTS = 50
GENRE_MATCH_WEIGHT = 0.04
THEME_MATCH_WEIGHT = 0.07
SCORE_WEIGHT = 0.02


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

    def _extract_reference_title(self, query: str) -> str | None:
        for pattern in SIMILARITY_PATTERNS:
            match = re.search(pattern, query, flags=re.IGNORECASE)

            if match:
                return match.group(1).strip(" .?!'\"")

        return None

    def _normalize_text(self, text: str) -> str:
        text = unicodedata.normalize("NFKD", text)
        text = "".join(character for character in text if not unicodedata.combining(character))
        text = re.sub(r"[^a-zA-Z0-9]+", " ", text)
        return text.lower().strip()

    def _get_metadata_titles(self, metadata: dict) -> list[str]:
        title_fields = ["title", "title_english", "title_japanese", "title_synonyms"]
        titles = []

        for field in title_fields:
            value = str(metadata.get(field, "")).strip()

            if value:
                titles.append(value)

        return titles

    def _metadata_contains_title(self, metadata: dict, title: str) -> bool:
        normalized_title = self._normalize_text(title)

        if not normalized_title:
            return False

        for metadata_title in self._get_metadata_titles(metadata):
            normalized_metadata_title = self._normalize_text(metadata_title)

            if not normalized_metadata_title:
                continue

            if normalized_title == normalized_metadata_title:
                return True

            if len(normalized_title) >= 5 and normalized_title in normalized_metadata_title:
                return True

            if len(normalized_metadata_title) >= 5 and normalized_metadata_title in normalized_title:
                return True

        return False

    def search_anime_by_title(self, title: str, limit: int = 10, include_embeddings: bool = False) -> list[dict]:
        include = ["metadatas"]

        if include_embeddings:
            include.append("embeddings")

        results = self.collection.get(include=include)
        metadatas = results["metadatas"]
        embeddings = results.get("embeddings")
        ids = results["ids"]
        matches = []

        for index, (anime_id, metadata) in enumerate(zip(ids, metadatas)):
            if self._metadata_contains_title(metadata, title):
                match = {
                    "id": anime_id,
                    "metadata": metadata,
                }

                if include_embeddings and embeddings is not None:
                    match["embedding"] = embeddings[index]

                matches.append(match)

                if len(matches) == limit:
                    break

        return matches

    def _find_reference_anime(self, title: str) -> dict | None:
        matches = self.search_anime_by_title(title, limit=1, include_embeddings=True)

        if matches:
            return matches[0]

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

    def _split_metadata_values(self, metadata: dict, field: str) -> set[str]:
        value = metadata.get(field, "")

        if value is None:
            return set()

        values = str(value).split(",")
        return {self._normalize_text(item) for item in values if item.strip()}

    def _get_score(self, metadata: dict) -> float:
        try:
            return float(metadata.get("Score", 0))
        except (TypeError, ValueError):
            return 0

    def _apply_hybrid_rerank(self, recommendation: dict, reference_metadata: dict | None) -> dict:
        if reference_metadata is None:
            recommendation["AdjustedDistance"] = recommendation["Distance"]
            recommendation["SharedGenres"] = ""
            recommendation["SharedThemes"] = ""
            return recommendation

        reference_genres = self._split_metadata_values(reference_metadata, "Genres")
        candidate_genres = self._split_metadata_values(recommendation, "Genres")
        reference_themes = self._split_metadata_values(reference_metadata, "Themes")
        candidate_themes = self._split_metadata_values(recommendation, "Themes")

        shared_genres = reference_genres.intersection(candidate_genres)
        shared_themes = reference_themes.intersection(candidate_themes)
        score_bonus = (self._get_score(recommendation) / 10) * SCORE_WEIGHT
        genre_bonus = len(shared_genres) * GENRE_MATCH_WEIGHT
        theme_bonus = len(shared_themes) * THEME_MATCH_WEIGHT

        recommendation["AdjustedDistance"] = recommendation["Distance"] - genre_bonus - theme_bonus - score_bonus
        recommendation["SharedGenres"] = ", ".join(sorted(shared_genres))
        recommendation["SharedThemes"] = ", ".join(sorted(shared_themes))
        return recommendation

    def _build_recommendations(
        self,
        anime_results: dict,
        n_results: int,
        min_score: float | None = None,
        genre: str | None = None,
        themes: str | None = None,
        max_distance: float | None = None,
        excluded_ids: set[str] | None = None,
        reference_title: str | None = None,
        reference_metadata: dict | None = None,
    ) -> list[dict]:
        recommendations = []
        metadatas = anime_results["metadatas"][0]
        distances = anime_results["distances"][0]
        ids = anime_results["ids"][0]

        for anime_id, metadata, distance in zip(ids, metadatas, distances):
            if excluded_ids is not None and anime_id in excluded_ids:
                continue

            if reference_title is not None and self._metadata_contains_title(metadata, reference_title):
                continue

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
            recommendation = self._apply_hybrid_rerank(recommendation, reference_metadata)
            recommendations.append(recommendation)

        sort_key = "AdjustedDistance" if reference_metadata is not None else "Distance"
        recommendations = sorted(recommendations, key=lambda recommendation: recommendation[sort_key])

        return recommendations[:n_results]

    def recommend(
        self,
        query: str,
        n_results: int = 5,
        status: str | None = "Finished Airing",
        min_score: float | None = None,
        anime_type: str | None = None,
        genre: str | None = None,
        themes: str | None = None,
        max_distance: float | None = None,
        fetch_results: int = 15,
    ) -> list[dict]:
        excluded_ids = set()
        reference_title = self._extract_reference_title(query)
        matched_reference_title = None
        reference_metadata = None
        print("Reference title:", reference_title)

        if reference_title is not None:
            reference_anime = self._find_reference_anime(reference_title)
            print("Reference anime:", reference_anime["metadata"] if reference_anime else None)

            if reference_anime is not None:
                query_embedding = [reference_anime["embedding"]]
                excluded_ids.add(reference_anime["id"])
                matched_reference_title = reference_title
                reference_metadata = reference_anime["metadata"]
                fetch_results = max(fetch_results, ITEM_TO_ITEM_FETCH_RESULTS) + 1
            else:
                query_embedding = self.embedder.encode([query]).tolist()
        else:
            query_embedding = self.embedder.encode([query]).tolist()

        chromadb_filter = self._build_chromadb_filter(status=status, anime_type=anime_type)

        anime_results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=fetch_results,
            where=chromadb_filter,
        )

        return self._build_recommendations(
            anime_results,
            n_results,
            min_score=min_score,
            genre=genre,
            themes=themes,
            max_distance=max_distance,
            excluded_ids=excluded_ids,
            reference_title=matched_reference_title,
            reference_metadata=reference_metadata,
        )

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
            print(f"   Adjusted distance: {recommendation['AdjustedDistance']:.4f}")

            if recommendation.get("SharedGenres"):
                print(f"   Shared genres: {recommendation['SharedGenres']}")

            if recommendation.get("SharedThemes"):
                print(f"   Shared themes: {recommendation['SharedThemes']}")

    def show_title_search_results(self, matches: list[dict], title: str) -> None:
        print(f"\nTitle search results for:\n{title}")

        if not matches:
            print("\nNo anime found with this title.")
            return

        for index, match in enumerate(matches, start=1):
            metadata = match["metadata"]
            print(f"\n{index}. {metadata.get('title', '')}")
            print(f"   English title: {metadata.get('title_english', '')}")
            print(f"   Japanese title: {metadata.get('title_japanese', '')}")
            print(f"   Synonyms: {metadata.get('title_synonyms', '')}")
            print(f"   Score: {metadata.get('Score', '')}")
            print(f"   Status: {metadata.get('Status', '')}")


if __name__ == "__main__":
    collection_name = "dataset_embeddings"
    model = "sentence-transformers/all-mpnet-base-v2"
    query = "An anime similar to 'Ore dake Level Up na Ken'."

    recommender = AnimeRecommender(collection_name, model)
    recommendations = recommender.recommend(
        query=query,
        n_results=5,
        status="Finished Airing",
        min_score=None,
        max_distance=None,
    )
    recommender.show_recommendations(recommendations, query)
