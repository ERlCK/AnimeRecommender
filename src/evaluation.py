"""Evaluate recommendation quality with hand-labeled query examples."""

from recommender import AnimeRecommender


EVALUATION_CASES = [
    {
        "query": "An anime about vikings and a history of revenge.",
        "relevant_titles": ["Vinland Saga", "Vinland Saga Season 2", "Chiisana Viking Vickie"],
    },
    {
        "query": "A futuristic cyberpunk police anime about crime and technology.",
        "relevant_titles": [
            "Psycho-Pass",
            "Ghost in the Shell",
            "Koukaku Kidoutai",
            "A.D. Police",
            "Dominion",
        ],
    },
    {
        "query": "A psychological thriller anime with mind games and strategy.",
        "relevant_titles": [
            "Death Note",
            "Monster",
            "Code Geass",
            "Tomodachi Game",
            "Deatte 5-byou de Battle",
            "Pet",
            "Mind Game",
        ],
    },
    {
        "query": "A dark fantasy anime about demons and survival.",
        "relevant_titles": ["Demon Slayer", "Kimetsu no Yaiba", "Berserk", "Claymore", "Dororo"],
    },
    {
        "query": "A sports anime about volleyball and teamwork.",
        "relevant_titles": ["Haikyuu!!", "Haikyuu!! Second Season", "2.43", "Ashita e Attack"],
    },
]


def normalize_title(title: str) -> str:
    return title.strip().lower()


def get_recommendation_titles(recommendation: dict) -> list[str]:
    title_fields = ["title", "title_english", "title_japanese", "title_synonyms"]
    titles = []

    for field in title_fields:
        value = recommendation.get(field)

        if value is None:
            continue

        value = str(value).strip()

        if value:
            titles.append(value)

    return titles


def title_matches(recommendation_titles: list[str], relevant_titles: list[str]) -> bool:
    normalized_recommendation_titles = [normalize_title(title) for title in recommendation_titles]
    normalized_relevant_titles = [normalize_title(title) for title in relevant_titles]

    for recommendation_title in normalized_recommendation_titles:
        for relevant_title in normalized_relevant_titles:
            if relevant_title in recommendation_title or recommendation_title in relevant_title:
                return True

    return False


def find_relevant_rank(recommendations: list[dict], relevant_titles: list[str]) -> int | None:
    for index, recommendation in enumerate(recommendations, start=1):
        recommendation_titles = get_recommendation_titles(recommendation)

        if title_matches(recommendation_titles, relevant_titles):
            return index

    return None


def evaluate_recommender(
    recommender: AnimeRecommender,
    evaluation_cases: list[dict],
    n_results: int = 10,
    fetch_results: int = 30,
) -> dict:
    top_1_hits = 0
    top_5_hits = 0
    top_10_hits = 0
    reciprocal_rank_sum = 0

    for case in evaluation_cases:
        query = case["query"]

        recommendations = recommender.recommend(
            query=query,
            n_results=n_results,
            status="Finished Airing",
            fetch_results=fetch_results,
        )

        relevant_titles = case["relevant_titles"]
        relevant_rank = find_relevant_rank(recommendations, relevant_titles)

        print(f"\nQuery: {query}")
        print(f"Relevant titles: {', '.join(relevant_titles)}")

        if relevant_rank is None:
            print("Result: miss")
        else:
            print(f"Result: hit at rank {relevant_rank}")
            reciprocal_rank_sum += 1 / relevant_rank

            if relevant_rank == 1:
                top_1_hits += 1

            if relevant_rank <= 5:
                top_5_hits += 1

            if relevant_rank <= 10:
                top_10_hits += 1

        for index, recommendation in enumerate(recommendations[:5], start=1):
            print(f"  {index}. {recommendation['title']} ({recommendation['Distance']:.4f})")

    total_cases = len(evaluation_cases)

    return {
        "total_cases": total_cases,
        "top_1_accuracy": top_1_hits / total_cases,
        "top_5_recall": top_5_hits / total_cases,
        "top_10_recall": top_10_hits / total_cases,
        "mrr": reciprocal_rank_sum / total_cases,
    }


def show_metrics(metrics: dict) -> None:
    print("\nEvaluation metrics:")
    print(f"Total cases: {metrics['total_cases']}")
    print(f"Top-1 accuracy: {metrics['top_1_accuracy']:.2%}")
    print(f"Top-5 recall: {metrics['top_5_recall']:.2%}")
    print(f"Top-10 recall: {metrics['top_10_recall']:.2%}")
    print(f"MRR: {metrics['mrr']:.3f}")


if __name__ == "__main__":
    collection_name = "dataset_embeddings"
    model = "sentence-transformers/all-mpnet-base-v2"

    recommender = AnimeRecommender(collection_name, model)
    metrics = evaluate_recommender(recommender, EVALUATION_CASES)
    show_metrics(metrics)
