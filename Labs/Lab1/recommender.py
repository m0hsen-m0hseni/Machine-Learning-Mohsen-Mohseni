import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity


def load_data():
    movies = pd.read_csv("../../Data/movies.csv")
    tags = pd.read_csv("../../Data/tags.csv")
    links = pd.read_csv("../../Data/links.csv")
    return movies, tags, links


def prepare_data(sample_size=5000):
    movies, tags, links = load_data()

    movies_sample = movies.sample(sample_size).reset_index(drop=True)

    tags_subset = tags[tags["movieId"].isin(movies_sample["movieId"])].copy()

    tags_grouped = (
        tags_subset.groupby("movieId")["tag"]
        .apply(lambda x: " ".join(x.dropna().astype(str)))
        .reset_index()
    )

    movies_content = movies_sample.merge(
        tags_grouped, on="movieId", how="left")
    movies_content = movies_content.merge(links, on="movieId", how="left")

    movies_content["tag"] = movies_content["tag"].fillna("")
    movies_content["genres_text"] = movies_content["genres"].str.replace(
        "|", " ", regex=False)
    movies_content["content"] = movies_content["genres_text"] + \
        " " + movies_content["tag"]

    return movies_content


def build_model(movies_content):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies_content["content"])

    knn_model = NearestNeighbors(metric="cosine", algorithm="brute")
    knn_model.fit(tfidf_matrix)

    return tfidf, tfidf_matrix, knn_model


def build_genre_similarity(movies_df):
    movies_df = movies_df.copy()
    movies_df["genres_list"] = movies_df["genres"].apply(
        lambda x: x.split("|"))

    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(movies_df["genres_list"])

    return cosine_similarity(genre_matrix)


def recommend_by_knn(movie_title, movies_df, model, matrix, top_n=5):
    matches = movies_df[movies_df["title"] == movie_title]

    if matches.empty:
        return pd.DataFrame(columns=["title", "genres", "tag"])

    movie_idx = matches.index[0]

    distances, indices = model.kneighbors(
        matrix[movie_idx], n_neighbors=top_n + 1)
    movie_indices = indices.flatten()[1:]

    result = movies_df.iloc[movie_indices][["title", "genres", "tag"]].copy()
    return result.reset_index(drop=True)


def recommend_by_content(movie_title, movies_df, similarity_matrix, top_n=5):
    matches = movies_df[movies_df["title"] == movie_title]

    if matches.empty:
        return pd.DataFrame(columns=["title", "genres", "tag"])

    movie_idx = matches.index[0]

    sim_score = list(enumerate(similarity_matrix[movie_idx]))
    sim_score = sorted(sim_score, key=lambda x: x[1], reverse=True)
    sim_score = sim_score[1:top_n + 1]

    movie_indices = [i[0] for i in sim_score]

    result = movies_df.iloc[movie_indices][["title", "genres", "tag"]].copy()
    return result.reset_index(drop=True)


def recommend_by_genre(movie_title, movies_df, similarity_matrix, top_n=5):
    matches = movies_df[movies_df["title"] == movie_title]

    if matches.empty:
        return pd.DataFrame(columns=["title", "genres"])

    movie_idx = matches.index[0]

    sim_score = list(enumerate(similarity_matrix[movie_idx]))
    sim_score = sorted(sim_score, key=lambda x: x[1], reverse=True)
    sim_score = sim_score[1:top_n + 1]

    movie_indices = [i[0] for i in sim_score]

    result = movies_df.iloc[movie_indices][["title", "genres"]].copy()
    return result.reset_index(drop=True)


def extract_year_from_title(title):
    if not isinstance(title, str):
        return "Unknown"

    if "(" in title and ")" in title:
        possible_year = title.split("(")[-1].replace(")", "").strip()
        if possible_year.isdigit():
            return possible_year

    return "Unknown"


def build_imdb_url(imdb_id):
    if pd.isna(imdb_id):
        return None
    try:
        return f"https://www.imdb.com/title/tt{int(imdb_id):07d}/"
    except Exception:
        return None


def build_tmdb_url(tmdb_id):
    if pd.isna(tmdb_id):
        return None
    try:
        return f"https://www.themoviedb.org/movie/{int(tmdb_id)}"
    except Exception:
        return None


def get_movie_details(movie_title):
    row = movies_content[movies_content["title"] == movie_title]

    if row.empty:
        return {
            "title": movie_title,
            "genres": "Not found",
            "tags": "Not found",
            "year": "Unknown",
            "imdb_url": None,
            "tmdb_url": None,
            "imdb_id": None,
            "tmdb_id": None
        }

    row = row.iloc[0]

    genres = row["genres"] if pd.notna(row["genres"]) else "Unknown"
    tags = row["tag"] if pd.notna(row["tag"]) and str(
        row["tag"]).strip() else "No tags available"

    return {
        "title": row["title"],
        "genres": genres,
        "tags": tags,
        "year": extract_year_from_title(row["title"]),
        "imdb_url": build_imdb_url(row.get("imdbId")),
        "tmdb_url": build_tmdb_url(row.get("tmdbId")),
        "imdb_id": row.get("imdbId"),
        "tmdb_id": row.get("tmdbId")
    }


movies_content = prepare_data()
movies_genre = movies_content[["movieId", "title", "genres"]].copy()
tfidf, tfidf_matrix, knn_model = build_model(movies_content)
content_similarity = cosine_similarity(tfidf_matrix)
genre_similarity = build_genre_similarity(movies_genre)
movie_titles = sorted(movies_content["title"].dropna().unique())
