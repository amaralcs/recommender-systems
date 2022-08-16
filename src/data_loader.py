import numpy as np
import pandas as pd
from urllib.request import urlretrieve
import zipfile

USERS_COLS = ["user_id", "age", "sex", "occupation", "zip_code"]
RATINGS_COLS = ["user_id", "movie_id", "rating", "unix_timestamp"]
GENRE_COLS = [
    "genre_unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]
MOVIES_COLS = [
    "movie_id",
    "title",
    "release_date",
    "video_release_date",
    "imdb_url",
] + GENRE_COLS


def download():
    print("Downloading movielens data...")

    urlretrieve(
        "http://files.grouplens.org/datasets/movielens/ml-100k.zip",
        "data/movielens.zip",
    )
    zip_ref = zipfile.ZipFile("data/movielens.zip", "r")
    zip_ref.extractall("data/")
    print("Done. Dataset contains:")
    print(zip_ref.read("ml-100k/u.info"))


def prep():
    def mark_genres(movies, genres):
        # Since some movies can belong to more than one genre, we create different
        # 'genre' columns as follows:
        # - all_genres: all the active genres of the movie.
        # - genre: randomly sampled from the active genres.
        def get_random_genre(gs):
            active = [genre for genre, g in zip(genres, gs) if g == 1]
            if len(active) == 0:
                return "Other"
            return np.random.choice(active)

        def get_all_genres(gs):
            active = [genre for genre, g in zip(genres, gs) if g == 1]
            if len(active) == 0:
                return "Other"
            return "-".join(active)

        movies["genre"] = [
            get_random_genre(gs) for gs in zip(*[movies[genre] for genre in genres])
        ]
        movies["all_genres"] = [
            get_all_genres(gs) for gs in zip(*[movies[genre] for genre in genres])
        ]

    # Load each data set (users, movies, and ratings).
    users = pd.read_csv(
        "data/ml-100k/u.user", sep="|", names=USERS_COLS, encoding="latin-1"
    )
    ratings = pd.read_csv(
        "data/ml-100k/u.data", sep="\t", names=RATINGS_COLS, encoding="latin-1"
    )

    # The movies file contains a binary feature for each genre.
    movies = pd.read_csv(
        "data/ml-100k/u.item", sep="|", names=MOVIES_COLS, encoding="latin-1"
    )

    # Since the ids start at 1, we shift them to start at 0.
    users["user_id"] = (users["user_id"] - 1).astype(str)
    movies["movie_id"] = (movies["movie_id"] - 1).astype(str)
    movies["year"] = movies["release_date"].apply(lambda x: str(x).split("-")[-1])
    ratings["movie_id"] = (ratings["movie_id"] - 1).astype(str)
    ratings["user_id"] = (ratings["user_id"] - 1).apply(lambda x: str(x - 1))
    ratings["rating"] = (ratings["rating"]).astype(float)

    # Compute the number of movies to which a genre is assigned.
    genre_occurences = movies[GENRE_COLS].sum().to_dict()
    mark_genres(movies, GENRE_COLS)

    # Create one merged DataFrame containing all the movielens data.
    movielens = ratings.merge(movies, on="movie_id").merge(users, on="user_id")

    ratings.to_csv("data/prepared/ratings.csv", index=False)
    movies.to_csv("data/prepared/movies.csv", index=False)
    users.to_csv("data/prepared/users.csv", index=False)
    movielens.to_csv("data/prepared/movielens.csv", index=False)


if __name__ == "__main__":
    download()
    prep()
