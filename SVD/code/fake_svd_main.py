# mohammad amin kiani 4003613052

import csv
from typing import List
import numpy as np
from numpy.linalg import svd
from dataclasses import dataclass
from operator import itemgetter
import pandas as pd


@dataclass
class Movies:
    movie_id: int
    title: str
    genres: str


def movies() -> List[Movies]:
    with open("C:\\Users\\Almahdi\\Desktop\\RSP\\SVD\\src\\movies.csv", "r", encoding="utf-8") as file:
        reader = csv.DictReader(file, fieldnames=['movieId', 'title', 'genres'])
        next(reader)  # skip header
        return [Movies(int(row["movieId"]), row["title"], row["genres"]) for row in reader]


@dataclass
class Ratings:
    user_id: int
    movie_id: int
    rating: float


def ratings() -> List[Ratings]:
    with open("C:\\Users\\Almahdi\\Desktop\\RSP\\SVD\\src\\ratings.csv", "r", encoding="utf-8") as file:
        reader = csv.DictReader(file, fieldnames=['userId', 'movieId', 'rating'])
        next(reader)  # skip header
        return [Ratings(int(row["userId"]), int(row["movieId"]), float(row["rating"])) for row in reader]


def main():

    list_of_rates = ratings()
    list_of_movies = movies()
    # --------------------------------------------------------------------

    user_id = int(input("Enter the User_id: "))

    movieDict = {j.movie_id: i for i, j in enumerate(list_of_movies)}

    rank_rows = []

    for each_user in range(1, 611):  # we have 610 user

        row_each = [0.0] * 9742    # we have 9742 movie

        for r in filter(lambda x: x.user_id == each_user, list_of_rates):

            row_each[movieDict[r.movie_id]] = r.rating

        rank_rows.append(row_each)

    # def ratings_matrix(num_users, num_items):
    # data = []
    # for i in range(num_users):
    # user = [np.random.randint(2) for _ in range(num_items)]
    # data.append(user)
    # matrix = pd.DataFrame(data)
    # matrix.index = ["User_id " + str(i) for i in range(num_users)]
    # matrix.columns = ["Movie_id" + str(i) for i in range(num_items)]
    # return matrix

    matrix = np.array(rank_rows)

    _, S, vT = svd(matrix)

    user_rowfield = vT[user_id]

    user_rate = list(filter(lambda x: x.user_id == user_id, list_of_rates))

    sorterid = []

    for i, j in enumerate(vT):

        if not any(map(lambda x: movieDict[x.movie_id] == i and x.rating != 0.0, user_rate)):

            norm = np.linalg.norm(j) * np.linalg.norm(user_rowfield)

            cosine_sim_point = np.dot(j, user_rowfield) / norm if norm != 0 else 0.0

            sorterid.append((i, cosine_sim_point))

    sorterid.sort(key=itemgetter(1))

    recommended_films_to_user_i([list_of_movies[i] for i, _ in sorterid])

    print("Done! look at C:\\Users\\Almahdi\\Desktop\\RSP\\SVD\\code\\recommended_films.csv")


def recommended_films_to_user_i(movies_list):
    with open('recommended_films.csv', mode='w', newline='', encoding='utf-8') as file:

        records = ['rec_movieId', 'title', 'genres']

        fill_records = csv.DictWriter(file, fieldnames=records)

        fill_records.writeheader()

        for i in movies_list:

            fill_records.writerow({'rec_movieId': i.movie_id, 'title': i.title, 'genres': i.genres})


main()
