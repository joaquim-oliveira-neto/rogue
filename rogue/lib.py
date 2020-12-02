# -*- coding: UTF-8 -*-
# Copyright (C) 2020 MDM
import numpy as np
import pandas as pd

if __name__ == '__main__':
    print("Hello World")


def get_users_ratings_df():
    df = pd.read_csv('raw_data/streamlit-data/ratings_lite.csv')

def make_user_ratings_df(movie_titles, ratings, users_ratings_df, zeros=True, std=True):
    number_of_movies = users_ratings_df.shape[1]
    if std:
        ratings_ = ratings - np.mean(ratings)
    else:
        ratings_ = ratings
    if zeros:
        new_user = pd.DataFrame(
            [np.zeros(number_of_movies)], columns=users_ratings_df.columns)
    else:
        new_user = pd.DataFrame(
            np.nan, index=[0], columns=users_ratings_df.columns)
    for movie_title, rating in zip(movie_titles, ratings_):
        new_user[movie_title] = rating

    return new_user


def get_similar_users(user_similarities_df, number_of_similar_users):
    return user_similarities_df.sort_values(by='similarity', ascending=False).head(number_of_similar_users).index

def get_movies_from_similar_users(users_ratings_df, similar_users, user_rated_movies, not_seen=True):
    if not_seen:
        movies_from_similar_users = users_ratings_df.loc[similar_users].drop(
            columns=user_rated_movies)
    else:
        movies_from_similar_users = users_ratings_df.loc[similar_users]

    movies_from_similar_users =  movies_from_similar_users.mean(
        axis=0, skipna=True).sort_values(ascending=False)

    return pd.DataFrame(movies_from_similar_users, columns=["Predicted Rating"])

def get_recommendations(df, user_movies, user_ratings):
    user_ratings_df = make_user_ratings_df(
        user_movies,
        user_ratings,
        df,
        zeros=False,
        std=False
    )
    user_similarities = df.corrwith(
        user_ratings_df.iloc[0], axis=1, drop=False, method='pearson')

    user_similarities_df = pd.DataFrame(
        user_similarities, index=df.index, columns=['similarity'])

    similar_users = get_similar_users(user_similarities_df, 30)

    recommended_movies = get_movies_from_similar_users(
        df, similar_users, user_movies, not_seen=True)

    return recommended_movies


