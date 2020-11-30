#!/usr/bin/env python
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity



def get_most_rated_movies_df(ratings_df, top_number):
    most_rated_movies = ratings_df.movieId.value_counts().head(top_number).index
    return ratings_df[ratings_df.movieId.isin(most_rated_movies)].copy()

def get_users_with_min_num_of_ratings_df(ratings_df, min_num_of_ratings):
    rates_per_user = ratings.userId.value_counts()
    bolean_users_with_min_num_of_ratings = ratings_df.userId.value_counts() >= min_num_of_ratings
    users_with_min_num_of_ratings = rates_per_user[bolean_users_with_min_num_of_ratings].index
    return ratings_df[ratings_df.userId.isin(users_with_min_num_of_ratings)].copy()

def standardize(row):
    new_row = (row - row.mean()) / (row.max() - row.min())
    return new_row

def ratings_top_std(ratings_df, top_number, min_num_of_ratings):
    ratings = get_most_rated_movies_df(ratings_df, top_number)
    ratings = get_users_with_min_num_of_ratings_df(ratings, min_num_of_ratings)
    ratings = ratings.pivot(index='userId', columns='title', values='rating')
    ratings = ratings.fillna(0)
    ratings = ratings.apply(standardize)
    return ratings

def similarity(ratings_df):
    item_similarity = cosine_similarity(ratings_df.T)
    item_similarity_df = pd.DataFrame(item_similarity, index=ratings_df.columns, columns=ratings_df.columns)
    return item_similarity_df

def get_similar_movies(movie_name,user_rating):
    similar_score = item_similarity_df[movie_name]*(user_rating-2.5)
    similar_score = similar_score.sort_values(ascending=False)
    return similar_score


