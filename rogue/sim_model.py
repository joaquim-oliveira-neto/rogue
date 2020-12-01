#!/usr/bin/env python
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity



class SimModel():

    def __init__(self):
        pass

    def create_matrix(soup):
    #create matriz from soup
        count = CountVectorizer()
        return count.fit_transform(soup)

    def get_similarity(matrix):
    #get cosine similarity with two given matrices
        return cosine_similarity(matrix, matrix)

    def get_recommendations(title, cosine_sim, df):

        df2 = df.copy()
        df2 = df2.reset_index()
        indices = pd.Series(df2.index, index=df2['title'])

        # idx Get the index of the movie that matches the title
        idx = indices[title]

        # Sim_scores creates a list of all movies and the cosine similarity related to the movie selected in 'title'
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Sort the movies based on the similarity scores
        # Note: Key is the field that the sort fuction will use to do the sort (position 1 of the tuple)
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the scores of the X most similar movies (position 0 will be the selected movie)
        sim_scores = sim_scores[0:30]

        # Get the movie indices
        movie_indices = [i[0] for i in sim_scores]

        # Get the movie cosine scores:
        cosine_scores = [i[1] for i in sim_scores]

        # Return the top 10 most similar movies
        #return df2['title'].iloc[movie_indices]
        a=pd.DataFrame(df2[['genres','title','actors','director','vote_average','popularity','poster_path']].iloc[movie_indices]).reset_index(drop=True)
        b=pd.DataFrame(cosine_scores, columns=['cosine_score'])

        return   pd.concat([a,b], axis = 1)#.set_index('title')


