import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from rogue.lib import get_recommendations, get_users_ratings_df

st.title('Rogue Inteligent Movie Recomendation systems')


@st.cache
def get_users_ratings_df():
    df = pd.read_csv('raw_data/streamlit-data/ratings_lite.csv')
    df = df.pivot(
        index='userId', columns='title', values='rating')
    return df

df = get_users_ratings_df()

movies_to_rate = [
    'Batman (1989)',
    'Memento (2000)',
    'Matrix, The (1999)',
    'Titanic (1997)',
    'Inception (2010)',
    'American Beauty (1999)',
    'Twelve Monkeys (a.k.a. 12 Monkeys) (1995)',
    'Lord of the Rings: The Fellowship of the Ring, The (2001)',
    'Monty Python and the Holy Grail (1975)'
]

key = 0
user_ratings = []

for title in movies_to_rate:
    key += 1
    st.write(title)
    user_ratings.append(st.number_input('Give a Rating',
                                   key=key, min_value=1, max_value=5))

if st.button('SUBMIT'):
    recommended_movies = get_recommendations(df, movies_to_rate, user_ratings)

    st.write(recommended_movies.iloc[1:])
