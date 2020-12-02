import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from rogue.lib import get_recommendations, get_users_ratings_df
from rogue.sim_model import SimModel

# st.image('streamlit-images/Logo-ong.png', width=300)
# st.title('Inteligent Movie Recomendation Systems')


@st.cache
def get_users_ratings_df():
    df = pd.read_csv('raw_data/streamlit-data/ratings_lite.csv')
    df = df.pivot(
        index='userId', columns='title', values='rating')
    return df

# @st.cache(suppress_st_warning=True)
# def get_dfs():
#     df_content = pd.read_csv('raw_data/soup/soup.csv')
#     df_rating = pd.read_csv('raw_data/streamlit-data/ratings_top_users_top.csv')
#     return df_content , df_rating


df_content , df_rating = get_dfs()

@st.cache(suppress_st_warning=True)
def get_sim(soup):
    matrix = SimModel.create_matrix(soup)
    similarity = SimModel.get_similarity(matrix)
    return similarity


st.image('streamlit-images/header.png')
st.image('streamlit-images/getrecom.png')
st.sidebar.selectbox(['Content','Rating'])

with col1:
    st.image(f"streamlit-images/contet.png", use_column_width=True)

    if st.sidebar.selectbox('Content'):
        similarity = get_sim(df_content['soup2'])

        filmnames = df_content['title'].values.tolist()

        options = st.multiselect('Which movie would you like to compare?',filmnames)

        #st.write('You selected:', options)

        if options is not None:
            for i in options:


                recommendation = SimModel.get_recommendations(i,similarity, df_content)
                #st.dataframe(recommendation)
                st.markdown("""## This is the film you chose""")
                st.header(recommendation.iloc[0].title)
                st.image(f"https://image.tmdb.org/t/p/original/{recommendation.iloc[0].poster_path}", width=170, use_column_width=False)

                options_sort = st.multiselect('Sort by',['popularity', 'vote_average'])
                if options_sort is not None:
                    for i in options_sort:
                        recommendation_2 = recommendation.copy().sort_values(by=i, ascending=False).drop([0])

                        st.markdown("""## This are some films you migth like""")
                        col1, col2, col3, col4 = st.beta_columns(4)
                        with col1:
                            caption=recommendation_2.iloc[1].title
                            st.image(f"https://image.tmdb.org/t/p/original/{recommendation_2.iloc[1].poster_path}", use_column_width=True , caption=caption)
                        with col2:
                            caption=recommendation_2.iloc[2].title
                            st.image(f"https://image.tmdb.org/t/p/original/{recommendation_2.iloc[2].poster_path}", use_column_width=True, caption=caption)
                        with col3:
                            caption=recommendation_2.iloc[3].title
                            st.image(f"https://image.tmdb.org/t/p/original/{recommendation_2.iloc[3].poster_path}", use_column_width=True, caption=caption)
                        with col4:
                            caption=recommendation_2.iloc[4].title
                            st.image(f"https://image.tmdb.org/t/p/original/{recommendation_2.iloc[4].poster_path}", use_column_width=True, caption=caption)
                        col1, col2, col3, col4 = st.beta_columns(4)
                        with col1:
                            caption=recommendation_2.iloc[5].title
                            st.image(f"https://image.tmdb.org/t/p/original/{recommendation_2.iloc[5].poster_path}", use_column_width=True , caption=caption)
                        with col2:
                            caption=recommendation_2.iloc[6].title
                            st.image(f"https://image.tmdb.org/t/p/original/{recommendation_2.iloc[6].poster_path}", use_column_width=True, caption=caption)
                        with col3:
                            caption=recommendation_2.iloc[7].title
                            st.image(f"https://image.tmdb.org/t/p/original/{recommendation_2.iloc[7].poster_path}", use_column_width=True, caption=caption)
                        with col4:
                            caption=recommendation_2.iloc[8].title
                            st.image(f"https://image.tmdb.org/t/p/original/{recommendation_2.iloc[8].poster_path}", use_column_width=True, caption=caption)


with col2:
    st.image(f"streamlit-images/rating.png", use_column_width=True)
    if st.st.sidebar.selectbox('Rating'):
        recommended_movies = get_recommendations(df_rating, movies_to_rate, user_ratings)
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
            recommended_movies = get_recommendations(df_rating, movies_to_rate, user_ratings)

            st.write(recommended_movies.iloc[1:])


# st.title('Rogue Inteligent Movie Recomendation systems')


# @st.cache
# def get_users_ratings_df():
#     df = pd.read_csv('raw_data/streamlit-data/ratings_top_users_top.csv')
#     return df

# df = get_users_ratings_df()



# movies_to_rate = [
#     'Batman (1989)',
#     'Memento (2000)',
#     'Matrix, The (1999)',
#     'Titanic (1997)',
#     'Inception (2010)',
#     'American Beauty (1999)',
#     'Twelve Monkeys (a.k.a. 12 Monkeys) (1995)',
#     'Lord of the Rings: The Fellowship of the Ring, The (2001)',
#     'Monty Python and the Holy Grail (1975)'
# ]

# key = 0
# user_ratings = []

# for title in movies_to_rate:
#     key += 1
#     st.write(title)
#     user_ratings.append(st.number_input('Give a Rating',
#                                    key=key, min_value=1, max_value=5))

# if st.button('SUBMIT'):
#     recommended_movies = get_recommendations(df_rating, movies_to_rate, user_ratings)

#     st.write(recommended_movies.iloc[1:])
