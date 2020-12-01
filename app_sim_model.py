import streamlit as st
import numpy as np
import pandas as pd
from rogue.sim_model import SimModel

st.title('Rogue Inteligent Movie Recomendation Systems - Content Based')

@st.cache(suppress_st_warning=True)
def get_dfs():
    df = pd.read_csv('raw_data/soup/soup.csv')
    return df

df = get_dfs()

@st.cache(suppress_st_warning=True)
def get_sim(soup):
    matrix = SimModel.create_matrix(soup)
    similarity = SimModel.get_similarity(matrix)
    return similarity

#Create our matrix to get recommendation

similarity = get_sim(df['soup2'])

filmnames = df['title'].values.tolist()

options = st.multiselect('Which movie would you like to compare?',filmnames)

#st.write('You selected:', options)


if options is not None:
    for i in options:


        recommendation = SimModel.get_recommendations(i,similarity, df)
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





