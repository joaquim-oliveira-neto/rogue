import streamlit as st
from streamlit.hashing import _CodeHasher
from google.cloud import storage

try:
    # Before Streamlit 0.65
    from streamlit.ReportThread import get_report_ctx
    from streamlit.server.Server import Server
except ModuleNotFoundError:
    # After Streamlit 0.65
    from streamlit.report_thread import get_report_ctx
    from streamlit.server.server import Server

import numpy as np
import pandas as pd
from rogue.lib import get_recommendations, get_users_ratings_df
from rogue.sim_model import SimModel
import time

@st.cache(suppress_st_warning=True)
def get_dfs():
    BUCKET_NAME = 'rogue-data'
    BUCKET_PATH = f"gs://{BUCKET_NAME}/raw_data"

    # df_content = pd.read_csv('raw_data/streamlit-data/soup_lite.csv')
    # df_rating = pd.read_csv('raw_data/streamlit-data/ratings_lite.csv')
    # df_rating_c = pd.read_csv('raw_data/streamlit-data/ratings_lite_content.csv')

    client = storage.Client()
    df_content = pd.read_csv(f'{BUCKET_PATH}/soup_lite.csv')
    df_rating = pd.read_csv(f'{BUCKET_PATH}/ratings_lite.csv')
    df_rating_c = pd.read_csv(f'{BUCKET_PATH}/ratings_lite_content.csv')

    df_rating = df_rating.pivot(
        index='userId', columns='title', values='rating')
    df_rating_c = pd.read_csv('raw_data/streamlit-data/ratings_lite_content.csv')
    return df_content , df_rating ,df_rating_c

df_content , df_rating, df_rating_c = get_dfs()

@st.cache(suppress_st_warning=True)
def get_sim(soup):
    matrix = SimModel.create_matrix(soup)
    similarity = SimModel.get_similarity(matrix)

    return model

def main():
    state = _get_state()
    pages = {
        "Home" : page_home,
        "Content": page_content,
        "Rating": page_rating,
    }

    st.sidebar.title("Recomendation Type")
    page = st.sidebar.radio("Select", tuple(pages.keys()))

    # Display the selected page with the session state
    pages[page](state)

    # Mandatory to avoid rollbacks with widgets, must be called at the end of your app
    state.sync()

def page_home(state):
    st.image('streamlit-images/header.png')
    st.image('streamlit-images/getrecom.png')
    col1, col2 = st.beta_columns(2)
    with col1:
        st.image(f"streamlit-images/content.png", use_column_width=True)
    with col2:
        st.image(f"streamlit-images/rating.png", use_column_width=True)


def page_content(state):
    st.image('streamlit-images/head-cont.png')
    matrix = get_sim(df_content['soup2'])

    filmnames = df_content['title'].values.tolist()

    options = st.multiselect('Which movie would you like to compare?',filmnames)

    if options is not None:
        for i in options:
            recommendation = SimModel.get_recommendations(i,similarity, df_content)
            st.markdown("""## This is the film you chose""")
            st.header(recommendation.iloc[0].title)
            st.image(f"https://image.tmdb.org/t/p/original/{recommendation.iloc[0].poster_path}", width=170, use_column_width=False)
            st.markdown('''## Sorty by:''')
            col1, col2 = st.beta_columns(2)
            with col1:
                st.markdown('''
                    ### Popularity
                    #### a number given by:
                    - *Number of votes for the day*
                    - *Number of views for the day*
                    - *Number of users who marked it as a "favourite" for the day*
                    - *Number of users who added it to their "watchlist" for the day*
                    - *Release date*
                    - *Number of total votes*
                    - *Previous days score*
                    ''')
            with col2:
                st.markdown('''
                    ### Vote Average
                    #### a number given by:
                    - *The sum of all votes given to a film divided by the number of votes*
                    ''')
            options_sort = st.multiselect('Select',['popularity', 'vote_average'])
            if options_sort is not None:
                for i in options_sort:
                    recommendation_2 = recommendation.copy().sort_values(by=i, ascending=False).drop([0])

                    st.markdown("""## This are some films you might like""")
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



def page_rating(state):
    st.image('streamlit-images/head-rating.png')

    @st.cache(suppress_st_warning=True)
    def get_random_subset(df_content):
        random_df = df_content[0:20].sort_values(by='total_ratings', ascending=False)
        random_df = df_content.sample(10)
        return random_df


    random_df = get_random_subset(df_rating_c)
    movies_to_rate = random_df['title'].values.tolist()


    key = 0
    movid = 0
    user_ratings = []
    col = 0


    ####### RATE FILMS #######

    cols1 = st.beta_columns(4)
    for movie in movies_to_rate:
        while key <= 3:
            with cols1[col]:
                poster = f"https://image.tmdb.org/t/p/original/{random_df.iloc[movid].poster_path}"
                caption=random_df.iloc[movid].title
                cols1[col].image(poster, width=150,caption=caption)
                if  st.button('Never Seen' , key=key):
                    pass
                else:
                    vote = cols1[col].number_input('Give a Rating', key =key, min_value=1,max_value=5)
                    user_ratings.append(vote)
                # vote = cols1[col].number_input('Give a Rating', key =key, min_value=1,max_value=5)
                # user_ratings.append(vote)
                col += 1
                key += 1
                movid += 1


    key = 4
    cols2 = st.beta_columns(4)
    col = 0
    for movie in movies_to_rate:
        while key <= 7:
            with cols2[col]:
                poster = f"https://image.tmdb.org/t/p/original/{random_df.iloc[movid].poster_path}"
                caption=random_df.iloc[movid].title
                cols2[col].image(poster, width=150,caption=caption)
                if  st.button('Never Seen' , key=key):
                    pass
                else:
                    vote = cols2[col].number_input('Give a Rating', key =key, min_value=1,max_value=5)
                    user_ratings.append(vote)
                # vote = cols2[col].number_input('Give a Rating', key =key, min_value=1,max_value=5)
                # user_ratings.append(vote)
                col += 1
                key += 1
                movid += 1

    key = 8
    cols3 = st.beta_columns(4)
    col = 1
    for movie in movies_to_rate:
        while key <= 9:
            with cols3[col]:
                poster = f"https://image.tmdb.org/t/p/original/{random_df.iloc[movid].poster_path}"
                caption=random_df.iloc[movid].title
                cols3[col].image(poster, width=150,caption=caption)
                if  st.button('Never Seen' , key=key):
                    pass
                else:
                    vote = cols3[col].number_input('Give a Rating', key =key, min_value=1,max_value=5)
                    user_ratings.append(vote)
                # vote = cols3[col].number_input('Give a Rating', key =key, min_value=1,max_value=5)
                # user_ratings.append(vote)
                col += 1
                key += 1
                movid += 1

    ####### PREDICT #######




    if st.button('SUBMIT'):


        my_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.1)
            my_bar.progress(percent_complete + 1)
        recommended_movies = get_recommendations(
            df_rating, movies_to_rate, user_ratings)
        recommended_movies = recommended_movies.iloc[1:11].reset_index()
        recommended_movies = recommended_movies.merge(df_rating_c, on='title')

        # st.write(recommended_movies)

        st.markdown("""## This are some films you might like""")
        col1, col2, col3, col4, col5 = st.beta_columns(5)
        with col1:
            caption=recommended_movies.iloc[0].title
            st.image(f"https://image.tmdb.org/t/p/original/{recommended_movies.iloc[0].poster_path}", use_column_width=True , caption=caption)
        with col2:
            caption=recommended_movies.iloc[1].title
            st.image(f"https://image.tmdb.org/t/p/original/{recommended_movies.iloc[1].poster_path}", use_column_width=True, caption=caption)
        with col3:
            caption=recommended_movies.iloc[2].title
            st.image(f"https://image.tmdb.org/t/p/original/{recommended_movies.iloc[2].poster_path}", use_column_width=True, caption=caption)
        with col4:
            caption=recommended_movies.iloc[3].title
            st.image(f"https://image.tmdb.org/t/p/original/{recommended_movies.iloc[3].poster_path}", use_column_width=True, caption=caption)
        with col5:
            caption=recommended_movies.iloc[4].title
            st.image(f"https://image.tmdb.org/t/p/original/{recommended_movies.iloc[4].poster_path}", use_column_width=True , caption=caption)
        col1, col2, col3, col4, col5 = st.beta_columns(5)
        with col1:
            caption=recommended_movies.iloc[5].title
            st.image(f"https://image.tmdb.org/t/p/original/{recommended_movies.iloc[5].poster_path}", use_column_width=True , caption=caption)
        with col2:
            caption=recommended_movies.iloc[6].title
            st.image(f"https://image.tmdb.org/t/p/original/{recommended_movies.iloc[6].poster_path}", use_column_width=True, caption=caption)
        with col3:
            caption=recommended_movies.iloc[7].title
            st.image(f"https://image.tmdb.org/t/p/original/{recommended_movies.iloc[7].poster_path}", use_column_width=True, caption=caption)
        with col4:
            caption=recommended_movies.iloc[8].title
            st.image(f"https://image.tmdb.org/t/p/original/{recommended_movies.iloc[8].poster_path}", use_column_width=True, caption=caption)
        with col5:
            caption=recommended_movies.iloc[9].title
            st.image(f"https://image.tmdb.org/t/p/original/{recommended_movies.iloc[9].poster_path}", use_column_width=True, caption=caption)



class _SessionState:

    def __init__(self, session, hash_funcs):
        """Initialize SessionState instance."""
        self.__dict__["_state"] = {
            "data": {},
            "hash": None,
            "hasher": _CodeHasher(hash_funcs),
            "is_rerun": False,
            "session": session,
        }

    def clear(self):
        """Clear session state and request a rerun."""
        self._state["data"].clear()
        self._state["session"].request_rerun()

    def sync(self):
        """Rerun the app with all state values up to date from the beginning to fix rollbacks."""

        # Ensure to rerun only once to avoid infinite loops
        # caused by a constantly changing state value at each run.
        #
        # Example: state.value += 1
        if self._state["is_rerun"]:
            self._state["is_rerun"] = False

        elif self._state["hash"] is not None:
            if self._state["hash"] != self._state["hasher"].to_bytes(self._state["data"], None):
                self._state["is_rerun"] = True
                self._state["session"].request_rerun()

        self._state["hash"] = self._state["hasher"].to_bytes(self._state["data"], None)


def _get_session():
    session_id = get_report_ctx().session_id
    session_info = Server.get_current()._get_session_info(session_id)

    if session_info is None:
        raise RuntimeError("Couldn't get your Streamlit Session object.")

    return session_info.session


def _get_state(hash_funcs=None):
    session = _get_session()

    if not hasattr(session, "_custom_session_state"):
        session._custom_session_state = _SessionState(session, hash_funcs)

    return session._custom_session_state


if __name__ == "__main__":
    main()
