import streamlit as st
from streamlit.hashing import _CodeHasher

try:
    # Before Streamlit 0.65
    from streamlit.ReportThread import get_report_ctx
    from streamlit.server.Server import Server
except ModuleNotFoundError:
    # After Streamlit 0.65
    from streamlit.report_thread import get_report_ctx
    from streamlit.server.server import Server

# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
from rogue.lib import get_recommendations, get_users_ratings_df
from rogue.sim_model import SimModel

# st.image('streamlit-images/Logo-ong.png', width=300)
# st.title('Inteligent Movie Recomendation Systems')


@st.cache(suppress_st_warning=True)
def get_dfs():
    df_content = pd.read_csv('raw_data/sim-model/soup.csv')
    df_rating = pd.read_csv('raw_data/streamlit-data/ratings_lite.csv')
    df_rating = df_rating.pivot(
        index='userId', columns='title', values='rating')

    return df_content , df_rating

df_content , df_rating = get_dfs()

@st.cache(suppress_st_warning=True)
def get_sim(soup):
    matrix = SimModel.create_matrix(soup)
    similarity = SimModel.get_similarity(matrix)
    return similarity


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

    # display_state_values(state)


def page_rating(state):
    st.image('streamlit-images/head-rating.png')

    @st.cache
    def get_random_subset(df_content):
        random_df = df_content[0:50].sort_values(by='vote_average', ascending=False)
        random_df = df_content.sample(10)
        return random_df


    random_df = get_random_subset(df_content)
    movies_to_rate = random_df['title'].values.tolist()
    ###### JORGE #######


    key = 0
    movid = 0
    user_ratings = []
    col = 0
    movies = []


    # col1, col2, col3, col4 = st.beta_columns(4)
    # with col1:
    #     poster = f"https://image.tmdb.org/t/p/original/{random_df.iloc[movid].poster_path}"
    #     caption=random_df.iloc[movid].title
    #     movies.append(caption)
    #     st.image(poster, width=150,caption=caption)
    #     if  st.button('Never Seen' , key=key):
    #         vote = 0
    #     else:
    #         vote = st.number_input('Give a Rating', key =key, min_value=1,max_value=5)
    #     user_ratings.append(vote)
    #     col += 1
    #     key += 1
    #     movid += 1
    # with col2:
    #     poster = f"https://image.tmdb.org/t/p/original/{random_df.iloc[movid].poster_path}"
    #     caption=random_df.iloc[movid].title
    #     movies.append(caption)
    #     st.image(poster, width=150,caption=caption)
    #     if  st.button('Never Seen' , key=key):
    #         vote = 0
    #     else:
    #         vote = st.number_input('Give a Rating', key =key, min_value=1,max_value=5)
    #     user_ratings.append(vote)
    #     col += 1
    #     key += 1
    #     movid += 1
    # with col3:
    #     poster = f"https://image.tmdb.org/t/p/original/{random_df.iloc[movid].poster_path}"
    #     caption=random_df.iloc[movid].title
    #     movies.append(caption)
    #     st.image(poster, width=150,caption=caption)
    #     if  st.button('Never Seen' , key=key):
    #         vote = 0
    #     else:
    #         vote = st.number_input('Give a Rating', key =key, min_value=1,max_value=5)
    #     user_ratings.append(vote)
    #     col += 1
    #     key += 1
    #     movid += 1
    # with col4:
    #     poster = f"https://image.tmdb.org/t/p/original/{random_df.iloc[movid].poster_path}"
    #     caption=random_df.iloc[movid].title
    #     movies.append(caption)
    #     st.image(poster, width=150,caption=caption)
    #     if  st.button('Never Seen' , key=key):
    #         vote = 0
    #     else:
    #         vote = st.number_input('Give a Rating', key =key, min_value=1,max_value=5)
    #     user_ratings.append(vote)
    #     col += 1
    #     key += 1
    #     movid += 1

    ######## JORGE 2 ########
    cols1 = st.beta_columns(4)
    for movie in movies_to_rate:
        while col <= 3:
            with cols1[col]:
                poster = f"https://image.tmdb.org/t/p/original/{random_df.iloc[movid].poster_path}"
                caption=random_df.iloc[movid].title
                movies.append(caption)
                cols1[key].image(poster, width=150,caption=caption)
                if  st.button('Never Seen' , key=key):
                    vote = 0
                else:
                    vote = cols1[key].number_input('Give a Rating', key =key, min_value=1,max_value=5)
                user_ratings.append(vote)
                col += 1
                key += 1
                movid += 1


    key = 4
    cols2 = st.beta_columns(4)
    col = 0
    for movie in movies_to_rate:
        while key <= 7:
            poster = f"https://image.tmdb.org/t/p/original/{random_df.iloc[movid].poster_path}"
            caption=random_df.iloc[movid].title
            cols2[col].image(poster, width=150,caption=caption)
            movies.append(caption)
            user_ratings.append(cols2[col].number_input('Give a Rating', key =key, min_value=1,max_value=5))
            col += 1
            key += 1
            movid += 1

    #### RENATIN #####
    # key = 0
    # ratings = []
    # movies = []
    # cols = st.beta_columns(5)
    # for movie in movies_to_rate:
    #     poster = f"https://image.tmdb.org/t/p/original/{random_df.iloc[key].poster_path}"
    #     caption=random_df.iloc[key].title
    #     cols[key].image(poster, width=150,caption=caption)
    #     movies.append(caption)
    #     ratings.append(cols[key].number_input('Give a Rating', key=key, min_value=1,max_value=5))
    #     key += 1

    # def standardize(row):
    #     new_row = (row - row.mean()) / (row.max() - row.min())
    #     return new_row


    # def get_similar_movies(movie_name,user_rating):
    #     similar_score = item_similarity_df[movie_name]*(user_rating-2.5)
    #     similar_score = similar_score.sort_values(ascending=False)
    #     return similar_score

    # similar_movies = pd.DataFrame()

    # if st.button('SUBMIT'):
    #     df = df_rating
    #     ratings_std = df.apply(standardize)
    #     item_similarity = cosine_similarity(ratings_std.T)
    #     item_similarity_df = pd.DataFrame(item_similarity, index=df.columns, columns=df.columns)

    #     data_set = np.array((movies,user_ratings)).T

    #     for movie,user_rating in data_set:
    #         similar_movies =  similar_movies.append(get_similar_movies(movie,int(user_rating)),ignore_index=True)

    #     st.write(similar_movies.sum().sort_values(ascending=False))

    ####### JOEL #######

    # if st.button('SUBMIT'):
    #     recommended_movies = get_recommendations(df_rating, movies_to_rate, user_ratings)

    #     st.write(recommended_movies.iloc[1:])




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
