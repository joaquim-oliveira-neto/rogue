import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
st.title('Rogue Inteligent Movie Recomendation systems')
df = pd.read_csv('raw_data/path.csv').head()
key = 0
ratings = []
movies = []
for movie in df.title:
    key += 1
    st.write(movie)
    movies.append(movie)
    ratings.append(st.number_input('Give a Rating',
                                   key=key, min_value=1, max_value=5))
if st.button('SUBMIT'):
    st.write(pd.DataFrame({'movies': movies, 'ratings': ratings}))
