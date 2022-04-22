import pandas as pd
import numpy as np
import os
import streamlit as st
import pickle
from datetime import datetime, timedelta, timezone
from time import sleep
import requests
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

path_to_repo = os.getcwd()
path_to_data = os.path.join(path_to_repo, 'data')


def bearer_oauth(r):
    """
    Method required by bearer token authentication.
    """
    bearer_token = "AAAAAAAAAAAAAAAAAAAAAKovZQEAAAAAWKm9VUw8iFuu27OQ1m34HwIR5VY%3DbPl1vQZq69vtsvc5JPq5HD9AiatIKCw58RBeXOugqsl5Holcog"
    r.headers["Authorization"] = f"Bearer {bearer_token}"
    r.headers["User-Agent"] = "v2RecentSearchPython"
    return r


def connect_to_endpoint(url, params):
    response = requests.get(url, auth=bearer_oauth, params=params)
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()


def load_tweets(topic):
    search_url = "https://api.twitter.com/2/tweets/search/recent"
    end_time = (datetime.now(timezone.utc).astimezone() - timedelta(seconds=30)).isoformat()

    query_params = {'query': topic,
                    'end_time': end_time,
                    'max_results': '100',
                    'tweet.fields': 'text,geo,created_at,lang,public_metrics,source',
                    'user.fields': 'id,name,username,created_at,description,public_metrics,verified',
                    'place.fields': 'full_name,id,country,country_code,geo,name,place_type'}

    json_response = connect_to_endpoint(search_url, query_params)

    return json_response


def clean_text(X):
    X = X.lower().split()
    X_clean = filter(str.isalnum, X)

    return ' '.join(X_clean)


def remove_stop_words(X):
    X = X.split()
    stopwords_list = stopwords.words('english')
    newStopWords = ["rt"]
    stopwords_list.extend(newStopWords)
    X_new = [x for x in X if x not in stopwords_list]

    return ' '.join(X_new)


def vectorize(test_set):
    x_train = pd.read_csv(os.path.join(path_to_data, 'final_dataset.csv'))
    cv = CountVectorizer()
    cv.fit_transform(x_train["final_tweet"])

    return cv.transform(test_set)


def table_from_response(search_term):
    tweets_loaded = load_tweets(search_term)
    tweets = pd.DataFrame.from_dict(tweets_loaded["data"])
    tweets = tweets[tweets.lang == 'en']

    return tweets


def classify(tweets):
    result = []
    for row in tweets.loc[:, "text"]:
        result.append(clean_text(row))
    tweets["clean_tweet"] = result

    result = []
    for row in tweets.loc[:, "clean_tweet"]:
        result.append(remove_stop_words(row))

    tweets["final_tweet"] = result
    tweets_cv = vectorize(tweets["final_tweet"])

    predictions = model.predict(tweets_cv)
    tweets["class"] = predictions

    result = np.unique(predictions, return_counts=True)

    return result, st.dataframe(tweets[["text", "created_at", "final_tweet", "class"]])


def update_state(state, search_term, tweets_table):
    tweets_table = table_from_response(search_term)
    state = False

    return tweets_table


with open("svm_model.pkl", 'rb') as file:
    model = pickle.load(file)

search_term = st.text_input("Which topic interests you today?")
live = st.checkbox('I want to monitor live')
state = False

placeholder = st.empty()
container = st.container()

if search_term != '':
    with placeholder:
        tweets_table = table_from_response(search_term)
        result, final_table = classify(tweets_table)
        container.write(f"Current hate level on twitter is: {round(result[1][1]/(result[1][1]+result[1][0])*100, 2)} %")


        while live:
            if state:
                with placeholder:
                    tweets_table = update_state(state, search_term, tweets_table)
                    result, final_table = classify(tweets_table)
                    container.write(f"Current hate level on twitter is: {round(result[1][1]/(result[1][1]+result[1][0])*100, 2)} %")
            sleep(5)

            state = True
            if not live:
                placeholder.empty()
                break
