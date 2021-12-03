import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from API_Twitter_bot_detection.api_get_data import *
import h5py
import gcsfs
import gensim


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
gcs_path_user = 'gs://tweet-project-713/models/twitter_bot_detector/v1/Logit_opt.joblib'
gcs_path_text = 'gs://tweet-project-713/models/twitter_bot_detector/v1/RNN_BIG_DROP.h5'
gcs_path_w2v = 'gs://tweet-project-713/data/word2vec.joblib'

model_user = joblib.load(tf.io.gfile.GFile(gcs_path_user, 'rb'))

model_text = load_model(h5py.File(tf.io.gfile.GFile(gcs_path_text, 'rb'), 'r'))

word2vec = joblib.load(tf.io.gfile.GFile(gcs_path_w2v, 'rb'))


# define a root `/` endpoint
@app.get("/predict")
def predict(username):
    user = user_data_request(username)
    if user != None:
        tweets = tweet_data_request(user)
        X_user = user_preprocessing(tweets, user)
        if 'empty' not in tweets.columns:
            X_tweet = tweet_preprocessing(tweets, word2vec)
            pred_user = model_user.predict(X_user)
            pred_text = np.mean(model_text.predict(X_tweet))
            return {'user_level_prediction': str(pred_user[0]), 'tweet_level_prediction': str(round(pred_text))}
        else:
            pred_user = model_user.predict(X_user)
            return {'user_level_prediction' : str(pred_user), 'tweet_level_prediction': 'could not fetch tweets for the specified user'}
    else:
        return {'oops': 'user could not be found'}
