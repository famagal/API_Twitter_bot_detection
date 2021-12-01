from API_Twitter_bot_detection.api_get_data import *
import ipdb
from fastapi import FastAPI
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import h5py

#gcs_path_w2v = 'gs://tweet-project-713/data/word2vec.joblib'
#word2vec = joblib.load(tf.io.gfile.GFile(gcs_path_w2v, 'rb'))

#user = user_data_request('andersson_93')
gcs_path_text = 'gs://tweet-project-713/models/twitter_bot_detector/v1/RNN_BIG.h5'
#tweets = tweet_data_request(user)
model_text = load_model(h5py.File(tf.io.gfile.GFile(gcs_path_text, 'rb'),'r' ))
#with FS.open(gcs_path_text, 'rb') as model_file:
# model_gcs = h5py.File(model_file, 'r')
# myModel = load_model(model_gcs)
#gcs_path = f"gs://{BUCKET_NAME}/models/{MODEL_NAME}/{MODEL_VERSION}/{LOCAL_MODEL_NAME}"
#print(gcs_path)
#loaded_model = tf.io.gfile.GFile(gcs_path, 'rb')
#model_gcs = h5py.File(loaded_model, 'r')
#print(model_gcs)
#X_tweet = tweet_preprocessing(tweets, word2vec)
ipdb.set_trace()
