import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import pandas as pd
from PIL import Image


cascade_path = './face emotion/haarcascade_frontalface_default.xml'
face_detector = cv2.CascadeClassifier(cascade_path)
emotion_dict = {0: 'Angry', 1: 'Disgusted', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}
music_dict = {0: './songs/angry.csv', 1: './songs/disgusted.csv', 2: './songs/fear.csv', 3: './songs/happy.csv',
              4: './songs/neutral.csv', 5: './songs/sad.csv', 6: './songs/surprise.csv'}
emt = [0]

"""Load model from json and h5 file"""
json_file = open('./model/model_accuracy_66%.json', 'r')
json = json_file.read()
json_file.close()
emotion_model = model_from_json(json)
emotion_model.load_weights('./model/model_accuracy_66%.h5')

global cap
cap = cv2.VideoCapture(0)

class VideoCamera(object):
  def get_frame(self):
    global df
    ret, frame = cap.read()
    frame = cv2.resize(frame, (650, 500))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))
    df = pd.read_csv(music_dict[emt[0]])
    df = df.head(12)
    for (x, y, w, h) in faces:
      roi_gray = gray[y:y + h, x:x + w]
      input = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
      prediction = emotion_model.predict(input)
      idx = np.argmax(prediction)
      emt[0] = idx
      df = get_music()
      cv2.rectangle(frame, (x, y), (x + w, y + h), (124, 252, 0), 2)
      cv2.putText(frame, emotion_dict[idx], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    last_frame = frame.copy()
    jpeg = cv2.imencode('.jpg', last_frame)[1].tobytes()
    return jpeg, df

def get_music():
  df = pd.read_csv(music_dict[emt[0]])
  df = df.head(12)
  return df