from flask import Flask, request, render_template, Response
from utils import *

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

def generate(camera):
    while True:
        global df
        frame, df = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/table')
def generate_table():
    return df.to_json(orient = 'records')

if __name__ == '__main__':
    app.run(host='localHost', port=3000, debug=True)