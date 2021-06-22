import cv2
import numpy as np 
import tensorflow as tf

from flask import Flask, render_template, Response


app = Flask(__name__)
camera = cv2.VideoCapture(1)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
input_shape =  (128,128,3)
labels_dict = {0:'Mask On' , 1:'Mask Off'}
color_dict = {0:(61,235,52) , 1:(0,0,255)}
model = tf.keras.models.load_model('mask_mnv2.h5')
size = 4


def gen_frames():  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            font = cv2.FONT_HERSHEY_SIMPLEX
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face = face_cascade.detectMultiScale(gray,1.3,5)
            
            for (x,y,w,h) in face:
                face_image = frame[y:y+h, x:x+w]
                resized = cv2.resize(face_image, (input_shape[0], input_shape[1]))
                reshaped = np.reshape(resized, (1,input_shape[0], input_shape[1],3))
                result = model.predict(reshaped)
                label = np.argmax(result,axis=1)[0]
                cv2.rectangle(frame , (x,y), (x+w,y+h), color_dict[label],2)
                cv2.rectangle(frame,(x,y-40), (x+w,y), color_dict[label],-1)
                cv2.putText(frame,labels_dict[label],(x,y-10),font,1,(255,255,255),2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route('/')
def home():
    return render_template('start.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)