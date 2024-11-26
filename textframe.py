from flask import Flask, Response
import cv2
from calc import calc

app04 = Flask(__name__)

def frames():
    for frame in calc():
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app04.route('/frame01') # 웹에서 받을 주소
def video():
    return Response(frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app04.run(host='0.0.0.0', port=5003)