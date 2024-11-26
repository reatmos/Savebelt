from flask import Flask, Response, request
import cv2
from detect03 import detect03

app03 = Flask(__name__)

def frames(source):
    for frame in detect03(source=source):
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app03.route('/video03') # 웹에서 받을 주소
def video():
    source = request.args.get('source', '1')  # source에 넣을 파일 경로 또는 웹캠('0', '1') 또는 'tello'
    return Response(frames(source), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app03.run(host='0.0.0.0', port=5002)