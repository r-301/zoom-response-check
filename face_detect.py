import cv2
import numpy as np
import boto3

# スケールや色などの設定
scale_factor = .15
green = (0,255,0)
red = (0,0,255)
frame_thickness = 2
#cap = cv2.VideoCapture(0)
rekognition = boto3.client('rekognition')

# フォントサイズ
fontscale = 1.0
# フォント色 (B, G, R)
color = (0, 120, 238)
# フォント
fontface = cv2.FONT_HERSHEY_DUPLEX


from PIL import ImageGrab

ImageGrab.grab().save("./capture/PIL_capture.png")

# フレームをキャプチャ取得
#ret, frame = cap.read()
frame = cv2.imread("./capture/PIL_capture.png")
height, width, channels = frame.shape
frame = cv2.resize(frame,(int(width/2),int(height/2)),interpolation = cv2.INTER_AREA)

    # jpgに変換 画像ファイルをインターネットを介してAPIで送信するのでサイズを小さくしておく
small = cv2.resize(frame, (int(width * scale_factor), int(height * scale_factor)))
ret, buf = cv2.imencode('.jpg', small)

    # Amazon RekognitionにAPIを投げる
faces = rekognition.detect_faces(Image={'Bytes':buf.tobytes()}, Attributes=['ALL'])

    # 顔の周りに箱を描画する
for face in faces['FaceDetails']:
    smile = face['Smile']['Value']
    cv2.rectangle(frame,
                    (int(face['BoundingBox']['Left']*width/2),
                    int(face['BoundingBox']['Top']*height/2)),
                    (int((face['BoundingBox']['Left']/2+face['BoundingBox']['Width']/2)*width),
                    int((face['BoundingBox']['Top']/2+face['BoundingBox']['Height']/2)*height)),
                    green if smile else red, frame_thickness)
    emothions = face['Emotions']
    i = 0
    score = 0
    for emothion in emothions:
        
        if emothion["Type"] == "HAPPY":
            score = score + emothion["Confidence"]
        elif emothion["Type"] == "DISGUSTED":
            score = score - emothion["Confidence"]
        elif emothion["Type"] == "SURPRISED":
            score = score + emothion["Confidence"]
        elif emothion["Type"] == "ANGRY":
            score = score - emothion["Confidence"]
        elif emothion["Type"] == "CONFUSED":
            score = score - emothion["Confidence"]
        elif emothion["Type"] == "CALM":
            score = score - emothion["Confidence"]
        elif emothion["Type"] == "SAD":
            score = score - emothion["Confidence"]
        i += 1
        if i == 7:
            cv2.putText(frame,
            "interested" +":"+ str(round(score,2)),
            (int(face['BoundingBox']['Left']*width/2),
            int(face['BoundingBox']['Top']*height/2)),
            fontface,
            fontscale,
            color)


        

# 結果をディスプレイに表示
cv2.imshow('frame', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()