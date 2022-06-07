import cv2
import argparse
from tensorflow.keras.models import load_model

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image',
                default='./dataset/sample.jpg',
                help="Path to image to infer.")
ap.add_argument('-c', '--cascade',
                default='./models/haarcascade_frontalface_default.xml',
                help="Path to FrontalFace Haarcascade file.")
ap.add_argument('-m', '--model',
                default='./models/baseline.hdf5',
                help="Path to trained keras model.")

args = vars(ap.parse_args())

face_detector = cv2.CascadeClassifier(args['cascade'])
smile_detector = load_model(args['model'])

img = cv2.imread(args['image'])
clone_img = img.copy()

gray = cv2.cvtColor(clone_img, cv2.COLOR_BGR2GRAY)
rects = face_detector.detectMultiScale(gray, scaleFactor=1.1,
                                       minNeighbors=5,
                                       minSize=(30, 30),
                                       flags=cv2.CASCADE_SCALE_IMAGE)

for (fX, fY, fW, fH) in rects:
    roi = gray[fY: fY + fH, fX: fX + fW]
    roi = cv2.resize(roi, (64, 64))
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi = roi.reshape(1, *roi.shape)

    pred = smile_detector.predict(roi)[0][0]
    label = 'smiling' if pred > 0.5 else 'not smiling'

    cv2.putText(img, f"{label} [{(pred * 100):.2f}%]", (fX, fY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    cv2.rectangle(img, (fX, fY), (fX + fW, fY + fH),
                  (0, 0, 255), 2)

cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
