import cv2
import sys


def nothing(val):
    pass


def draw_features(image, gray, face):
    NOSE_SCALE_FACTOR = cv2.getTrackbarPos(trackbar_nose, window_name) / 10
    MOUTH_SCALE_FACTOR = cv2.getTrackbarPos(trackbar_mouth, window_name) / 10
    EYE_SCALE_FACTOR = cv2.getTrackbarPos(trackbar_eyes, window_name) / 10

    NOSE_NEIGHBOURS = cv2.getTrackbarPos(trackbar_nose_neighbours, window_name)
    MOUTH_NEIGHBOURS = cv2.getTrackbarPos(trackbar_mouth_neighbours, window_name)
    EYE_NEIGHBOURS = cv2.getTrackbarPos(trackbar_eyes_neighbours, window_name)

    if NOSE_SCALE_FACTOR < 1.1:
        NOSE_SCALE_FACTOR = 1.1
    if MOUTH_SCALE_FACTOR < 1.1:
        MOUTH_SCALE_FACTOR = 1.1
    if EYE_SCALE_FACTOR < 1.1:
        EYE_SCALE_FACTOR = 1.1

    for (x, y, w, h) in face:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        eyes = eye_cascade.detectMultiScale(
            gray[y:y + int(h / 2), x:x + w],
            scaleFactor=EYE_SCALE_FACTOR,
            minNeighbors=EYE_NEIGHBOURS,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        left_eyes = left_eye_cascade.detectMultiScale(
            gray[y:y + int(h / 2), x:x + w],
            scaleFactor=EYE_SCALE_FACTOR,
            minNeighbors=EYE_NEIGHBOURS,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        right_eyes = right_eye_cascade.detectMultiScale(
            gray[y:y + int(h / 2), x:x + w],
            scaleFactor=EYE_SCALE_FACTOR,
            minNeighbors=EYE_NEIGHBOURS,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        nose = nose_cascade.detectMultiScale(
            gray[y + int(0.2 * h):y + int(0.8 * h), x:x + w],
            scaleFactor=NOSE_SCALE_FACTOR,
            minNeighbors=NOSE_NEIGHBOURS,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        mouth = mouth_cascade.detectMultiScale(
            gray[y + int(h / 2):y + h, x:x + w],
            scaleFactor=MOUTH_SCALE_FACTOR,
            minNeighbors=MOUTH_NEIGHBOURS,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        for (x_eye, y_eye, w_eye, h_eye) in left_eyes:
            cv2.rectangle(
                image,
                (x + x_eye, y + y_eye),
                (x + x_eye + w_eye, y + y_eye + h_eye),
                (255, 0, 0),
                2
            )
        for (x_eye, y_eye, w_eye, h_eye) in right_eyes:
            cv2.rectangle(
                image,
                (x + x_eye, y + y_eye),
                (x + x_eye + w_eye, y + y_eye + h_eye),
                (255, 0, 0),
                2
            )
        for (x_eye, y_eye, w_eye, h_eye) in eyes:
            cv2.rectangle(
                image,
                (x + x_eye, y + y_eye),
                (x + x_eye + w_eye, y + y_eye + h_eye),
                (255, 0, 0),
                2
            )
        for (x_nose, y_nose, w_nose, h_nose) in nose:
            cv2.rectangle(
                image,
                (x + x_nose, y + y_nose + int(0.2 * h)),
                (x + w_nose + x_nose, y_nose + int(0.2 * h) + y + h_nose // 2),
                (0, 0, 255),
                2
            )
        for (x_mouth, y_mouth, w_mouth, h_mouth) in mouth:
            cv2.rectangle(
                image,
                (x + x_mouth, y + int(h / 2) + y_mouth + h_mouth // 2),
                (x + w_mouth + x_mouth, y_mouth + y + int(h / 2) + h_mouth),
                (255, 0, 255),
                2
            )


def detect_and_display(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    FACE_SCALE_FACTOR = cv2.getTrackbarPos(trackbar_face, window_name) / 10
    FACE_NEIGHBOURS = cv2.getTrackbarPos(trackbar_face_neighbours, window_name)

    if FACE_SCALE_FACTOR < 1.1:
        FACE_SCALE_FACTOR = 1.1

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=FACE_SCALE_FACTOR,
        minNeighbors=FACE_NEIGHBOURS,
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    draw_features(image, gray, faces)
    cv2.imshow(window_name, image)


mode = int(sys.argv[1])
if mode == 0:
    try:
        imagePath = sys.argv[2]
    except IndexError:
        print("No file provided")
        exit(0)

face_cascade_path = 'haarcascades/haarcascade_frontalface_alt2.xml'
eye_cascade_path = 'haarcascades/haarcascade_eye.xml'
right_eye_cascade_path = 'haarcascades/haarcascade_righteye_2splits.xml'
left_eye_cascade_path = 'haarcascades/haarcascade_lefteye_2splits.xml'
nose_cascade_path = 'haarcascades/haarcascade_nose_2.xml'
mouth_cascade_path = 'haarcascades/haarcascade_smile.xml'

face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
right_eye_cascade = cv2.CascadeClassifier(right_eye_cascade_path)
left_eye_cascade = cv2.CascadeClassifier(left_eye_cascade_path)
nose_cascade = cv2.CascadeClassifier(nose_cascade_path)
mouth_cascade = cv2.CascadeClassifier(mouth_cascade_path)


window_name = "Picture"
cv2.namedWindow(window_name)
trackbar_face = "Face"
trackbar_nose = "Nose"
trackbar_eyes = "Eyes"
trackbar_mouth = "Mouth"
trackbar_face_neighbours = "Face Neighbours"
trackbar_nose_neighbours = "Nose Neighbours"
trackbar_eyes_neighbours = "Eyes Neighbours"
trackbar_mouth_neighbours = "Mouth Neighbours"

cv2.createTrackbar(trackbar_face, window_name, 11, 30, nothing)
cv2.createTrackbar(trackbar_face_neighbours, window_name, 5, 30, nothing)
cv2.createTrackbar(trackbar_nose, window_name, 11, 30, nothing)
cv2.createTrackbar(trackbar_nose_neighbours, window_name, 5, 30, nothing)
cv2.createTrackbar(trackbar_mouth, window_name, 14, 30, nothing)
cv2.createTrackbar(trackbar_mouth_neighbours, window_name, 8, 30, nothing)
cv2.createTrackbar(trackbar_eyes, window_name, 11, 60, nothing)
cv2.createTrackbar(trackbar_eyes_neighbours, window_name, 5, 60, nothing)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print('Error opening video capture')
    exit(0)

while True:

    if mode == 0:
        image = cv2.imread(imagePath)
        detect_and_display(image)

    if mode == 1:
        ret, frame = cap.read()
        if frame is None:
            print('No captured frame')
            break
        detect_and_display(frame)

    key = cv2.waitKey(0)
    if key == 27:
        break

if mode == 1:
    cap.release()

cv2.destroyWindow(window_name)
