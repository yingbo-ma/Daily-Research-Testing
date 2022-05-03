import os
import cv2
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Define paths
base_dir = os.path.dirname(__file__)

print(base_dir)

if not os.path.exists('left_faces'):
    print("New directory created")
    os.makedirs('left_faces')

if not os.path.exists('right_faces'):
    print("New directory created")
    os.makedirs('right_faces')

vc = cv2.VideoCapture("all.mpeg")
ret, first_frame = vc.read()
imageIndex = 0
resize_dim = 600

while (vc.isOpened()):

    # resize_dim = 600
    max_dim = max(first_frame.shape)
    scale = resize_dim / max_dim
    first_frame = cv2.resize(first_frame, None, fx=scale, fy=scale)
    gray = cv2.cvtColor(src=first_frame, code=cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        x1 = face.left() # left point
        y1 = face.top()  # top point
        x2 = face.right()  # right point
        y2 = face.bottom()  # bottom point

        if(x1 < 0):
            x1 = 0

        if(y1 < 0):
            y1 = 0

        if(x2 < 0):
            x2 = 0

        if(y2 < 0):
            y2 = 0

        # Draw a rectangle
        cv2.rectangle(img=first_frame, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=4)
        landmarks = predictor(image=gray, box=face)

        # draw dots on detected faces
        # Jaw Points = 0–16
        # Right Brow Points = 17–21
        # Left Brow Points = 22–26
        # Nose Points = 27–35
        # Right Eye Points = 36–41
        # Left Eye Points = 42–47
        # Mouth Points = 48–60
        # Lips Points = 61–67
        #
        for n in range(48, 67):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            # Draw a circle
            cv2.circle(img=first_frame, center=(x, y), radius=3, color=(0, 255, 0), thickness=-1)

        face_to_save = first_frame[y1:y2, x1:x2]  # the problem is here the code didn't get x1, x2, y1, y2
        # break

    # cv2.imshow(winname="Face", mat=first_frame)
    cv2.imshow(winname="Face", mat=face_to_save)
    # cv2.imwrite(base_dir + '/left_faces/' + str(imageIndex) + ".jpg", face_to_save)

    # Read a frame from video
    fps = vc.get(cv2.CAP_PROP_FPS)
    for j in range(int(fps) - 1):
        vc.grab()

    ret, next_frame = vc.read()
    first_frame = next_frame
    imageIndex += 1

    # Frame are read by intervals of 500 millisecond.
    # The programs breaks out of the while loop when the user presses the ‘q’ key
    if cv2.waitKey(500) & 0xFF == ord('q'):
        break

vc.release()
cv2.destroyAllWindows()