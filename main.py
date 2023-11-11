# Import Libraries
import cv2 as cv
import face_recognition as fc
import os 

# Constants
GREEN = (0,255,0)
Red = (0,0,255)
Blue = (255,0,0)
RECTANGLE_THICKNESS = 3
PEOPLE_DIR = '/Users/dan/Documents/dev/opencv_project/Face Recognition Project/Known People'

# Video capture from the webcam
cap = cv.VideoCapture(0)

# Check if webcam is working correctly
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# people = []

# for person in os.listdir('/Users/dan/Documents/dev/opencv_project/Face Recognition Project/Known People'):
#     if not person.startswith('.'):
#         curr_img_path = os.path.join('/Users/dan/Documents/dev/opencv_project/Face Recognition Project/Known People',person)
#         load_curr_img = fc.load_image_file(curr_img_path)
#         curr_img_encoding = fc.face_encodings(load_curr_img)
#         people.append(curr_img_encoding)

# While loop that continously shows the video from the webcam with face 
# recogntion. Press q key to exit.
while True:
    # Read Frame by frame
    ret,frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Convert the frame to grayscale for face detection
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    # gray = cv.resize(gray,(0, 0), fx=0.25, fy=0.25)

    # Face detecion using the built in haar cascade
    haar_cascade = cv.CascadeClassifier('haar_face.xml')

    faces = haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)
    
    # faces = fc.face_locations(gray)
    
    # Draw a rectangle around the detected face
    for (x,y,w,h) in faces:
        cv.rectangle(frame,(x,y),(x+w,y+h),GREEN,thickness=RECTANGLE_THICKNESS)

    # Show the current frame omn the display
    cv.imshow('Face Recognition', frame)

    # Press q key to break the loop and quit the program
    if cv.waitKey(1) == ord('q'):
        break


# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
