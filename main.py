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

    # Resize Image for faster face detection
    resized_frame = cv.resize(frame,(0, 0), fx=0.25, fy=0.25)
    
    # Aquire all faces locations 
    faces = fc.face_locations(resized_frame)
    
    # Draw a rectangle around the detected face
    for (top,right,bottom,left) in faces:
        cv.rectangle(frame,(left*4,top*4),(right*4,bottom*4),GREEN,thickness=RECTANGLE_THICKNESS)

    # Show the current frame omn the display
    cv.imshow('Face Recognition', frame)

    # Press q key to break the loop and quit the program
    if cv.waitKey(1) == ord('q'):
        break


# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
