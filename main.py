# Import Libraries
import cv2 as cv
import face_recognition as fc
import os 

# Colors
GREEN = (0,255,0)
BLUE = (0,0,255)
BLUE = (255,0,0)
WHITE = (255,255,255)

# Constants
RECTANGLE_THICKNESS = 5
PEOPLE_DIR = '/Users/dan/Documents/dev/opencv_project/Face Recognition Project/People'
FONT = cv.FONT_HERSHEY_DUPLEX
SCALE = 0.25

# Video capture from the webcam
cap = cv.VideoCapture(0)

# Check if webcam is working correctly
if not cap.isOpened():
    print("Cannot open camera")
    exit()

people_encodings = []
people_names = []

# Initilize all known people and their encodings
for person in os.listdir(PEOPLE_DIR):
    if not person.startswith('.'):
        curr_img_path = os.path.join(PEOPLE_DIR,person)
        load_curr_img = fc.load_image_file(curr_img_path)
        curr_img_encoding = fc.face_encodings(load_curr_img)[0]
        people_encodings.append(curr_img_encoding)
        person = person.replace('.jpg','')
        person = person.replace('.jpeg','')
        people_names.append(person)

# While loop that continously shows the video from the webcam with face 
# recogntion. Press q key to exit.
while True:

    name = 'Unknown'

    # Read Frame by frame
    ret,frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Resize Image for faster face detection
    resized_frame = cv.resize(frame,(0, 0), fx=SCALE, fy=SCALE)

    # Convert input frame to rgb:
    rgb_frame = cv.cvtColor(resized_frame, cv.COLOR_BGR2RGB)

    # Faces Location:
    faces_location = fc.face_locations(rgb_frame)
    
    # Frame encoding 
    frame_encodings = fc.face_encodings(rgb_frame,faces_location)

    for frame_encoding in frame_encodings:

        # return a list of all mathces if the webcam recognizes a face from known people
        list_of_matches = fc.compare_faces(people_encodings,frame_encoding)

        # length of matches
        n = len(list_of_matches)

        # If current face matches a known person then change the name to the persons name
        for i in range(n):
            if list_of_matches[i]:
                name = people_names[i]
        
        
    # Draw a rectangle around the detected face
    for (top,right,bottom,left) in faces_location:

        # Scale back positions
        top *=4
        right *=4
        bottom *=4
        left *=4

        cv.rectangle(frame,(left,top),(right,bottom),GREEN,thickness=RECTANGLE_THICKNESS)
        cv.rectangle(frame,(left,bottom),(right,bottom-35),GREEN,thickness=RECTANGLE_THICKNESS)
        cv.putText(frame,name,(left + 6, bottom - 6),FONT,1.0,WHITE,2)

    # Show the current frame omn the display
    cv.imshow('Face Recognition', frame)

    # Press q key to break the loop and quit the program
    if cv.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
