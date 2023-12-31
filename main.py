# Import Libraries
import cv2 as cv
import face_recognition as fc
import os 

# Colors
GREEN = (0,255,0)
RED = (0,0,255)
BLUE = (255,0,0)
WHITE = (255,255,255)

# Constants
RECTANGLE_THICKNESS = 5
PEOPLE_DIR = '/Users/dan/Documents/dev/opencv_project/Face Recognition Project/People'
FONT = cv.FONT_HERSHEY_DUPLEX
SCALE = 0.25

# List people encodings and another list with their names. Both list share the same index 
people_encodings = []
people_names = []
faces_not_found = ''

# Initilize all people I want to recognize and their encodings
for person in os.listdir(PEOPLE_DIR):
    if not person.startswith('.'):
        # Load image
        curr_img_path = os.path.join(PEOPLE_DIR,person)
        load_curr_img = fc.load_image_file(curr_img_path)
        # Delete the jpg extension
        person = person.replace('.jpg','')
        person = person.replace('.jpeg','')
        face_location = fc.face_locations(load_curr_img)
        # check if face was found in the given image
        # If not add to the string of faces not found
        # If found then add encoding and the corresponding name
        if len(face_location)!=1:
            faces_not_found+=' ' + person + ','
        else:
            curr_img_encoding = fc.face_encodings(load_curr_img)[0]
            people_encodings.append(curr_img_encoding)
            people_names.append(person)

# Print which faces where not found
if faces_not_found:
    print('The following faces provided were not found:' + faces_not_found[0:-1])
else:
    print('All faces were found')

answer = input('Do you wish to continue? [y/n]')

# Continue the app if the user chose to continue
if answer in ('y','Y','yes','Yes'):
    # Video capture from the webcam
    cap = cv.VideoCapture(0)

    # Check if webcam is working correctly
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

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
        resized_frame = cv.resize(frame,(0, 0), fx=SCALE, fy=SCALE)

        # Convert input frame (opencv works with BGR) to RGB (Face recogniton):
        rgb_frame = cv.cvtColor(resized_frame, cv.COLOR_BGR2RGB)

        # Faces Location:
        faces_location = fc.face_locations(rgb_frame)
        
        # Frame encoding 
        frame_encodings = fc.face_encodings(rgb_frame,faces_location)

        # check every face encoding the known people
        for frame_encoding,(top,right,bottom,left) in zip(frame_encodings,faces_location):

            # Initilize Unknown namm
            name = 'Unknown'

            # return a list of all matches if the webcam recognizes a face from known people
            list_of_matches = fc.compare_faces(people_encodings,frame_encoding)

            # length of matches
            n = len(list_of_matches)

            # If current face matches a known person then change the name to the persons name
            for i in range(n):
                if list_of_matches[i]:
                    name = people_names[i]
            
            # Scale back positions
            top *=4
            right *=4
            bottom *=4
            left *=4

            # Draw rectangle around detected faces
            cv.rectangle(frame,(left,top),(right,bottom),GREEN,thickness=RECTANGLE_THICKNESS)
            cv.rectangle(frame,(left,bottom),(right,bottom-35),GREEN,thickness=RECTANGLE_THICKNESS)
            # Put name of the person found
            cv.putText(frame,name,(left + 6, bottom - 6),FONT,1.0,WHITE,2)
        
        # Show the current frame omn the display
        cv.imshow('Face Recognition', frame)

        # Press q key to break the loop and quit the program
        if cv.waitKey(1) == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()

# Indicate to the user the program is done
print('---------------------------Quiting Program---------------------------')
