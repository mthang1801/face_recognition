import face_recognition
import cv2 
import numpy as np

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0);

# Load Myself picture and learn how to recognize it
mvt_image = face_recognition.load_image_file("imgs/known/MVT.jpeg");
mvt_face_encodings = face_recognition.face_encodings(mvt_image)[0];

# Load Trump picture and learn how to recognize it 
trump_image = face_recognition.load_image_file("imgs/known/Donald Trump.jpg");
trump_face_encodings = face_recognition.face_encodings(trump_image)[0];

#Load Ronaldo picture and learn how to recognize it
ronaldo_image = face_recognition.load_image_file("imgs/known/Ronaldo.jpg");
ronaldo_face_encodings = face_recognition.face_encodings(ronaldo_image)[0];

#Load Messi picture and learn how to recognize it
messi_image = face_recognition.load_image_file("imgs/known/Messi.jpg");
messi_face_encodings = face_recognition.face_encodings(messi_image)[0];

#Load Messi picture and learn how to recognize it
co_image = face_recognition.load_image_file("imgs/known/Co.jpeg");
co_face_encodings = face_recognition.face_encodings(co_image)[0];

#Create arrays of known face encodings and their names
known_face_encodings = [
  mvt_face_encodings,
  trump_face_encodings, 
  ronaldo_face_encodings,
  messi_face_encodings,
  co_face_encodings,
]

known_face_names = [
  "MVT",
  "Trump",
  "Ronaldo",
  "Messi",
  "Co"
]

# Initialize some variables
face_locations = [];
face_encodings = [] ;
face_names = [] ;
face_landmarks_list = [] ;
process_this_frame = True;

while 1 : 
  # Grab a single frame of video
  ret, frame = video_capture.read();

  # Resize frame of video to 1/4 size for faster face recognition processing
  small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25);

  # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
  rgb_small_frame = small_frame[:, :, ::-1];
  # Only process every other frame of video to save time
  if process_this_frame : 
    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_small_frame);
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations);
    face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame);

    face_names = []; 
    for face_encoding in face_encodings : 
      matches = face_recognition.compare_faces(known_face_encodings, face_encoding);
      name= "Unknown";
      face_distances = face_recognition.face_distance(known_face_encodings, face_encoding);
      best_match_index = np.argmin(face_distances);
      if matches[best_match_index] :
        name = known_face_names[best_match_index] ;
        face_names.append(name);
  process_this_frame = not process_this_frame;  

  # Display the results
  for (top, right, bottom, left), name, face_landmarks in zip(face_locations, face_names, face_landmarks_list) : 
    
    # Scale back up face locations since the frame we detected in was scaled to 1/4 size
    top *=4;
    right *=4 ;
    bottom *=4;
    left*=4;
    FACIAL_LANDMARKS_IDXS = {};
    for face_landmark in face_landmarks:
      FACIAL_LANDMARKS_IDXS[face_landmark] =  face_landmarks[face_landmark];
   
    for (i, key) in enumerate(FACIAL_LANDMARKS_IDXS.keys()):
      pts = np.array(FACIAL_LANDMARKS_IDXS[key], np.int32);
      cv2.polylines(frame, [pts], True, (0,255,255))
    #Draw a box around the face
    cv2.rectangle(frame, (left, top), (right, bottom), (255, 140, 0), 2);
    #Draw label with name below the face
    cv2.rectangle(frame, (left, bottom -30), (right, bottom), (255, 140,0), cv2.FILLED);
    font = cv2.FONT_HERSHEY_COMPLEX;
    cv2.putText(frame,name, (left+6, bottom-10),font, 1.0, (255,255,255), 1 );
  # Display the resulting image
  cv2.imshow("Video", frame);
  # Hit 'q' on the keyboard to quit!
  if cv2.waitKey(1) & 0xFF == ord("q") : 
    break;

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()

    

      