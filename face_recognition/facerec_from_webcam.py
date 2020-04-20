import face_recognition
import numpy as np
import cv2 

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

#Create arrays of known face encodings and their names
known_face_encodings = [
  mvt_face_encodings,
  trump_face_encodings, 
  ronaldo_face_encodings
]

known_face_names = [
  "MVT",
  "Trump",
  "Ronaldo"
]

while True: 
  # Grab a single frame of video
  ret, frame = video_capture.read();
  #Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
  rgb_frame = frame[:, :, ::-1];
  # Find all the faces and face enqcodings in the frame of video
  face_locations = face_recognition.face_locations(rgb_frame);
  face_encodings = face_recognition.face_encodings(rgb_frame, face_locations);

  # Loop through each face in this frame of video
  for (top, right, bottom, left) , face_encoding in zip(face_locations, face_encodings) : 
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding);
    name = "Unknown";
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding);
    best_match_index = np.argmin(face_distances);
    if matches[best_match_index] : 
      name = known_face_names[best_match_index];
    #Draw box around the face
    cv2.rectangle(frame, (left, top), (right, bottom), (13, 0, 255),2);
    # Draw a label with a name below the face
    cv2.rectangle(frame, (left , bottom-35), (right, bottom), (13,0,255), cv2.FILLED);
    font = cv2.FONT_HERSHEY_SIMPLEX;
    cv2.putText(frame,name, (left+6, bottom -12), font, 1.0, (255,255,255), 1);

  #Display the resulting image
  cv2.imshow("Video", frame);
  # Hit 'q' on the keyboard to quit!
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

# Release handle to the webcam
video_capture.release();
cv2.destroyAllWindows();