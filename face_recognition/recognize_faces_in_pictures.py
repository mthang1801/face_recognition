import face_recognition;

#Load file jpg into a numpy Array
bill_image = face_recognition.load_image_file("imgs/known/Bill Gates.jpg");
trump_image = face_recognition.load_image_file("imgs/known/Donald Trump.jpg");
elon_image = face_recognition.load_image_file("imgs/known/Elon Musk.jpg");
ronaldo_image = face_recognition.load_image_file("imgs/known/Ronaldo.jpg");
dybala_image = face_recognition.load_image_file("imgs/known/Paulo Dybala.jpg");
steve_image = face_recognition.load_image_file("imgs/known/Steve Jobs.jpg");
pence_image = face_recognition.load_image_file("imgs/known/Mike Pence.jpg");
unknown_image = face_recognition.load_image_file("imgs/unknown/04vid-rip-speech-pelosi-videoSixteenByNine1050.jpg");

# Get a face encodings for each face in each image file
# Becase there could be more than one face in each image, it returns list of encodings
# But since I know each image only has one face, I only care about the first encoding in each image, so I grab index 0.
try : 
  bill_face_encodings = face_recognition.face_encodings(bill_image)[0];
  trump_face_encodings = face_recognition.face_encodings(trump_image)[0];
  elon_face_encodings = face_recognition.face_encodings(elon_image)[0];
  ronaldo_face_encodings = face_recognition.face_encodings(ronaldo_image)[0];
  dybala_face_encodings = face_recognition.face_encodings(dybala_image)[0];
  steve_face_encodings = face_recognition.face_encodings(steve_image)[0];
  pence_face_encodings = face_recognition.face_encodings(pence_image)[0];
  unknown_face_encodings = face_recognition.face_encodings(unknown_image)[0];
except IndexError : 
  print(f"I wasn't able to locate any faces in at least one of the images. Check the image files. Aborting... ");
  quit();

known_faces = [
  bill_face_encodings,
  trump_face_encodings,
  elon_face_encodings,
  ronaldo_face_encodings,
  dybala_face_encodings, 
  steve_face_encodings,
  pence_face_encodings
]

#results is array True/False telling if the unknown face matched anyone in the known_faces array
results = face_recognition.compare_faces(known_faces, unknown_face_encodings);
print(f"Is Bill ? {results[0]}");
print(f"Is Trump ? {results[1]}");
print(f"Is Elon ? {results[2]}");
print(f"Is Ronaldo ? {results[3]}");
print(f"Is Dybala ? {results[4]}");
print(f"Is Steve ? {results[5]}");
print(f"Is Pence ? {results[6]}");
print(f"Is the unknown face a new person that we've never seen before? {not True in results}");