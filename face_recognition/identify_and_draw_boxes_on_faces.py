from  PIL import Image, ImageDraw 
import numpy as np 
import face_recognition

# Load first sample picture and learn how to recognize it
ronaldo_image = face_recognition.load_image_file("imgs/known/Ronaldo.jpg");
ronaldo_face_encodings = face_recognition.face_encodings(ronaldo_image)[0];

# Load second sample picture and learn how to recognize it
dybala_image = face_recognition.load_image_file("imgs/known/Paulo Dybala.jpg");
dybala_face_encodings = face_recognition.face_encodings(dybala_image)[0];

known_face_encodings = [
  ronaldo_face_encodings,
  dybala_face_encodings
]

known_face_names = [
  "Cristiano Ronaldo",
  "Dybala",
]

# Load an image with an unknown face
unknown_image = face_recognition.load_image_file("imgs/groups/juventusfc.jpg");
#Find all the faces and face encodings in unknown image
face_locations = face_recognition.face_locations(unknown_image);
face_encodings = face_recognition.face_encodings(unknown_image,face_locations);
pil_image = Image.fromarray(unknown_image);
draw = ImageDraw.Draw(pil_image);

for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings) : 
  # See if the face is a match for the known face(s)
  matches = face_recognition.compare_faces(known_face_encodings, face_encoding);
  name = "Unknown";
  face_distance = face_recognition.face_distance(known_face_encodings, face_encoding);
  best_match_index = np.argmin(face_distance);
  if matches[best_match_index] : 
    name = known_face_names[best_match_index];

  # Draw a box around the face using the Pillow module
  draw.rectangle(((left, top), (right, bottom)), outline=(0,0,255));

  #  Draw a label with a name below the face
  text_width, text_height = draw.textsize(name);
  draw.rectangle(((left, bottom -text_height - 10), (right, bottom)), fill= (0,0,255), outline=(0,0,255));
  draw.text((left+6, bottom- text_height- 5), name, fill=(255,255,255,255));
#delete draw
del draw;

#show image
pil_image.show();

pil_image.save("identify_and_draw_boxes_faces.jpg")