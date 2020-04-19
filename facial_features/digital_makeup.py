from PIL import Image, ImageDraw
import face_recognition

#Load file jpg into an numpy array
image = face_recognition.load_image_file("imgs/known/Ronaldo.jpg");

#Find all ficial features in all the faces in the image
face_landmarks_list = face_recognition.face_landmarks(image);

pil_image = Image.fromarray(image);

for face_landmarks in face_landmarks_list: 
  for facial_feature in face_landmarks.keys() : 
      print(f"The \"{facial_feature}\" in this face has the following points : {face_landmarks[facial_feature]}");
  d = ImageDraw.Draw(pil_image,"RGBA");
  # Make the eyebrows into a nightmare
  d.polygon(face_landmarks["left_eyebrow"], fill=(68, 54, 39, 190));
  d.polygon(face_landmarks["right_eyebrow"], fill=(68, 54, 39, 190)) ;
  d.line(face_landmarks["left_eyebrow"], fill=(68, 54, 39, 150), width=2);
  d.line(face_landmarks["right_eyebrow"], fill=(68, 54, 39, 150), width=2);

  # Gloss the lips
  d.polygon(face_landmarks["top_lip"], fill=(150, 0, 0, 128));
  d.polygon(face_landmarks["bottom_lip"], fill=(150, 0, 0, 128));
  # d.line(face_landmarks["top_lip"], fill=(150, 0, 0, 64), width =8);
  # d.line(face_landmarks["bottom_lip"], fill=(150, 0, 0, 64), width = 8);

  # Sparkle the eyes
  d.polygon(face_landmarks["left_eye"], fill=(255, 255, 255, 70));
  d.polygon(face_landmarks["right_eye"], fill=(255, 255, 255, 70));

  # Apply some eyeliner
  d.line(face_landmarks["left_eye"] + [face_landmarks["left_eye"][0]] , fill=(0,0,0,110), width = 6);
  d.line(face_landmarks["right_eye"] + [face_landmarks["right_eye"][0]] , fill=(0,0,0,110), width = 6);

  # dark chin
  d.line(face_landmarks["chin"] , fill=(0, 0, 0, 82), width =5)
pil_image.show();

    
    