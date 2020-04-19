from PIL import  Image, ImageDraw
import face_recognition

#Load the jpg file into numpy array
image = face_recognition.load_image_file("imgs/groups/juventusfc.jpg");

# Find all facial features in all the faces in the image
face_landmarks_list = face_recognition.face_landmarks(image);

print(f"I found {len(face_landmarks_list)} people in this photo");

# Create a PIL imagedraw object so we can draw on the picture
pil_image = Image.fromarray(image);
d = ImageDraw.Draw(pil_image);

for face_landmark in face_landmarks_list: 
  for facial_feature in face_landmark.keys() : 
    #Print the location of each facial feature in this image
    print(f"The \"{facial_feature}\" in this face has the following points : {face_landmark[facial_feature]}");
    #Let's trace out each facial feature in the image with a line!
    d.line(face_landmark[facial_feature], width =5);

#show image
pil_image.show();
pil_image.save("/identify-specific-facial_features.jpg");