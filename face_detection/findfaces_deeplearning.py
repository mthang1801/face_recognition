from PIL import Image 
import face_recognition

# Load the jpg file into a numpy array
image = face_recognition.load_image_file("imgs/known/Ronaldo.jpg");

# Find all the faces in the image using the default HOG-based model.
# This method is fairly accurate, but not as accurate as the CNN model and not GPU accelerated.
face_locations = face_recognition.face_locations(image);

print(f"I found {len(face_locations)} in this photo");

for face_location in face_locations : 
  # Print the location of each face in this image
  top, right, bottom, left = face_location;
  print(f"A face is located at pixel location Top: {top}, Right :{right}, Bottom: {bottom}, Left: {left}");

  #access the actual face itself like this:
  face_image = image[top:bottom, left: right];
  print(face_image)
  pil_image = Image.fromarray(face_image);
  pil_image.show();
