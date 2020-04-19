#find faces in photograph
import face_recognition

image = face_recognition.load_image_file("imgs/groups/team1.jpg");
find_locations = face_recognition.face_locations(image);

#Array of cords of each face
print(find_locations);

print(f"There are {len(find_locations)} people in this image");
