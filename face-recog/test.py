import face_recognition
from PIL import Image
from io import BytesIO
import requests

response = requests.get("https://github.com/ageitgey/face_recognition/blob/master/examples/obama2.jpg?raw=true")
response2 = requests.get("https://github.com/ageitgey/face_recognition/blob/master/examples/obama.jpg?raw=true")

img_get_from_url_1 = BytesIO(response.content)
img_get_from_url_2 = BytesIO(response2.content)

img1 = Image.open(img_get_from_url_1)
img2 = Image.open(img_get_from_url_2)

pic1 = face_recognition.load_image_file(img_get_from_url_1)
my_face_encoding = face_recognition.face_encodings(pic1)[0]

unknown_picture = face_recognition.load_image_file(img_get_from_url_2)
unknown_face_encoding = face_recognition.face_encodings(unknown_picture)[0]

results = face_recognition.compare_faces([my_face_encoding], unknown_face_encoding)

if results[0] == True:
    print("It's a picture of obama!")
else:
    print("It's not a picture of obama!")