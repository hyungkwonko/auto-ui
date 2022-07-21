import pickle
import face_recognition
import PIL
import PIL.Image
import PIL.ImageFont
from PIL import ImageOps
import PIL.ImageDraw
from tkinter import filedialog
from tkinter import *
import math

with open('encoded_people.pickle', 'rb') as filename:
    people = pickle.load(filename)

print("[INFO] Face Detection")
img = face_recognition.load_image_file('test.jpeg')
img_loc = face_recognition.face_locations(img, model="hog")
img_enc = face_recognition.face_encodings(img, known_face_locations=img_loc)
face_img = PIL.Image.fromarray(img)

print("[INFO] Face Tagging")
unknown_faces_location = []
unknown_faces_enconded = []
for i in range(0,len(img_enc)):
    best_match_count = 0
    best_match_name = "unknown"
    for k,v in people.items():
        result = face_recognition.compare_faces(v,img_enc[i],tolerance=0.5)
        count_true = result.count(True)
        if  count_true > best_match_count: # TO find the best person that matches with the face
            best_match_count = count_true
            best_match_name = k
    # Draw and write on photo
    top,right,bottom,left = img_loc[i]
    draw = PIL.ImageDraw.Draw(face_img)
    font = PIL.ImageFont.truetype("timesbd.ttf",size=max(math.floor((right-left)/6),16))
    draw.rectangle([left,top,right,bottom], outline="red", width=3)
    draw.rectangle((left, bottom, left + font.getbbox(best_match_name)[0] , bottom +  font.getbbox(best_match_name)[1]*1.2), fill='black')
    draw.text((left,bottom), best_match_name, font=font )
    if best_match_count == 0: # keep a list of unknown faces for Learning Phase
        unknown_faces_location.append(img_loc[i])
        unknown_faces_enconded.append(img_enc[i])

face_img.save('result.png')

print(type(face_img))

print(f"# of unknown people: {len(unknown_faces_enconded)}")