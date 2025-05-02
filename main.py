import cv2
import os
import glob
from deepface import DeepFace


# Configuration
KNOWN_FACES_DIR = "dataset/known_faces" #Folder with subfolder of each known person
MODEL_NAME = "ArcFace"  # You can also try: "Facenet", "VGG-Face", "Dlib", etc.
DETECTOR = "retinaface" #backend for detecting faces

# Load model (trigger GPU warm-up)
print("🧠 Loading DeepFace model...")
DeepFace.build_model(MODEL_NAME) #Load the model into memory

# Find first known image in subfolders
#Recurcively finds all .jpg files in known_faces
#This is used to trigger the DeepFace face database embedding process
image_candidates = glob.glob(f"{KNOWN_FACES_DIR}/**/*.jpg", recursive=True) + \
                   glob.glob(f"{KNOWN_FACES_DIR}/**/*.jpeg", recursive=True)

#Stop the program if there are no images in the dataset
if not image_candidates:
    print("❌ No face images found in known_faces!")
    exit()


# Picks the first image from the list for initializing the face database
first_image = image_candidates[0]

# Optional: warm-up the database
print("🔄 Finding face representations in known_faces...")
DeepFace.find(
    img_path=first_image,
    db_path=KNOWN_FACES_DIR,
    model_name=MODEL_NAME,
    enforce_detection=False, #avoids crash if no face is detected
    silent=True, #supresses the output
    detector_backend=DETECTOR
)

# Start webcam (index 2 for ny configuration)
cap = cv2.VideoCapture(2)

#if the camera do not open exit with an error
if not cap.isOpened():
    print("❌ Failed to connect to camera stream.")
    exit()

#Confirm webcam started successfully
print("📸 Camera stream connected")

# Use OpenCV for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    ret, frame = cap.read() #Grabs a frame from the webcam #ret is True if successful
    
    #If the webcam did not give a frame, it exits the loop
    if not ret:
        print("⚠️ No frame received")
        break


    #COnvert the webcane image to grayscale which is required for Haar detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detects faces in the grauscale frame
    #scaleFactor: how much the image size is reduced at each image scale
    #minNeighbors: how many neighbors each candidate rectangle should have retain
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    #Loops over all detected face rectangles
    for (x, y, w, h) in faces:
        face_crop = frame[y:y + h, x:x + w] #Crops each face out of the original full-color frame for recognition

        
        try:
            result = DeepFace.find( #use DeepFace to find the closest known peson to the cropped face
                img_path=face_crop,
                db_path=KNOWN_FACES_DIR,
                model_name=MODEL_NAME, # use the specified model
                enforce_detection=False,
                silent=True,
                detector_backend=DETECTOR # and detector
            )

            #if a match is found
            if len(result[0]) > 0:
              top_result = result[0].iloc[0] #extract the top match
              #result[0]: the first Dataframe in the list by DeepFace, it contains all the matches sorted by distance

              #iloc[0]: selects the first row which is the closet match (the face with the closest image)

              distance = top_result['distance'] #how close the embedding is to the known face
              
              threshold = 0.45  #controls match sensitivity
              
              #if the match is within the threshold, it's a known pesrson
              if distance <= threshold:
                  
                  #The name is taken from the folder name
                  identity_path = top_result['identity']
                  person_name = os.path.basename(os.path.dirname(identity_path))
              else:
                  person_name = "Unknown"
            else:
              person_name = "Unknown"

        except:
            person_name = "Unknown"

        # Draw box and label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, person_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Real-Time Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
