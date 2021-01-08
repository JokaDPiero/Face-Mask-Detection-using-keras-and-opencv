import cv2
from keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Capture the Video from webcam..
cap = cv2.VideoCapture(0)

# load haarcascade file
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

#Loading the pre-trained model to predict mask or no mask
model = load_model("mask_recog_ver2 (1).h5")

results={
    0:'Without Mask',
    1:'Mask'
}

color={
    0:(0,0,255),#red
    1:(0,255,0) #green
}

# Infinte Loop
while True:

	# Read the Webcam Image
    ret, frame  = cap.read()
    
    frame=cv2.flip(frame,1) #(not mirror image)
    
    # If not able to read image
    if ret == False:
        continue


	# Detect faces on the current frame
    faces = face_cascade.detectMultiScale(frame)
    
    face_list=[]
    preds=[]

	# Plot rectangle around all the faces
    for (x,y,w,h) in faces:
        
        face_img = frame[y:y+h, x:x+w]
        #as the model is trained in RGB images, we convert the frames to RGB
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        #as the model is trained in images resesized to 224x224 px , we resize the frames
        face_img = cv2.resize(face_img, (224, 224))  
        #converting to float32
        face_img = img_to_array(face_img)
        #as face_img is 2-D we increase its dimension to 3D to make it compatible with model
        face_img = np.expand_dims(face_img, axis=0)
        #as the frames are batches of images i.e. captured continuously we normalise it
        face_img =  preprocess_input(face_img)
        #storing the continuous output
        face_list.append(face_img)

        #predicting whether the capture contains mask or not
        if len(face_list)>0:
            preds = model.predict(face_list)
        
        #probability of wearing a mask and vice versa
        for pred in preds:
            (mask,withoutMask) = pred 
            
        #calculating mask detection percentage
        percentage=max(mask, withoutMask) * 100
        
        if mask > withoutMask:
            idx=1
        else:
            idx=0
        
        #creating the text format
        label = "{}: {:.2f}%".format(results[idx], percentage)
        
        #putting the text below the rectangle
        cv2.putText(frame, label, (x, y+h+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color[idx], 2)
        #creating the rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h),color[idx], 2)
        


	# Display the frame
    cv2.imshow("Video Frame", frame)
	

	# Find the key pressed
    key_pressed = cv2.waitKey(1) & 0xFF

	# If keypressed is q then quit the screen
    if key_pressed == ord('q'):
        break
    

# release the camera resource and destroy the window opened.
cap.release()
cv2.destroyAllWindows()
