#Based on "Hashtag Programação" on YouTube
import cv2
import mediapipe as mp

#Run OpenCV and Mediapipe
#-Define your cam by index
cam = cv2.VideoCapture(0)
#-Define and run your reccon tools
face_reccon = mp.solutions.face_detection
recogniser = face_reccon.FaceDetection()
#-Define your drawing
draw = mp.solutions.drawing_utils

while True:
    #Read cam info
    verifier, frame = cam.read()
    if not verifier:
        break
    
    #Reccon faces
    list_faces = recogniser.process(frame)

    if list_faces.detections:
        for face in list_faces.detections:
            #Draw on images
            draw.draw_detection(frame, face)
        
    #Show cam image
    cv2.imshow("Faces on cam", frame)

    #Press Esc (key 27) to stop, and add a timer on image showing
    if cv2.waitKey(5) == 27:
        break

#Shutdown your Camera    
cam.release()
cv2.destroyAllWindows()
