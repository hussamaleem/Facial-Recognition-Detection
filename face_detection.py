import os
import cv2 as cv
import glob
import numpy as np
import caer

def face_detection():

    
    folders = {}
    path= '/Users/hussamaleem/Downloads/archive/lfw_funneled'
    haar_cas = cv.CascadeClassifier(os.path.join(os.getcwd(),'face_detect.xml'))

    counter = 0  
    
    for folder in os.listdir(path):
        
        images = glob.glob(os.path.join(path, folder,'*.jpg'))
        
        folders[folder] = len(images)
        
    folders_sort = caer.sort_dict(folders,descending=True)
        
    chars = [key for key,value in folders_sort[:10]]

    for i in range(len(chars)):
        images = glob.glob(os.path.join(path, chars[i], '*.jpg'))
        
        saving_path = os.path.join(os.getcwd(), 'Cropped Face Images', chars[i])
        if not os.path.exists(saving_path):
            os.makedirs(saving_path)
        for image_path in images:
            
            image = cv.imread(image_path)
            combined_faces = np.zeros_like(image)
            gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            faces_rect = haar_cas.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=3)

            for (x, y, w, h) in faces_rect:
                cropped_face = image[y:y+h, x:x+w]
                combined_faces[y:y+h, x:x+w] = cropped_face
                
            contours, _ = cv.findContours(cv.cvtColor(combined_faces, cv.COLOR_BGR2GRAY), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
            for contour in contours:
                x, y, w, h = cv.boundingRect(contour)
                cropped = combined_faces[y:y+h, x:x+w]
                cv.imwrite(os.path.join(saving_path, f'face_{counter}.jpg'), cropped)
            counter += 1  

face_detection()
        