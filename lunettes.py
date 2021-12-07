import numpy as np
import cv2
from PIL import Image

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# la premiere cam est prise par OBS
# on peut utiliser une vid -> e <- o a la place

#cap = cv2.VideoCapture(1)

lunettes = cv2.imread('lunettesMieux.png', cv2.IMREAD_UNCHANGED)
lunettes_ = Image.open('lunettesMieux.png')

image_ = Image.open('image.jpg')

hauteurLunettes, largeurLunettes, _ = lunettes.shape
print('lunettes.shape', lunettes.shape)
print(lunettes)

while 1:
    image = cv2.imread('image.jpg') # mauvais 
    image__ = image_.copy()
    """
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    """
    imageGrise = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces  = face_cascade.detectMultiScale(image, 1.3, 5)
    for (x,y,w,h) in [faces[0]]: # :D
        cv2.rectangle(imageGrise,(x,y),(x+w,y+h),(255,0,0),2)
        faces = imageGrise[y:y+h, x:x+w] # isole les faces
        #roi_color = img[y:y+h, x:x+w]
        
        yeux = eye_cascade.detectMultiScale(imageGrise)

        oeilG, oeilD = yeux
        oeilGX, oeilGY, oeilGL, oeilGH = oeilG
        oeilDX, oeilDY, oeilDL, oeilDH = oeilD

        milieuX = (oeilGX + oeilGL // 2 + oeilDX + oeilDL // 2) // 2
        milieuY = (oeilGY + oeilGH // 2 + oeilDY + oeilDH // 2) // 2
        
        pupilleGX = oeilGX + oeilGL // 2
        pupilleDX = oeilDX + oeilDL // 2
        pupilleGY = oeilGY + oeilGH // 2
        pupilleDY = oeilDY + oeilDH // 2
        
        #print(oeilG)
        for (ex,ey,ew,eh) in yeux: # deux yeux
            cv2.rectangle(image,(ex+ew//2,ey+eh//2),(ex+ew//2+1,ey+eh//2+1),(0,255,0),2)
        hauteurAdaptee = max(oeilGH, oeilDH)
        facteur = hauteurAdaptee / hauteurLunettes
        largeurAdaptee = int(largeurLunettes * facteur)
        print(facteur)
        if(facteur > 0):
            lunettesAdaptees = cv2.resize(lunettes, (largeurAdaptee, hauteurAdaptee))
            print("lunettesAdaptees.shape", lunettesAdaptees.shape)
            x = milieuX - largeurAdaptee // 2 # moche moche
            #image[oeilGY:oeilGY+hauteurAdaptee, x:x+largeurAdaptee] = lunettesAdaptees[:,:,:-1]
            lunettes_ = Image.fromarray(lunettesAdaptees)
            image__.paste(lunettes_,(x,oeilGY),lunettes_)
        image = cv2.cvtColor(np.asarray(image__),cv2.COLOR_RGB2BGR)
        cv2.rectangle(image,(milieuX, oeilGY, 5, oeilGH),(255,0,0), 4)
        cv2.rectangle(image,(oeilGX, oeilGY, oeilGL, oeilGH),(0,255,0), 4)
        cv2.rectangle(image,(oeilDX, oeilDY, oeilDL, oeilDH),(0,255,0), 4)
    cv2.imshow('Lunettes',image)
    k = cv2.waitKey(30) & 0xff
    if k == 27: # 27 -> touche echap
        break

#cap.release()
cv2.destroyAllWindows()

