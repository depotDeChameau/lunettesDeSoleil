import numpy as np
import cv2
from PIL import Image

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


image = cv2.imread('starsDEntreprise.jpg') # np.array
lunettes = cv2.imread('lunettesMieux.png', cv2.IMREAD_UNCHANGED)

print(lunettes.shape)

def pupille(visage, oeil):
    print(visage, oeil)
    vX, vY, *_ = visage
    ex, ey, ew, eh = oeil
    
    return vX + ex + ew // 2, vY + ey + eh // 2

def f(image : np.array, lunettes : np.array) -> np.array :
    print("Dimension de l'image contenant les visages :", image.shape)
    print("Dimension de l'image contenant les lunettes :", lunettes.shape)
    
    imageGrise = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    visages  = face_cascade.detectMultiScale(image, 1.3, 5)

    hauteurLunettes, largeurLunettes, _ = lunettes.shape

    for visage in visages:
        vX,vY,vL,vH = visage
        cv2.rectangle(image,(vX,vY),(vX+vL,vY+vH),(255,0,0),2)
        imageVisage = imageGrise[vY:vY+vH, vX:vX+vL] # isole le visage
        
        yeux = eye_cascade.detectMultiScale(imageVisage) # detection d'yeux uniquement dans une zone de visage
        print("len(yeux) =", len(yeux))
        if len(yeux) >= 2: # si au moins deux yeux ont ete detectes
            oeilG, oeilD, *_ = yeux # on ne garde que les deux premiers
            for oeil in oeilG, oeilD: # 2 iterations ^^
                ex,ey,ew,eh = oeil
                cv2.rectangle(image,(vX+ex,vY+ey),(vX+ex+ew,vY+ey+eh),(0,255,0),2)
            
                cv2.circle(image,pupille(visage, oeil),2,(0,0,255),2) # BGR
  
                hauteurAdaptee = eh
                facteur = hauteurAdaptee / hauteurLunettes
                largeurAdaptee = int(largeurLunettes * facteur)

                if(facteur > 0):
                    lunettesAdaptees = cv2.resize(lunettes, (largeurAdaptee, hauteurAdaptee))
                    lunettesAdapteesPIL = Image.fromarray(lunettes)
                    #imagePIL.paste(lunettesAdapteesPIL, (x, oeilGY), lunettesAdapteesPIL)

        imagePIL = Image.fromarray(image)

    return np.asarray(imagePIL)
    
resultat = f(image, lunettes)
    
cv2.imshow('Lunettes', resultat)
cv2.waitKey(0) # waits until a key is pressed
cv2.destroyAllWindows() # destroys the window showing image


def traiterImage():
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
            # paste gere la transparence
            image__.paste(lunettes_,(x,oeilGY),lunettes_)
        image = cv2.cvtColor(np.asarray(image__),cv2.COLOR_RGB2BGR)
        cv2.rectangle(image,(milieuX, oeilGY, 5, oeilGH),(255,0,0), 4)
        cv2.rectangle(image,(oeilGX, oeilGY, oeilGL, oeilGH),(0,255,0), 4)
        cv2.rectangle(image,(oeilDX, oeilDY, oeilDL, oeilDH),(0,255,0), 4)
    cv2.imshow('Lunettes',image)


