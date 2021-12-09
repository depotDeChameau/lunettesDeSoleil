import numpy as np
import cv2
from PIL import Image

def pupille(visage, oeil):
    vX, vY, *_ = visage
    ex, ey, ew, eh = oeil
    
    return vX + ex + ew // 2, vY + ey + eh // 2

def milieu(oeilG, oeilD):
    x, y, l, h = oeilG
    X, Y, L, H = oeilD
    
    return (x+l//2+X+L//2)//2, (y+h//2+Y+H//2)//2 

def f(image : np.array, lunettes : np.array, dbg : bool = False) -> np.array :
    #print("Dimension de l'image contenant les visages :", image.shape)
    #print("Dimension de l'image contenant les lunettes :", lunettes.shape)
    
    imageGrise = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imagePIL = Image.fromarray(image)
    visages  = face_cascade.detectMultiScale(image, 1.3, 5)

    hauteurLunettes, largeurLunettes, _ = lunettes.shape

    for (i,visage) in enumerate(visages):
        vX,vY,vL,vH = visage
        if dbg:
            cv2.rectangle(image,(vX,vY),(vX+vL,vY+vH),(255,0,0),1)
            cv2.putText(image,str(i),(vX,vY),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        imageVisage = imageGrise[vY:vY+vH*3//5, vX:vX+vL] # isole le visage
        
        yeux = eye_cascade.detectMultiScale(imageVisage, 1.3,5) # detection d'yeux uniquement dans une zone de visage
        #print("len(yeux) =", len(yeux))
        if len(yeux) >= 2: # si au moins deux yeux ont ete detectes
            oeilG, oeilD, *_ = yeux # on ne garde que les deux premiers

            if tuple(oeilD) < tuple(oeilG) : # si jamais les yeux sont pas dans le bon sens
                oeilG, oeilD = oeilD, oeilG
            
            #print("oeilG =", oeilG, "oeilD =", oeilD)
            oeilGX, oeilGY, oeilGL, oeilGH = oeilG
            oeilDX, oeilDY, oeilDL, oeilDH = oeilD

            if dbg:
                cv2.rectangle(image,(vX+oeilGX,vY+oeilGY),(vX+oeilGX+oeilGL,vY+oeilGY+oeilGH),(0,255,0),1)
                cv2.circle(image,pupille(visage, oeilG),2,(0,0,255),1) # BGR

                cv2.rectangle(image,(vX+oeilDX,vY+oeilDY),(vX+oeilDX+oeilDL,vY+oeilDY+oeilDH),(0,0,255),1)
                cv2.circle(image,pupille(visage, oeilD),2,(0,0,255),1) # BGR
            
            hauteurAdaptee = int(0.65*max(oeilGH, oeilDH))
            facteur = hauteurAdaptee / hauteurLunettes
            largeurAdaptee = int(facteur * largeurLunettes)
            
            #print("largeurAdaptee =", largeurAdaptee, "hauteurAdaptee =", hauteurAdaptee)
            
            lunettesAdaptees = cv2.resize(lunettes, (largeurAdaptee, hauteurAdaptee)) # deformation des lunettes toleree
            lunettesAdapteesPIL = Image.fromarray(lunettesAdaptees)

            imagePIL = Image.fromarray(image)
            mX, mY = milieu(oeilG, oeilD)
            imagePIL.paste(lunettesAdapteesPIL, (vX+mX-largeurAdaptee//2, vY+mY-int(0.35*hauteurAdaptee)), lunettesAdapteesPIL)

            image = np.asarray(imagePIL)
        else: # si les yeux des visages de sont pas reconnus
            imagePIL = Image.fromarray(image)
            
    return np.asarray(imagePIL)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


image = cv2.imread('jacquouille.jpg') # np.array
lunettes = cv2.imread('lunettesMieux.png', cv2.IMREAD_UNCHANGED)

print(lunettes.shape)

#cap = cv2.VideoCapture('greta.mp4')
cap = cv2.VideoCapture(0)

compteur = 0

while True:
    #cv2.imshow('Lunettes', resultat)
    ret, imageVideo = cap.read()
    if True or compteur % 5 == 0:
        resultat = f(imageVideo, lunettes, dbg=False) 
    cv2.imshow('Greta', resultat)
    if cv2.waitKey(1) == ord('q'):
        break
    compteur += 1
#Image.fromarray(resultat).show()

cap.release()
cv2.destroyAllWindows() # destroys the window showing image
