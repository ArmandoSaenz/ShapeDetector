import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(3, activation='softmax')
])

#Se cargan los pesos
model.load_weights('modelos_chidos.h5')

#leomenso

cap = cv2.VideoCapture(0)
#cap.set(cv2.CAP_PROP_BRIGHTNESS,5)
#cap.set(cv2.CAP_PROP_CONTRAST,5)
#cap.set(cv2.CAP_PROP_SATURATION, 250)
#cap.set(cv2.CAP_PROP_AUTOFOCUS,1)
cap.set(cv2.CAP_PROP_FOCUS,85)
contador = 1
while True:
    ret, frame = cap.read()



    if (ret):
        grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.blur(grayframe, (10,10))
        _, binaryframe = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)
        countours, _ = cv2.findContours(binaryframe, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in countours:
            M = cv2.moments(contour)
            if M['m00'] != 0:
                centroidx = int(M['m10']/M['m00'])
                centroidy = int(M['m01']/M['m00'])
                centroid = (centroidx, centroidy)
                if centroidx >= 100 and centroidy >= 100:
                    xmin = centroidx - 100
                    ymin = centroidy - 100
                    xmax = centroidx + 100
                    ymax = centroidy + 100

                    figura = binaryframe[ymin:ymax, xmin:xmax]
                    image_normalized = figura/255.0
                    image_normalized = np.expand_dims(image_normalized, axis=-1)
                    image_normalized = np.expand_dims(image_normalized, axis=0)
                    #Se realiza la predicción
                    #print(image_normalized.shape)
                    formas = []
                    formas = image_normalized.shape
                    if formas[1] == 200 and formas[2] == 200:
                        predictions = model.predict(image_normalized)
                        predicted_class = np.argmax(predictions)
                        if xmin >= 0 and xmax >= 0 and ymin >= 0 and ymax >= 0 and (ymin - 10) >= 0:
                        
                            match predicted_class:
                                case 0:
                                    label = "Triangulo"
                                    cv2.putText(frame,label, (xmin, ymin + 50),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                                case 1:
                                    label = "Cuadrado"
                                    cv2.putText(frame,label, (xmin, ymin + 50),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                                case 2:
                                    label = "Circulo"
                                    cv2.putText(frame,label, (xmin, ymin + 50),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                            cv2.rectangle(frame, (xmin + 50, ymin + 50), (xmax-50, ymax-50), (0,255,0),1)
                    
                    #label = f"{predicted_class}"
                    
                
                
                #cv2.rectangle(binaryframe, (xmin,ymin), (xmax,ymax), (255,255,255), 1)
                #cv2.circle(binaryframe, centroid, 2, (0,255,0), -1)
            
        cv2.imshow('Figures',frame)

        keypress = cv2.waitKey(1)
        if (keypress == ord('q') or keypress == ord('Q')):
            #for i in countours:
                #print(i)
                #cv2.imwrite(f"./fototeta/figura_{i}.png",binaryframe[ymin:ymax,xmin:xmax])
            break
        if (keypress == ord('f') or keypress == ord('F')):
            for i in countours:
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    centroidx = int(M['m10']/M['m00'])
                    centroidy = int(M['m01']/M['m00'])
                    centroid = (centroidx, centroidy)
                    xmin = centroidx - 100
                    ymin = centroidy - 100
                    xmax = centroidx + 100
                    ymax = centroidy + 100
                    figura = binaryframe[ymin:ymax,xmin:xmax]
                    cv2.imwrite(f"./fototeta/foto.png",figura)
            #figura = cv2.imread("./fototeta/foto.png", cv2.IMREAD_GRAYSCALE)
            #cv2.imwrite(f"./fototeta/c_{contador}.png",binaryframe[0:200,0:200])
            #contador += 1
            #figura = binaryframe[ymin:ymax,xmin:xmax]
            #image = cv2.imread("./fototeta/s_1.png", cv2.IMREAD_GRAYSCALE)
            #figura_resize = cv2.resize(figura, (200,200))
            #image_normalized = figura_resize / 255.0
                    image_normalized = figura/255.0
                    image_normalized = np.expand_dims(image_normalized, axis=-1)
                    image_normalized = np.expand_dims(image_normalized, axis=0)
            #Se realiza la predicción
                    predictions = model.predict(image_normalized)
                    print(predictions)

            #Se identifica el número con mayor porcentaje
                    predicted_class = np.argmax(predictions)

            #Se imprime la salida
                    print("Clase predicha:", predicted_class)
            #for i in countours:
                #cv2.imwrite(f"./fototeta/figura_{i}.png",binaryframe[ymin:ymax,xmin:xmax])
cv2.destroyAllWindows()
cap.release()