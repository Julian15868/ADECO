## Librerias
#Comunes
import pandas as pd
import numpy as np
import os
import sys
#Para imagenes y polygonos
from shapely.geometry import Polygon, MultiPolygon
from PIL import Image
import rasterio
import geojson
from skimage import measure
from skimage.measure import find_contours
from skimage.measure import approximate_polygon
#Para el modelo
import torch
import torch.nn.functional as F
from joblib import dump, load
# Nuevos
import json
import geopandas as gpd
import matplotlib.pyplot as plt
import cv2
#####################
## Comandos
tolerancia = 2 # La que estuvo hasta ahora -> A mayor tolerancia, mas cuadriculado

## Para que el modelo que carguemos funcione, necesitamos saber como es la estructura de la red neuronal, para eso agregamos lo siguiente
def conv3x3_bn(ci, co):
    return torch.nn.Sequential(
        torch.nn.Conv2d(ci, co, 3, padding=1),
        torch.nn.BatchNorm2d(co),
        torch.nn.ReLU(inplace=True)
    )

def encoder_conv(ci, co):
  return torch.nn.Sequential(
        torch.nn.MaxPool2d(2),
        conv3x3_bn(ci, co),
        conv3x3_bn(co, co),
    )

class deconv(torch.nn.Module):
    def __init__(self, ci, co):
        super(deconv, self).__init__()
        self.upsample = torch.nn.ConvTranspose2d(ci, co, 2, stride=2)
        self.conv1 = conv3x3_bn(ci, co)
        self.conv2 = conv3x3_bn(co, co)

    # recibe la salida de la capa anterior y la salida de la etapa
    # correspondiente del encoder
    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX, 0, diffY, 0))
        # concatenamos los tensores
        x = torch.cat([x2, x1], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x
# Este es el modelo
class UNet(torch.nn.Module):
    def __init__(self, n_classes=2, in_ch=1): # Cambiamos clases a 2
        super().__init__()

        # lista de capas en encoder-decoder con número de filtros
        c =  c = [32, 64, 128, 256] 
        # primera capa conv que recibe la imagen
        self.conv1 = torch.nn.Sequential(
          conv3x3_bn(in_ch, c[0]),
          conv3x3_bn(c[0], c[0]),
        )
        # capas del encoder
        self.conv2 = encoder_conv(c[0], c[1])
        self.conv3 = encoder_conv(c[1], c[2])
        self.conv4 = encoder_conv(c[2], c[3])

        # capas del decoder
        self.deconv1 = deconv(c[3],c[2])
        self.deconv2 = deconv(c[2],c[1])
        self.deconv3 = deconv(c[1],c[0])

        # útlima capa conv que nos da la máscara
        self.out = torch.nn.Conv2d(c[0], n_classes, 3, padding=1)

    def forward(self, x):
        # encoder
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x = self.conv4(x3)
        # decoder
        x = self.deconv1(x, x3)
        x = self.deconv2(x, x2)
        x = self.deconv3(x, x1)
        x = self.out(x)
        return x
    
##################### 
# Funcion para la transformar en latitud y longitud (esto es con la informacion de la imagen.tif que tiene metadata)
def toCoords(image,geo_coord_x,geo_coord_y):
  a,b,tx,c,d,ty,  no,im,porta = image.meta["transform"] # El "no importa" no lo requerimos.
  for ap in range(len(geo_coord_x)): #
    x_nuevo = [((a * x) + (b * y) + tx) for x,y in zip(geo_coord_y,geo_coord_x)]
    y_nuevo = [((c * x) + (d * y) + ty) for x,y in zip(geo_coord_y,geo_coord_x)]
  return x_nuevo, y_nuevo

# Cargamos el modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelo_cargado = torch.load("modeloDeteccion.pth",map_location=torch.device('cpu'))  #"modeloDeteccion.pth" -> ACA SE ACTUALIZA CON EL NOMBRE DEL ULTIMO MODELO RECIBIDO

## Cargamos la imagen estando en el directorio actual
directorio = os.getcwd() 
# Primera forma, pasando como argumento
if len(sys.argv)>=2:
   archivo = sys.argv[1]
   imagen =  rasterio.open(directorio+"//"+archivo)
   print(directorio+"/"+archivo)
else:
  # Segunda forma poniendolo en el codigo
  ruta_imagen = directorio+"/imgsTIF/" # EJEMPLO "/imgsTIF/"
  #ruta_imagen = directorio+"/pruebas/Facil/" # EJEMPLO "/imgsTIF/"
  #ruta_imagen = directorio+"/imgsTIF3/"
  # Si queres tomar por ejemplo la imagen 13 de tal carpeta lo pones asi:
  ####A BORRAR
  #i = np.random.randint(0,379)
  i = 10
  print(i)
  ####
  
  archivo = os.listdir(ruta_imagen)[i] 
  # Y si no pones el nombre de tu archivo: (Descomentar la linea de abajo)
  #archivo = dma_203_20221215.tif
  imagen =  rasterio.open(ruta_imagen+archivo)

## Ahora hacemos las modificaciones de la imagen para que el modelo la tome bien
imagenL = imagen.read(2) # 2 actual
#imagenL = imagenL/255
imagenL = imagenL/np.max(imagenL)
# Tamaño real de la imagen
width = imagenL.shape[0]
height = imagenL.shape[1]


# Hacemos una imagen de zero, expandiendo el tamaño de la imagen original para que sea divisible por 512
imagenL = np.hstack((imagenL, np.zeros((imagenL.shape[0],512-imagenL.shape[1]%512))))
imagenL = np.vstack((imagenL, np.zeros((512-imagenL.shape[0]%512,imagenL.shape[1]))))
pred_all = np.zeros((imagenL.shape[0],imagenL.shape[1])) 

#######################################
#  PRIMERO DETECTAMOS LOS BORDES SIN HABER HECHO NINGUNA TRANSFORMACION!
bordes = cv2.Canny((imagenL*255).astype(np.uint8) , 120, 150)  # Ajusta los umbrales según tu imagen (100,150)
# Definir el kernel para dilatación
kernel = np.ones((4, 4), np.uint8) #########################(4,4)
# Aplicar dilatación a los bordes detectados
bordes_dilatados = cv2.dilate(bordes, kernel, iterations=1)
# Mostrar los bordes detectados y dilatados
#cv2.imshow('Bordes dilatados', bordes_dilatados)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

imagen_sin_bordes = cv2.bitwise_and(imagenL, imagenL, mask=cv2.bitwise_not(bordes_dilatados))
#plt.imshow(imagen_sin_bordes,cmap="gray")
#plt.show()
#####################################

## IMAGEN DESVIO STANDARD # ELIMINAR->>_>_>--<
#sobreTierraLista = np.array([x for row in imagenL for x in row if x != 0])
sobreTierraLista = np.array([x for row in imagen_sin_bordes for x in row if x != 0]) # DESVIO STGANDARD EN LA IMAGEN SIN BORDES

media = np.mean(sobreTierraLista)
std = np.std(sobreTierraLista)
##print(media,"<<>>",std,"<<>>",round(100*std/media,1))
##print(len(sobreTierraLista))

### Imagen Difuminada
mask = np.where(imagenL != 0, 255, 0).astype('uint8')
imagen_difuminada = cv2.blur(imagenL, (13, 13))
imagen_difuminada = cv2.bitwise_and(imagen_difuminada, imagen_difuminada, mask=mask)

#cv2.imshow('Imagen Difuminada', imagen_difuminada)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
####

#STAD
if std < 0.1:
   factor = 1.7
else:
   factor = 0.1
imagenNueva = [[1 if x>(media+factor*std) else 0 for x in row] for row in imagen_difuminada] # 0.5 es bueno
#imagenNueva = [[1 if x>(media+3*std) else 0 for x in row] for row in imagenL] # PARa imagenes <10
##plt.subplot(1,3,1)
##plt.imshow(imagenL,cmap="gray")
#plt.subplot(1,4,2)
#plt.imshow(imagenNueva,cmap="gray")
imagenLatente = imagenNueva.copy() # Une puntos sueltos si tienen cierta cantida de vecinos

for i in range(len(imagenNueva)):
    for j in range(len(imagenNueva[0])):
        contador = 0
        for x, y in zip([-1, -1, 0, 1, 1, 1, 0, -1], [0, -1, -1, -1, 0, 1, 1, 1]): # todas las direcciones(vecinos)
            if 0 <= i + x < len(imagenNueva) and 0 <= j + y < len(imagenNueva[0]):
                if imagenNueva[i + x][j + y] == 1:
                    contador += 1
        if contador>4:
           imagenLatente[i][j] = 1

#plt.subplot(1,4,3)
imagenNueva = imagenLatente
#plt.imshow(imagenNueva,cmap="gray")
      
#plt.show()
##############
################ A ELIMINAR TAMBIEN

import cv2
# Convertir la matriz en una imagen binaria (0 o 255)
matriz_imagen = np.array(imagenNueva).astype(np.uint8) * 255
# Aplicar detector de bordes Canny
edges = cv2.Canny(matriz_imagen, 0, 150)
# Aplicar transformada de Hough para detectar líneas
##lines = cv2.HoughLinesP(edges, 2, np.pi / 180, threshold=80, minLineLength=15, maxLineGap=100) # cambiar el segundo argumento cambia todo, el maxilineGap tambien, 2 np,60,15,100
#lines = cv2.HoughLinesP(edges, 1, 10*np.pi / 180, threshold=100, minLineLength=30, maxLineGap=70) # cambiar el segundo argumento cambia todo, el maxilineGap tambien, 2 np,60,15,100 # para imagenes <13

lines = cv2.HoughLinesP(edges, 2, np.pi / 1500, threshold=71, minLineLength=44, maxLineGap=12)


# Crear una imagen en blanco y negro del mismo tamaño que la original
lineas_eliminar = np.zeros_like(matriz_imagen)
# Dibujar las líneas detectadas en la máscara
if lines is not None:
    for linea in lines:
        x1, y1, x2, y2 = linea[0]
        cv2.line(lineas_eliminar, (x1, y1), (x2, y2), 255, 7)  # Dibujar la línea en blanco # (255,8)
# Invertir la máscara de líneas (donde hay líneas será 0 y viceversa)
lineas_eliminar = cv2.bitwise_not(lineas_eliminar)
# Aplicar la máscara para eliminar las líneas de la imagen original
dibujos_sin_lineas = cv2.bitwise_and(matriz_imagen, lineas_eliminar)

##dibujos_sin_lineas = matriz_imagen


# Operaciones de morfología para eliminar puntos sueltos
kernel = np.ones((5, 5), np.uint8) #4,4 5.5 ## (9,9)
#dibujos_sin_lineas = cv2.morphologyEx(dibujos_sin_lineas, cv2.MORPH_CLOSE, kernel)
dibujos_sin_lineas = cv2.morphologyEx(dibujos_sin_lineas, cv2.MORPH_OPEN, kernel)

dibujos_sin_lineas = cv2.bitwise_and(dibujos_sin_lineas, dibujos_sin_lineas, mask=cv2.bitwise_not(bordes_dilatados))############------------------
# Mostrar el resultado
##plt.subplot(1,3,2)
##plt.imshow(dibujos_sin_lineas,cmap="gray")
#plt.show()





############## Desactivamos modelo anterior
"""# Aca vamos guardando las predicciones de la imagen
for j in range(imagenL.shape[1]//512):
  for i in range(imagenL.shape[0]//512):
    img = torch.tensor([imagenL[512*i:512*(i+1),512*j:512*(j+1)].tolist()])
    output = modelo_cargado(img.float().unsqueeze(0).to(device))[0]
    ##Agregado
    output_probabilities = F.softmax(output, dim=0) # Agregado
    max_prob, pred_class = torch.max(output_probabilities, dim=0) # Agregado
    mask_confianza = max_prob > 0.99 #0.5 #0.99 bien # Agregado
    ##
    pred_mask = torch.argmax(output, axis=0)
    pred_mask = torch.zeros_like(pred_class)  # Crear un tensor de ceros del mismo tamaño que pred_class # Agregado
    pred_mask[mask_confianza] = pred_class[mask_confianza]  # Asignar las predicciones que superan el umbral # Agregado
    #
    pred_all[512*i:512*(i+1),512*j:512*(j+1)] = np.array(pred_mask.tolist())"""

#### AGREGADO para visualizar (Se eliminara en la ultima version)
"""#Img Real
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.subplot(1,2,1)
plt.imshow(imagenL[0:width,0:height],cmap="gray")
#Img Pred 
plt.subplot(1,2,2)
plt.imshow(pred_all[0:width,0:height],cmap="gray")
plt.show()"""
###############

###ACTIVAMOS MODEL NUEVO
pred_all = dibujos_sin_lineas
#####################
#Buscamos los contornos
contornos = measure.find_contours(pred_all[0:width,0:height], 0)#pred_mask
contornos = [Polygon(p) for p in contornos]
poligonos_filtrados = []
for i, p1 in enumerate(contornos):
    contiene_otro = False
    for j, p2 in enumerate(contornos):
        if i != j and p2.contains(p1):  # Verificar si p1 esta contenido en p2
            contiene_otro = True
            break
    if not contiene_otro:  # Si p1 no esta contenido en ningun otro, lo agregamos a la lista
        poligonos_filtrados.append(contornos[i])#poligonos

# Dibujamos los contornos encontrados
poligonos = []
for contour in poligonos_filtrados:
  coords = contour.exterior.coords  # Obtener las coordenadas del contorno
  coords_np = np.array(coords)  # Anumpy
  simplified_contour = approximate_polygon(coords_np, tolerance=tolerancia)
  poligonos.append(simplified_contour)

#####################
# Limpiamos el poligono (descartamos los poligonos que tengan menos de 3 vertices)
pol_limpio = [pol for pol in poligonos if len(pol) > 2]

try:
  # Unimos los poligonos que se interecan
  poligonos = []
  multi_poligono = MultiPolygon()
  pol_unidos = poligonos_filtrados[0] #Empezamos viendo el primero
  pol_descartados = []
  poligonos_fitrados = pol_limpio
  for i in range(1,len(poligonos_filtrados)):
    if pol_unidos.intersects(poligonos_filtrados[i]):
      pol_unidos.union(poligonos_filtrados[i])
    else:
      pol_descartados.append((poligonos_filtrados[i]))
  # Unimos los descartados (Los que no tenian interseccion) al poligono general anterior
  multi_poligono = multi_poligono.union(pol_unidos)
  for h in range(len(pol_descartados)):
    try:
      multi_poligono = multi_poligono.union(pol_descartados[h])
    except:
      continue
  # Segun la tolerancia se simplifica el poligono
  for pols in multi_poligono.geoms:
    try:
      coords = pols.exterior.coords  # Obtener las coordenadas del contorno
      coords_np = np.array(coords)  # A numpy
      simplified_contour = approximate_polygon(coords_np, tolerance=tolerancia) 
      poligonos.append(simplified_contour)
    except:
      continue

  # Lo ponemos en las coordenadas que tiene la meta data de la imagen tif
  poligonos = pol_limpio
  pol_x = [poligonos[i][:,0] for i in range(len(poligonos))]
  pol_y = [poligonos[i][:,1] for i in range(len(poligonos))]

  #### AGREGADO para visualizar (Se eliminara en la ultima version)
  ##plt.subplot(1,3,3)
  plt.imshow(imagenL.T,cmap="gray")
  for i in range(len(pol_x)):
    plt.fill(pol_x[i], pol_y[i], alpha=0.8)
  plt.show()
  #########

  # Transformamos para que vaya a una coordenada en el mapa
  poligono_x,poligono_y = toCoords(imagen,pol_x,pol_y)[0],toCoords(imagen,pol_x,pol_y)[1]

  # Ponemos como multipolygon de la forma en que estaba en el geojson para asi guardarlo
  polygons = []
  for i in range(len(pol_x)):
      polygon = [elemento for sublist in [(xx.tolist(),yy.tolist()) for xx,yy in zip(poligono_x[i],poligono_y[i])] for elemento in sublist]
      polygons.append(polygon)
  #polygons = geojson.MultiPolygon(polygons)
  # Creamos  DataFrame usando la lista de poligonos
  polygons =  [[[sublista[i:i + 2] for i in range(0, len(sublista), 2)]]for sublista in polygons]
  # SOLUCION PARCIAL:
  data = {
    "type": "Feature",
    "geometry": {
      "type": "MultiPolygon",
      "coordinates": polygons,
    },
    "properties": {
      "name": "Ejemplo punto"
    }
  }
  ## Creacion GEOJSON
  geojson_data = json.dumps(data)
  # Crea un GeoDataFrame a partir del GeoJSON
  gdf = gpd.GeoDataFrame.from_features([json.loads(geojson_data)])
  #Guardamos en la carpeta polygons
  print(archivo)
  if "/" in archivo:
     archivo = archivo.split("/")[-1].replace(".tif","")
  gdf.to_file(directorio+"/polygons/"+archivo+"_polygon.geojson", driver="GeoJSON")


except:
  print("No se encontro poligonos")
