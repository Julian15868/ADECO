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
#####################
import matplotlib.pyplot as plt
import cv2
## Comandos
tolerancia = 3 # La que estuvo hasta ahora -> A mayor tolerancia, mas cuadriculado

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
        c =  c = [16, 32, 64, 128] 
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
  ruta_imagen = directorio+"/imgsTIF3/" # EJEMPLO "/imgsTIF/"
  #ruta_imagen = directorio+"/pruebas/Facil/" 
  i = 0
  #print(i) ----------------------------------->>>>
  archivo = os.listdir(ruta_imagen)[i] 
  # Y si no pones el nombre de tu archivo: (Descomentar la linea de abajo)
  #archivo = dma_203_20221215.tif
  imagen =  rasterio.open(ruta_imagen+archivo)

## Ahora hacemos las modificaciones de la imagen para que el modelo la tome bien
imagenL = imagen.read(2) # 2 actual
#imagenL = imagenL/255
"""if np.max(imagenL)>1:
   imagenL/255"""
imagenL = imagenL/(np.max(imagenL)+0.001)
# Tamaño real de la imagen
width = imagenL.shape[0]
height = imagenL.shape[1]

# Hacemos una imagen de zero, expandiendo el tamaño de la imagen original para que sea divisible por 512
imagenL = np.hstack((imagenL, np.zeros((imagenL.shape[0],512-imagenL.shape[1]%512))))
imagenL = np.vstack((imagenL, np.zeros((512-imagenL.shape[0]%512,imagenL.shape[1]))))

import cv2
# Verifica si la lectura de la imagen fue exitosa
# Aplicar el operador de Sobel o Canny para detectar bordes
bordes = cv2.Canny((imagenL*255).astype(np.uint8), 150, 200)  #400 y 500 estaba esto
kernel = np.ones((4, 4), np.uint8)
bordes_ensanchados = cv2.dilate(bordes, kernel, iterations=1)
bordes_ensanchados[0:2,0:] = 255; bordes_ensanchados[-2:,0:] = 255
bordes_ensanchados[0:,0:2] = 255; bordes_ensanchados[0:,-2:] = 255
bordes_ensanchados = bordes_ensanchados.astype(np.float32)/255

#
#plt.imshow(bordes_ensanchados, cmap='gray')#----------------------------------->>>> 
#plt.title('Máscara de Contornos') #----------------------------------->>>> 
#plt.show() #----------------------------------->>>>

pred_all = np.zeros((imagenL.shape[0],imagenL.shape[1])) 
#######################################
# Aca vamos guardando las predicciones de la imagen
for j in range(imagenL.shape[1]//512):
  for i in range(imagenL.shape[0]//512):
    img = torch.tensor([imagenL[512*i:512*(i+1),512*j:512*(j+1)].tolist()]) #IMAGENL->imagen_sin_bordes
    output = modelo_cargado(img.float().unsqueeze(0).to(device))[0]
    pred_mask = (torch.tanh(img.float()) > 0.41).float() # 0.35 bueno #0.45 y 41
    #
    pred_all[512*i:512*(i+1),512*j:512*(j+1)] = np.array(pred_mask.tolist())
pred_all[0:2, :height] = 0
pred_all[width-2:width, :height] = 0
pred_all[:width, 0:2] = 0  
pred_all[:width, height-2:height] = 0
pred_all = pred_all.astype(np.float32)/255
#plt.subplot(1,3,1) #----------------------------------->>>>
#plt.imshow(imagenL,cmap="gray") #----------------------------------->>>>
#plt.subplot(1,3,2) #----------------------------------->>>>
pred_all = np.subtract(pred_all, bordes_ensanchados)
#plt.imshow(pred_all,cmap="gray") #----------------------------------->>>>


"""maskLatente = pred_all.copy()
for i in range(len(pred_all)):
    for j in range(len(pred_all[0])):
        contador = 0
        for x, y in zip([-1, -1, 0, 1, 1, 1, 0, -1], [0, -1, -1, -1, 0, 1, 1, 1]): # todas las direcciones(vecinos)
            if 0 <= i + x < len(pred_all) and 0 <= j + y < len(pred_all[0]):
                if pred_all[i + x][j + y] == 1:
                    contador += 1
        if contador>4:
          maskLatente[i][j] = 1
        else:
          maskLatente[i][j] = 0
pred_all = maskLatente.copy()
"""

###ACTIVAMOS MODEL NUEVO
#red_all = dibujos_sin_lineas
#####################
#Buscamos los contornos
#pred_all = maskLatente #AVER
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
      if contornos[i].area > 100:
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
#print(pol_limpio) # a veces muy largo

try:
  # Unimos los poligonos que se interecan
  """poligonos = []
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
      continue"""
  # Segun la tolerancia se simplifica el poligono
  #plt.subplot(1,3,3) #----------------------------------->>>>
  #plt.imshow(imagenL[0:width,0:height],cmap="gray") #----------------------------------->>>>
  for pols in poligonos_filtrados: # aca cambie multi_poligono.geoms a poligonos_filtrados ->LAST
    try:
      coords = pols.exterior.coords  # Obtener las coordenadas del contorno
      coords_np = np.array(coords)  # A numpy
      simplified_contour = approximate_polygon(coords_np, tolerance=tolerancia) 
      poligonos.append(simplified_contour)
      plt.fill(simplified_contour[:, 1], simplified_contour[:, 0], linewidth=1, color='blue',alpha=0.5) #----------------------------------->>>>
    except:
      continue
  #plt.tight_layout()  #----------------------------------->>>>
  #plt.show() #----------------------------------->>>>
  

  # Lo ponemos en las coordenadas que tiene la meta data de la imagen tif
  poligonos = pol_limpio # Esto no tiene sentido
  pol_x = [poligonos[i][:,0] for i in range(len(poligonos))]
  pol_y = [poligonos[i][:,1] for i in range(len(poligonos))]

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
  #print(archivo)
  if "/" in archivo:
    archivo = archivo.split("/")[-1].replace(".tif","")
  gdf.to_file(directorio+"/polygons/"+archivo+"_polygon.geojson", driver="GeoJSON")
  print("Se ha guardado el poligono")

except:
  print("No se encontro poligono")
