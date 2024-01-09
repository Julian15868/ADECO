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
## Comandos
tolerancia = 5 # La que estuvo hasta ahora -> A mayor tolerancia, mas cuadriculado

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
  # Si queres tomar por ejemplo la imagen 13 de tal carpeta lo pones asi:
  i = 0
  archivo = os.listdir(ruta_imagen)[i] 
  # Y si no pones el nombre de tu archivo: (Descomentar la linea de abajo)
  #archivo = dma_203_20221215.tif
  imagen =  rasterio.open(ruta_imagen+archivo)

## Ahora hacemos las modificaciones de la imagen para que el modelo la tome bien
imagenL = imagen.read(2)
imagenL = imagenL/255
# Tamaño real de la imagen
width = imagenL.shape[0]
height = imagenL.shape[1]
# Hacemos una imagen de zero, expandiendo el tamaño de la imagen original para que sea divisible por 512
imagenL = np.hstack((imagenL, np.zeros((imagenL.shape[0],512-imagenL.shape[1]%512))))
imagenL = np.vstack((imagenL, np.zeros((512-imagenL.shape[0]%512,imagenL.shape[1]))))
pred_all = np.zeros((imagenL.shape[0],imagenL.shape[1])) 

# Aca vamos guardando las predicciones de la imagen
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
    pred_all[512*i:512*(i+1),512*j:512*(j+1)] = np.array(pred_mask.tolist())

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
  """plt.figure(figsize=(6, 6))
  for i in range(len(pol_x)):
    plt.fill(pol_x[i], pol_y[i], alpha=0.3)
  plt.show()"""
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
