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
import time
## Comandos
tolerancia = 2  # La que estuvo hasta ahora -> A mayor tolerancia, mas cuadriculado

## Para que el modelo que carguemos funcione, necesitamos saber como es la estructura de la red neuronal, para eso agregamos lo siguiente
# Modelo y sus componentes
##############################################################
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
    
###### 
import torchvision
import torch
import torchvision.models as models

# Crear el modelo ResNet con pesos preentrenados utilizando el nuevo parámetro 'weights'
resnet_weights = models.ResNet18_Weights.IMAGENET1K_V1
model = models.resnet18(weights=resnet_weights)

class out_conv(torch.nn.Module):
    def __init__(self, ci, co, coo):
        super(out_conv, self).__init__()
        self.upsample = torch.nn.ConvTranspose2d(ci, co, 2, stride=2)
        self.conv = conv3x3_bn(ci, co)
        self.final = torch.nn.Conv2d(co, coo, 1)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX, 0, diffY, 0))
        x = self.conv(x1)
        x = self.final(x)
        return x

class UNetResnet(torch.nn.Module):
    def __init__(self, n_classes=2, in_ch=3):
        super().__init__()

        #self.encoder = torchvision.models.resnet18(pretrained=True)
        self.encoder = torchvision.models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        if in_ch != 3:
          self.encoder.conv1 = torch.nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.deconv1 = deconv(512,256)
        self.deconv2 = deconv(256,128)
        self.deconv3 = deconv(128,64)
        self.out = out_conv(64, 64, n_classes)

    def forward(self, x):
        #x_in = torch.tensor(x.clone())
        x_in = x.clone().detach()
        x = self.encoder.relu(self.encoder.bn1(self.encoder.conv1(x)))
        x1 = self.encoder.layer1(x)
        x2 = self.encoder.layer2(x1)
        x3 = self.encoder.layer3(x2)
        x = self.encoder.layer4(x3)
        x = self.deconv1(x, x3)
        x = self.deconv2(x, x2)
        x = self.deconv3(x, x1)
        x = self.out(x, x_in)
        return x
##############################################################

## Segunda parte del codigo: Se recibe una imagen con extension tif y se genera una prediccion en el formato requerido
# Funcion para la transformar en latitud y longitud (esto es con la informacion de la imagen.tif que tiene metadata)
def toCoords(image,geo_coord_x,geo_coord_y):
  a,b,tx,c,d,ty,  no,im,porta = ~image.meta["transform"] # El "no importa" no lo requerimos.
  for ap in range(len(geo_coord_x)): #
    x_nuevo = [((a * x) + (b * y) + tx) for x,y in zip(geo_coord_y,geo_coord_x)]
    y_nuevo = [((c * x) + (d * y) + ty) for x,y in zip(geo_coord_y,geo_coord_x)]
  return x_nuevo, y_nuevo

# Cargamos el modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelo_name = "modeloDeteccion.pth"
modelo_cargado = torch.load(modelo_name,map_location=torch.device('cpu'))  #"modeloDeteccion.pth" -> ACA SE ACTUALIZA CON EL NOMBRE DEL ULTIMO MODELO RECIBIDO ModeloCONTRANSFER.pth

# Lectura de imagen 
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
  i = 15 # Poner el archivo que quieras
  archivo = os.listdir(ruta_imagen)[i] 
  imagen =  rasterio.open(ruta_imagen+archivo)

def normalize2(r,g,b):
        array_r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)  # Manejar NaNs e infinitos
        array_r_min, array_r_max = array_r.min(), array_r.max()
        array_g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)  # Manejar NaNs e infinitos
        array_g_min, array_g_max = array_g.min(), array_g.max()
        array_b = np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)  # Manejar NaNs e infinitos
        array_b_min, array_b_max = array_b.min(), array_b.max()

        array_min = min(array_r_min,array_g_min,array_b_min)
        array_max = max(array_r_max,array_g_max,array_b_max)

        if array_min == array_max:  # Evitar división por cero
            return np.zeros(array_r.shape, dtype=np.uint8),np.zeros(array_r.shape, dtype=np.uint8),np.zeros(array_r.shape, dtype=np.uint8)
      
        n_r = ((r - array_min) / (array_max - array_min) * 255).astype(np.uint8)
        n_g = ((g - array_min) / (array_max - array_min) * 255).astype(np.uint8)
        n_b = ((b - array_min) / (array_max - array_min) * 255).astype(np.uint8)
        return n_r,n_g,n_b

r = imagen.read(6) #
g = imagen.read(4) #
b = imagen.read(2) #
r, g, b = normalize2(r, g, b)
imagenL = np.dstack((r,g,b))

# Preparacion y prediccion
# Tamaño real de la imagen
width = imagenL.shape[0]
height = imagenL.shape[1]

# Hacemos una imagen de zero, expandiendo el tamaño de la imagen original para que sea divisible por 512
imagenL = np.hstack((imagenL, np.zeros((imagenL.shape[0],512-imagenL.shape[1]%512,imagenL.shape[-1]), dtype=np.uint8)))
imagenL = np.vstack((imagenL, np.zeros((512-imagenL.shape[0]%512,imagenL.shape[1],imagenL.shape[-1]), dtype=np.uint8)))

pred_all = np.zeros((imagenL.shape[0],imagenL.shape[1])) 
# Predicciones
# Aca vamos guardando las predicciones de la imagen
for j in range(imagenL.shape[1]//512):
  for i in range(imagenL.shape[0]//512):
    img = torch.tensor([imagenL[512*i:512*(i+1),512*j:512*(j+1),:].tolist()]).permute(0,3,1,2) #IMAGENL->imagen_sin_bordes
    output = modelo_cargado(img.float().to(device))[0] #.unsqueeze(0)
    pred_mask = (torch.tanh(output.cpu()) > 0.65).float() #(torch.tanh(output.cpu()) > 0).float()
    pred_mask = pred_mask[1]
    pred_all[512*i:512*(i+1),512*j:512*(j+1)] = np.array(pred_mask.tolist())

pred_all[0:2, :height] = 0
pred_all[width-2:width, :height] = 0
pred_all[:width, 0:2] = 0  
pred_all[:width, height-2:height] = 0
pred_all = pred_all.astype(np.float32)#/255

# Busqueda de contornos (a partir de la prediccion) y aproximacion por poligonos
tiempo_inicio = time.time()
contornos = measure.find_contours(pred_all[0:width,0:height], 0)
contornos = [Polygon(p) for p in contornos]
poligonos_filtrados = []
for i, p1 in enumerate(contornos):
    tiempo_ejecucion = time.time() - tiempo_inicio
    if tiempo_ejecucion > 60: break
    contiene_otro = False
    for j, p2 in enumerate(contornos):
        tiempo_ejecucion = time.time() - tiempo_inicio
        if tiempo_ejecucion > 60: break
        if i != j and p2.contains(p1):  # Verificar si p1 esta contenido en p2
            contiene_otro = True
            break
    if not contiene_otro:  # Si p1 no esta contenido en ningun otro, lo agregamos a la lista
      if contornos[i].area > 0: # Que el area sea mayo a cero (no lo descartamos por ahora)
        poligonos_filtrados.append(contornos[i])

# Dibujamos los contornos encontrados
poligonos = []
for contour in poligonos_filtrados:
  tiempo_ejecucion = time.time() - tiempo_inicio
  if tiempo_ejecucion > 60: break
  coords = contour.exterior.coords  # Obtener las coordenadas del contorno
  coords_np = np.array(coords)  # Anumpy
  simplified_contour = approximate_polygon(coords_np, tolerance=tolerancia)
  poligonos.append(simplified_contour)

#####################
# Limpiamos el poligono (descartamos los poligonos que tengan menos de 3 vertices)
pol_limpio = [pol for pol in poligonos if len(pol) > 2]

try:
  for pols in poligonos_filtrados: # aca cambie multi_poligono.geoms a poligonos_filtrados ->LAST
    tiempo_ejecucion = time.time() - tiempo_inicio
    if tiempo_ejecucion > 60: break
    try:
      coords = pols.exterior.coords  # Obtener las coordenadas del contorno
      coords_np = np.array(coords)  # A numpy
      simplified_contour = approximate_polygon(coords_np, tolerance=tolerancia)
      poligonos.append(simplified_contour)
      plt.plot(simplified_contour[:, 1], simplified_contour[:, 0], linewidth=1, color='blue',alpha=1) #----------------------------------->>>>cambie de fill a plot
    except:
      continue

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
  if "/" in archivo:
    archivo = archivo.split("/")[-1].replace(".tif","")
  gdf.to_file(directorio+"/polygons/"+archivo+"_polygon.geojson", driver="GeoJSON")
  print("Se ha guardado el poligono")

except:
  print("No se encontro poligono")
