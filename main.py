import tkinter as tk
from tkinter import filedialog  as fd #Ventanas de dialogo
from PIL import Image
from PIL import ImageTk
import cv2
import imutils
import numpy as np

url_imagen = ""
class Aplication:
    
    def __init__(self):
        self.raiz = tk.Tk() 
        self.raiz.title("HORMIGA NEGRA") #Cambiar el nombre de la ventana 
        self.raiz.geometry("600x500") #Configurar tamaño
        self.raiz.resizable(width=0, height=0)
        self.raiz.iconbitmap("./Imagenes/ant.ico") #Cambiar el icono
        self.imagen= tk.PhotoImage(file="./Imagenes/fondo2.png")
        tk.Label(self.raiz, image=self.imagen, bd=0).pack()

        #Labels
        self.image_label = tk.Label(self.raiz, bg="white")
        self.image_label.place(x=220, y=150)


        #Botones
        self.boton = tk.Button(text="Elegir imagen", bg="#73B731", fg="#ffffff",font=("Verdana", 9, "bold"), borderwidth = 0, command=self.seleccionar)
        self.boton.place(x=245, y=120, width=120)
        self.boton_canales = tk.Button(self.raiz, text="Canales RGB", width=2, height=2, bg="#73B731", fg="#ffffff",font=("Verdana", 9, "bold"), borderwidth = 0, command=self.split_channels, state="disabled")
        self.boton_canales.place(x= 40, y= 330, width=95)
        self.boton_binari = tk.Button(self.raiz, text="Binarizacion", width=2, height=2, bg="#73B731", fg="#ffffff",font=("Verdana", 9, "bold"), borderwidth = 0, command=self.binarizacion, state="disabled")
        self.boton_binari.place(x= 150, y= 330, width=100)
        self.boton_exp = tk.Button(self.raiz, text="Expansión histograma", width=2, height=2, bg="#73B731", fg="#ffffff",font=("Verdana", 9, "bold"), borderwidth = 0, command=self.expansion_contraccion, state="disabled")
        self.boton_exp.place(x= 265, y= 330, width=150)
        self.boton_gaussiano = tk.Button(self.raiz, text="Filtro Gaussiano", width=2, height=2, bg="#73B731", fg="white",font=("Verdana", 9, "bold"), borderwidth = 0, command=self.gaussiano, state="disabled")
        self.boton_gaussiano.place(x= 40, y= 375, width=150)
        self.boton_laplaciano = tk.Button(self.raiz, text="Filtro Laplaciano Negativo", width=2, height=2, bg="#73B731", fg="white",font=("Verdana", 9, "bold"), borderwidth = 0, command=self.laplacianoNegative, state="disabled")
        self.boton_laplaciano.place(x= 210, y= 375, width=205)
        self.boton_or = tk.Button(self.raiz, text="OR", width=2, height=2, bg="#73B731", fg="white",font=("Verdana", 9, "bold"), borderwidth = 0, command=self.operacionOR, state="disabled")
        self.boton_or.place(x= 40, y= 420, width=60)

        self.boton_fminimo = tk.Button(self.raiz, text="Filtro Minimo", width=2, height=2, bg="#73B731", fg="white",font=("Verdana", 9, "bold"), borderwidth = 0, command=self.filtromediano, state="disable")
        self.boton_fminimo.place(x= 120, y= 420, width=120)



        self.raiz.mainloop() 

    def seleccionar(self):
            global url_imagen
            print(url_imagen)
            nomArchivo = fd.askopenfilename(initialdir= "C:/Users/USER/OneDrive/Escritorio/AImages", title= "Seleccionar Archivo", filetypes= (("Image files", "*.png; *.jpg; *.gif"),("todos los archivos", "*.*")))
            url_imagen = nomArchivo
            print(url_imagen)
            if nomArchivo!='':
                self.image = Image.open(nomArchivo)
                self.image = self.image.resize((170, 150))#Normalizamos la imagen
                self.photo = ImageTk.PhotoImage(self.image)
                self.image_label.configure(image=self.photo)
            
                # Habilitar botones
                self.boton_exp.configure(state="normal")
                self.boton_gaussiano.configure(state="normal")
                self.boton_laplaciano.configure(state="normal")
                self.boton_canales.configure(state="normal")
                self.boton_binari.configure(state="normal")
                self.boton_or.configure(state="normal")
                self.boton_fminimo.configure(state="normal")
    
    def expansion_contraccion(self):
        global url_imagen
        img = url_imagen
        print(url_imagen)
        image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        
    
        min_value = 0  # Reemplaza con el valor mínimo deseado (rango 1-255)
        max_value = 256  # Reemplaza con el valor máximo deseado (rango 1-255)
        # Obtiene el valor mínimo y máximo de intensidad en la imagen
        current_min = np.min(image)
        current_max = np.max(image)

        # Aplica la contracción y expansión del histograma
        new_image = np.interp(image, (current_min, current_max), (min_value, max_value))

        # Convierte la imagen resultante a tipo entero sin signo de 8 bits
        new_image = new_image.astype(np.uint8)

        # Guarda la imagen resultante
        cv2.imwrite('ImagenExpansion.jpg', new_image)
        #Mostramos imagen
        cv2.imshow('Expansion del histograma',new_image)
        cv2.waitKey()

    def gaussiano(self):
        global url_imagen
        url = url_imagen
        image = cv2.imread(url)

        # Verifica si la imagen se ha cargado correctamente
        if image is None:
            print('No se pudo cargar la imagen.')
            return
        
        kernel_size = 13  # Tamaño del kernel del filtro gaussiano (impar)
        sigma = 0  # Desviación estándar del filtro gaussiano

        # Aplica el filtro gaussiano
        filtered_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        
        # Guarda la imagen resultante
        cv2.imwrite('ImagenFiltroGaussiano.jpg', filtered_image)

        #Mostramos la imagen
        cv2.imshow('Filtro Gaussiano',filtered_image)
        cv2.waitKey()

    def laplacianoNegative(self):
        global url_imagen
         # Lee la imagen en escala de grises
        image = cv2.imread(url_imagen, cv2.IMREAD_GRAYSCALE)

        # Aplica el filtro Laplaciano
        laplacian = cv2.Laplacian(image, cv2.CV_64F)

        # Obtiene el valor máximo absoluto de la imagen filtrada
        max_value = np.max(np.abs(laplacian))

        # Normaliza y crea el filtro Laplaciano negativo
        negative_laplacian = (255 / max_value) * np.abs(laplacian)
        negative_laplacian = negative_laplacian.astype(np.uint8)

        # Guarda la imagen resultante
        cv2.imwrite('ImagenFiltroLaplaciano.jpg', negative_laplacian)

        #Mostramos la imagen
        cv2.imshow('Filtro Laplaciano Negativo',negative_laplacian)
        cv2.waitKey()


    def split_channels(self):
        global url_imagen
        # Lee la imagen a color
        image = cv2.imread(url_imagen)

        # Divide la imagen en sus canales RGB
        blue_channel, green_channel, red_channel = cv2.split(image)

        # Guarda cada canal en archivos separados
        #cv2.imwrite('CanalRojo.jpg', red_channel)
        cv2.imwrite('CanalVerdeGrises.jpg', green_channel)
        #cv2.imwrite('CanalAzul.jpg', blue_channel)

        imagen_red=np.copy(image) # creo una copia de la imagen para preservar la original
        imagen_red[:,:,0]=0
        imagen_red[:,:,1]=0

        imagen_green=np.copy(image) # creo una copia de la imagen para preservar la original
        imagen_green[:,:,0]=0
        imagen_green[:,:,2]=0

        imagen_blue=np.copy(image) # creo una copia de la imagen para preservar la original
        imagen_blue[:,:,1]=0
        imagen_blue[:,:,2]=0

        # Guarda la imagen resultante
        cv2.imwrite('ImagenVerde.jpg', imagen_green)

        #Mostramos imagenes
        cv2.imshow('Red',imagen_red)
        cv2.imshow('Green',imagen_green)
        cv2.imshow('Blue',imagen_blue)
        cv2.waitKey()

        

    def binarizacion(self):
        global url_imagen
        umbral = 54
        # Lee la imagen a color
        image = cv2.imread(url_imagen, cv2.IMREAD_GRAYSCALE)

        _ , imgBin = cv2.threshold(image, umbral ,255, cv2.THRESH_BINARY)

         # Guarda la imagen resultante
        cv2.imwrite('ImagenBinarizada.jpg', imgBin)

        cv2.imshow('binaria', imgBin)
        cv2.waitKey(0)                   
        cv2.destroyWindow('binaria') 

    def operacionOR(self):
        global url_imagen
        # Cargar las dos imágenes
        imagen1 = cv2.imread(url_imagen)
        imagen2 = cv2.imread("ImagenFiltroLaplaciano.jpg")

        # Verificar que las imágenes tienen el mismo tamaño
        if imagen1.shape == imagen2.shape:
            # Aplicar la operación OR entre las dos imágenes
            resultado = cv2.bitwise_or(imagen1, imagen2)
            # Guarda la imagen resultante
            cv2.imwrite('ImagenSegmentada.jpg', resultado)

            # Mostrar la imagen resultante
            cv2.imshow('Resultado', resultado)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print('Las imágenes tienen tamaños diferentes.')
        
    
    def segmentacion(self):
       global url_imagen
       image = cv2.imread(url_imagen)
    
       # Divide la imagen en sus canales RGB
       blue_channel, green_channel, red_channel = cv2.split(image)
       # Guarda cada canal en archivos separados
       cv2.imwrite('CanalVerdeGrises.jpg', green_channel)

       #Filtro Minimo
       ksize = (4, 4)
       # Aplicar el filtro mínimo
       imagen_filtrada = cv2.erode("CanalVerdeGrises.jpg", np.ones(ksize, np.uint8))
       # Guarda la imagen resultante
       cv2.imwrite('ImagenFiltroMinimo.jpg', imagen_filtrada)

       #Binarización
       umbral = 50
       imgBin = cv2.threshold('ImagenFiltroMinimo.jpg', umbral ,255, cv2.THRESH_BINARY)
       # Guarda la imagen resultante
       cv2.imwrite('ImagenBinarizada.jpg', imgBin)

       #Filtro Laplaciano Negativo
       # Aplica el filtro Laplaciano
       laplacian = cv2.Laplacian('ImagenBinarizada.jpg', cv2.CV_64F)
       # Obtiene el valor máximo absoluto de la imagen filtrada
       max_value = np.max(np.abs(laplacian))
       # Normaliza y crea el filtro Laplaciano negativo
       negative_laplacian = (255 / max_value) * np.abs(laplacian)
       negative_laplacian = negative_laplacian.astype(np.uint8)
       # Guarda la imagen resultante
       cv2.imwrite('ImagenFiltroLaplaciano.jpg', negative_laplacian)

       #Operación OR
       # Cargar las dos imágenes
       imagen1 = cv2.imread(url_imagen)
       imagen2 = cv2.imread("ImagenFiltroLaplaciano.jpg")
       # Aplicar la operación OR entre las dos imágenes
       resultado = cv2.bitwise_or(imagen1, imagen2)
       # Mostrar la imagen resultante
       cv2.imshow('Resultado', resultado)
       cv2.waitKey(0)
       cv2.destroyAllWindows()


    def filtromediano(self):
        global url_imagen
        imagen = cv2.imread(url_imagen, cv2.IMREAD_GRAYSCALE)
        ksize = (4, 4)

        # Aplicar el filtro mínimo
        imagen_filtrada = cv2.erode(imagen, np.ones(ksize, np.uint8))

        # Guarda la imagen resultante
        cv2.imwrite('ImagenFiltroMinimo.jpg', imagen_filtrada)

        cv2.imshow('Segmentación', imagen_filtrada)
        cv2.waitKey(0)   

aplication = Aplication()
