import cv2
import numpy as np

def process_image(image_path, output_size=(640, 640)):
    # Leer la imagen desde la ruta proporcionada
    img = cv2.imread(image_path)

    if img is None:
        print(f"No se pudo leer la imagen en la ruta: {image_path}")
        return None

    # Escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY_INV)

    # Erosión y dilatación
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(mask, kernel, iterations=1)
    dilation = cv2.dilate(erosion, kernel, iterations=1)

    # Máscara final
    result = cv2.bitwise_and(img, img, mask=dilation)

    # Recorte según el tamaño de salida especificado
    contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Escalar la región recortada al tamaño de salida deseado
        cropped_result = img[y:y+h, x:x+w]
        resized_result = cv2.resize(cropped_result, output_size)

        return resized_result
    else:
        print(f"No se encontraron contornos en la imagen: {image_path}")
        return None
