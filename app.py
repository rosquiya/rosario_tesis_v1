from flask import Flask, render_template, Response, jsonify
import cv2
import os
import base64
import json
import numpy as np
from ultralytics import YOLO
from prepro import process_image  # Asegúrate de tener este archivo con la función process_image

app = Flask(__name__)

# Ruta a la carpeta de imágenes
IMG_FOLDER = os.path.join('static', 'img')
if not os.path.exists(IMG_FOLDER):
    os.makedirs(IMG_FOLDER)

# Función para obtener la cámara disponible
def get_camera():
    chosen_camera = None

    # Intentar abrir la cámara con índices 1 y 0 (en ese orden)
    for index in [1, 0]:
        camera = cv2.VideoCapture(index)
        if camera.isOpened():
            chosen_camera = camera
            break  # Salir del bucle si se encuentra una cámara disponible
        else:
            camera.release()  # Liberar la cámara si no está disponible

    if chosen_camera is None:
        print("No se encontró ninguna cámara disponible.")
    else:
        # Cargar configuraciones desde el archivo JSON
        with open('camera_config.json', 'r') as f:
            camera_config = json.load(f)

        # Configurar la cámara
        chosen_camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Desactivar el autoenfoque
        chosen_camera.set(cv2.CAP_PROP_BRIGHTNESS, camera_config.get('brightness', 128))
        chosen_camera.set(cv2.CAP_PROP_SATURATION, camera_config.get('saturation', 128))
        chosen_camera.set(cv2.CAP_PROP_CONTRAST, camera_config.get('contrast', 128))
        chosen_camera.set(cv2.CAP_PROP_FOCUS, camera_config.get('focus', 50))

    return chosen_camera

# Cargar modelo YOLO
model = YOLO('best.pt')  # Asegúrate de que 'best.pt' esté en la ubicación correcta

# Renderizar la página principal
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/user')
def user():
    return render_template('pantalla_usuario.html')

# Generar flujo de video
def gen_frames():
    camera = get_camera()
    if camera is None:
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + b'\r\n')
    else:
        while True:
            success, frame = camera.read()
            if not success:
                break
            else:
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Capturar imagen y realizar inferencia
@app.route('/capture', methods=['POST'])
def capture():
    camera = get_camera()
    if camera is None:
        return jsonify({'success': False, 'message': 'No se encontró ninguna cámara disponible.'})

    success, frame = camera.read()
    if not success:
        return jsonify({'success': False, 'message': 'Failed to capture image.'})
    
    img_path = os.path.join(IMG_FOLDER, 'captured_image.jpg')
    cv2.imwrite(img_path, frame)
    
    # Procesar la imagen
    processed_image = process_image(img_path)
    
    if processed_image is not None:
        results = model(processed_image)
        names_dict = results[0].names
        probs = results[0].probs.data.tolist()
        max_prob_index = np.argmax(probs)
        
        with open(img_path, 'rb') as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode('utf-8')

        return jsonify({
            'success': True,
            'image': img_base64,
            'predictions': {
                'names_dict': names_dict,
                'probs': probs,
                'max_prob': names_dict[max_prob_index]
            }
        })
    else:
        return jsonify({'success': False, 'message': f'Failed to process image: {img_path}'})

if __name__ == "__main__":
    app.run(port=8000, host="0.0.0.0", debug=True)
