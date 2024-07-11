# Usa una imagen base oficial de Python
FROM python:3.9-slim

# Establece el directorio de trabajo en el contenedor
WORKDIR /app

# Copia los archivos de requerimientos al directorio de trabajo
COPY requirements.txt .

# Instala las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto de la aplicación al contenedor
COPY . .

# Exponer el puerto en el que la aplicación correrá
EXPOSE 5000

# Comando para correr la aplicación
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
