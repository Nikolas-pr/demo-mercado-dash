# Imagen base actualizada y segura
FROM python:3.13-slim

# Variables de entorno recomendadas
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV LANG=C.UTF-8

# Instala dependencias del sistema necesarias para compilar librerías Python
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libpq-dev \
    libssl-dev \
    libffi-dev \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia todos los archivos del proyecto al contenedor
COPY . /app

# Instala dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Expone el puerto en el que corre Dash por defecto
EXPOSE 8050

# Comando para arrancar la app con Gunicorn en producción
CMD ["gunicorn", "--bind", "0.0.0.0:8050", "app:server"]

