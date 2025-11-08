# ===========================================
# Etapa 1: Imagen base ligera con Python
# ===========================================
FROM python:3.11-slim AS base

# Evita que Python genere archivos .pyc y que use buffering (mejor para logs)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app


# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia solo los archivos necesarios primero (para aprovechar la caché de Docker)
COPY requirements.txt ./

# Instala dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Ahora copia el resto de los archivos del proyecto
COPY . .

# Expón el puerto en el que correrá Flask
EXPOSE 8000

# Establece una variable de entorno para Flask (opcional pero recomendado)
ENV FLASK_APP=app/app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_ENV=production

# ===========================================
# Etapa 2: Ejecución de la app
# ===========================================
# Usa un comando robusto para producción (en lugar de flask run)
CMD ["python", "app/app.py"]
