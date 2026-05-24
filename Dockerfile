# ============================================================
# Proyecto Final — ML con PySpark y Docker
# Universidad Santo Tomás · Estadística · 2026-I
# ============================================================
# Imagen base: Jupyter con PySpark preinstalado
FROM jupyter/pyspark-notebook:latest

# Copiar dependencias adicionales
COPY requirements.txt /tmp/requirements.txt

# Instalar paquetes adicionales necesarios para el Bloque 3 (NLP)
# --no-cache-dir reduce el tamaño de la imagen
RUN pip install --no-cache-dir -r /tmp/requirements.txt
