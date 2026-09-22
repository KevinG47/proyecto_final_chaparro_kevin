# Machine Learning con PySpark y Docker

**Curso:** Machine Learning con PySpark y Docker | 2026-I  
**Estudiante:** Kevin Leonardo Chaparro Reyes  
**Universidad:** Universidad Santo Tomás · Programa de Estadística  
**Profesora:** Luz Adriana Gutiérrez Rodríguez

---

## ¿De qué trata este proyecto?

Este proyecto constituye el componente práctico del parcial final del curso. Se desarrolla un análisis integral sobre datos reales colombianos, combinando tres ejes metodológicos:

- **Bloque 1 — Análisis exploratorio** de la contratación pública colombiana (SECOP II), identificando patrones en modalidades de contratación, distribución del gasto por sector y detección de atípicos.
- **Bloque 2 — Machine Learning supervisado y no supervisado** para predecir si un contrato se adjudica sin competencia abierta, complementado con PCA y K-Means para descubrir perfiles de contratación.
- **Bloque 3 — Procesamiento de lenguaje natural** sobre reseñas de aplicaciones financieras colombianas, comparando un pipeline clásico (TF-IDF + Regresión Logística) con un modelo pre-entrenado de Hugging Face (robertuito).

---

## Datasets utilizados

| Bloque | Dataset | Descripción | Fuente |
|--------|---------|-------------|--------|
| 1 y 2 | SECOP II — Contratos Electrónicos | 52,832 contratos de contratación pública firmados en 2026, con información de entidad, sector, valor, modalidad y proveedor | [datos.gov.co](https://www.datos.gov.co/Estad-sticas-Nacionales/SECOP-II-Contratos-Electr-nicos/jbjy-vk9h) (API Socrata) |
| 3 | Reseñas de apps financieras colombianas | 2,316 reseñas en español de Google Play Store (Nequi, Bancolombia, Rappi) con calificación en estrellas | Google Play Store vía `google-play-scraper` |

Los datasets se descargan automáticamente desde los notebooks. No es necesario descargarlos manualmente.

---

## Estructura del repositorio

```
proyecto_final_chaparro_kevin/
├── README.md                          ← Este archivo
├── Dockerfile                         ← Imagen Docker con dependencias NLP
├── docker-compose.yml                 ← Configuración del contenedor
├── requirements.txt                   ← Librerías adicionales (transformers, torch, etc.)
├── .gitignore
├── datos/
│   ├── README.md                      ← Descripción de los archivos de datos
│   ├── secop_contratos.csv            ← Se genera en Bloque 1
│   ├── secop_limpio.parquet/          ← Se genera en Bloque 1
│   ├── resenas_apps.csv               ← Se genera en Bloque 3
│   └── corpus_sentimiento.csv         ← Se genera en Bloque 3
├── bloque1_eda/
│   ├── bloque1_eda_chaparro.ipynb     ← Notebook del análisis exploratorio
│   └── *.png                          ← Figuras generadas
├── bloque2_ml/
│   ├── bloque2_ml_chaparro.ipynb      ← Notebook de Machine Learning
│   └── *.png                          ← Figuras generadas
├── bloque3_nlp/
│   ├── bloque3_nlp_chaparro.ipynb     ← Notebook de NLP
│   └── *.png                          ← Figuras generadas
└── reporte_ejecutivo.pdf              ← Resumen ejecutivo del proyecto
```

---

## Cómo ejecutar el proyecto paso a paso

### Requisitos previos

Antes de empezar hay que tener instalado:
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (con al menos 10 GB de espacio libre)
- [Git](https://git-scm.com/downloads)
- [Visual Studio Code](https://code.visualstudio.com/) (recomendado)
- Conexión a internet (para la descarga inicial de datos y del modelo de Hugging Face)

---

### Paso 1 — Clonar el repositorio

1. Abrir **Visual Studio Code**
2. En el menú superior: **Terminal** → **New Terminal**
3. Escribir el siguiente comando y presionar Enter:
```bash
git clone https://github.com/KevinG47/proyecto_final_chaparro_kevin.git
```
4. Entrar a la carpeta del proyecto:
```bash
cd proyecto_final_chaparro_kevin
```

---

### Paso 2 — Construir y levantar el contenedor

1. Asegurarse de que **Docker Desktop** esté abierto y corriendo (el ícono de la ballena debe estar quieto en la barra de tareas)
2. En la terminal de VS Code, ejecutar:
```bash
docker-compose up --build
```
3. La **primera vez** tarda entre 10 y 15 minutos porque descarga la imagen base (~4 GB) e instala las dependencias de NLP (transformers, torch). Es completamente normal.
4. El contenedor está listo cuando aparece un mensaje similar a:
```
proyecto_final_ml | [I ... ServerApp] Jupyter Server is running at:
proyecto_final_ml | [I ... ServerApp] http://localhost:8888/lab
```

---

### Paso 3 — Abrir JupyterLab

1. Abrir el navegador (Chrome, Firefox, Edge)
2. Ir a: **http://localhost:8888**
3. Se abrirá JupyterLab con la estructura del proyecto visible en el panel izquierdo

---

### Paso 4 — Ejecutar los notebooks

Ejecutar los notebooks **en este orden**, ya que el Bloque 2 depende de los datos procesados en el Bloque 1:

1. `bloque1_eda/bloque1_eda_chaparro.ipynb`
2. `bloque2_ml/bloque2_ml_chaparro.ipynb`
3. `bloque3_nlp/bloque3_nlp_chaparro.ipynb`

Para cada notebook: **Kernel** → **Restart Kernel and Run All Cells** → Confirmar con **Restart**

**Tiempos aproximados:**
- Bloque 1: ~3 minutos
- Bloque 2: ~10 minutos (incluye CrossValidator con 45 modelos)
- Bloque 3: ~8 minutos (incluye inferencia con robertuito en CPU)

---

### Para detener el proyecto

1. Volver a la terminal de VS Code
2. Presionar `Ctrl + C` para detener el contenedor
3. Ejecutar:
```bash
docker-compose down
```

---

## Versiones del entorno

| Componente | Versión |
|-----------|---------|
| Python | 3.11 |
| PySpark | 3.5.0 |
| Pandas | 2.0.3 |
| Scikit-learn | 1.3.1 |
| Transformers | 4.x |
| Torch | 2.x |
| Matplotlib | 3.8.0 |

---

## Principales hallazgos

### Bloque 1 — Análisis Exploratorio
- El 58.6% de los contratos públicos se adjudican sin competencia abierta (contratación directa o régimen especial).
- La distribución de valores presenta asimetría extrema (72.94): la mediana es 24.5 millones COP mientras la media es 311.5 millones.
- La diferencia de valores entre contratos directos y competitivos es estadísticamente significativa (Mann-Whitney, p < 0.001).

### Bloque 2 — Machine Learning
- El Random Forest optimizado predice la modalidad de contratación con 95.0% de accuracy y AUC de 0.989.
- Las variables más predictivas son el sector (32.8%), si el proveedor es PYME (27.2%) y el tipo de contrato (18.5%).
- K-Means reveló 5 perfiles de contratación diferenciados, desde contratos pequeños de salud bajo régimen especial hasta grandes licitaciones de servicios públicos.

### Bloque 3 — NLP
- TF-IDF + Regresión Logística alcanza 90.4% de accuracy en clasificación de sentimiento.
- robertuito alcanza 96.1% sobre el mismo test set, pero clasifica el 31% de las reseñas como neutro.
- El análisis de casos difíciles reveló etiquetas ruidosas: usuarios que dan 5 estrellas con texto negativo.

---

## Explicación del docker-compose.yml

**`build: .`**  
En lugar de usar una imagen directamente, construye una imagen personalizada a partir del `Dockerfile` incluido en el repositorio. Esto permite pre-instalar las dependencias de NLP (transformers, torch, sentence-transformers).

**`ports: "8888:8888"`**  
Conecta el puerto del contenedor con el del computador local, permitiendo acceder a JupyterLab desde el navegador.

**`volumes`**  
Montan las carpetas locales de cada bloque y la carpeta de datos dentro del contenedor. Esto garantiza que los archivos generados persistan al apagar el contenedor.

**`shm_size: '2g'`**  
Aumenta la memoria compartida disponible para PySpark, evitando errores con datasets grandes.

**`command: start-notebook.sh --NotebookApp.token=''`**  
Inicia JupyterLab sin requerir token de autenticación, simplificando el acceso.

---

---

### Reproducibilidad

Todas las operaciones aleatorias (splits, K-Means, Random Forest) utilizan 
`seed=42`. Los datasets se incluyen en el repositorio para garantizar que los 
resultados sean idénticos al ejecutar los notebooks. El orden de ejecución 
(Bloque 1 → 2 → 3) debe respetarse porque el Bloque 2 depende del archivo 
Parquet generado por el Bloque 1.

---

## Conclusión integrada

El análisis de la contratación pública colombiana (SECOP II) y del sentimiento en reseñas de aplicaciones financieras revela patrones estructurales claros en ambos dominios. En contratación pública, la modalidad de adjudicación es predecible con 95% de accuracy a partir de variables institucionales como el sector y el tipo de contrato, reflejando un sistema regulatorio que predetermina en gran medida las condiciones de contratación. En el ecosistema financiero digital, tanto métodos clásicos de NLP (TF-IDF, 90.4%) como modelos pre-entrenados (robertuito, 96.1%) capturan efectivamente el sentimiento de los usuarios, con trade-offs distintos entre interpretabilidad, cobertura y precisión. Los tres bloques demuestran que PySpark permite gestionar eficientemente el flujo completo desde datos crudos hasta modelos interpretables sobre datos reales colombianos.
