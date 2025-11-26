# YOLO Object Tracking & Depth Estimation

Este proyecto implementa un sistema de seguimiento de objetos en tiempo real utilizando **YOLOv11**. Incluye dos versiones:

1. **Versión Python**: Con visualización de estelas de movimiento y estimación de profundidad opcional (MiDaS)
2. **Versión Web**: Ejecutable en el navegador, ideal para GitHub Pages

## Características

### Versión Python
- **Detección y Seguimiento**: Utiliza YOLOv11 para detectar y seguir objetos
- **Estelas de Movimiento**: Dibuja una línea que muestra la trayectoria reciente de los objetos
- **Estimación de Profundidad (Opcional)**: Estima la distancia relativa de los objetos utilizando MiDaS
- **Dirección de Movimiento**: Indica si el objeto se mueve hacia arriba, abajo, izquierda o derecha

### Versión Web
- **Ejecutable en Navegador**: No requiere instalación, funciona directamente en el navegador
- **Acceso a Webcam**: Utiliza la cámara web del dispositivo
- **Tracking en Tiempo Real**: Seguimiento de objetos con IDs persistentes
- **Visualización de Trayectorias**: Muestra las estelas de movimiento
- **Compatible con GitHub Pages**: Puede desplegarse como sitio estático

## Requisitos

### Versión Python
- Python 3.8+
- Cámara web

### Versión Web
- Navegador moderno (Chrome, Firefox, Edge, Safari)
- Conexión HTTPS (requerida para acceso a webcam)

## Instalación

### Versión Python

1.  Clonar el repositorio:
    ```bash
    git clone https://github.com/tu_usuario/tu_repositorio.git
    cd tu_repositorio
    ```

2.  Crear y activar un entorno virtual (recomendado):
    ```bash
    python -m venv .venv
    # Windows
    .\.venv\Scripts\Activate
    # Linux/Mac
    source .venv/bin/activate
    ```

3.  Instalar las dependencias:
    ```bash
    pip install -r requirements.txt
    ```

### Versión Web

1.  Abrir `index.html` en un navegador web
2.  Para GitHub Pages: Subir los archivos `index.html`, `style.css`, `app.js` y `yolo11n.onnx` al repositorio

## Uso

### Versión Python

Ejecutar el script principal:

```bash
python test.py
```

#### Configuración

Para activar o desactivar la estimación de profundidad, edita la variable `ENABLE_DEPTH` en `test.py`:

```python
# test.py
ENABLE_DEPTH = True  # Para activar
ENABLE_DEPTH = False # Para desactivar (más rápido)
```

### Versión Web

1.  Abrir `index.html` en un navegador
2.  Hacer clic en "Start Camera"
3.  Permitir el acceso a la cámara web
4.  El sistema comenzará a detectar y seguir objetos automáticamente

> **Nota**: La versión web requiere HTTPS para acceder a la cámara. Si estás probando localmente, usa un servidor local como `python -m http.server` o despliega en GitHub Pages.

## Estructura del Proyecto

```
yolotest/
├── test.py              # Versión Python
├── index.html           # Versión Web - HTML
├── style.css            # Versión Web - Estilos
├── app.js               # Versión Web - Lógica
├── yolo11n.onnx         # Modelo YOLO en formato ONNX
├── requirements.txt     # Dependencias Python
├── LICENSE              # Licencia CC BY-NC 4.0
└── README.md            # Este archivo
```

## Autor

**Alejandro Rebolledo**  
Correo: [arebolledo@udd.cl](mailto:arebolledo@udd.cl)

## Licencia

Este proyecto está bajo la licencia **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**.
Ver el archivo `LICENSE` para más detalles.

## Descargo de Responsabilidad

ESTE SOFTWARE SE PROPORCIONA "TAL CUAL", SIN GARANTÍA DE NINGÚN TIPO, EXPRESA O IMPLÍCITA. EL USO DE ESTE CÓDIGO ES BAJO SU PROPIO RIESGO. EL AUTOR NO SE HACE RESPONSABLE DE NINGÚN DAÑO O PROBLEMA DERIVADO DE SU USO.
