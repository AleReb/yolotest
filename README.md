# YOLOv11 Object Tracking - Web Version

Este proyecto es una implementación web de detección y seguimiento de objetos en tiempo real utilizando **YOLOv11** y **ONNX Runtime**. Funciona completamente en el navegador, utilizando la cámara web para detectar y rastrear objetos con visualización de estelas y dirección de movimiento.

![Screenshot](img/screenshot.jpg)

## Características

*   **Detección en Tiempo Real**: Utiliza el modelo YOLOv11n (exportado a ONNX) para detectar 80 clases de objetos (personas, vehículos, animales, etc.).
*   **Seguimiento de Objetos (Tracking)**: Asigna IDs únicos a los objetos y mantiene su rastro a través de los frames.
*   **Visualización de Estelas**: Dibuja la trayectoria de movimiento de cada objeto.
*   **Modo Rendimiento (Lite)**: Incluye una opción para cambiar dinámicamente a un modelo más ligero (320x320) para dispositivos menos potentes.
*   **Interfaz Neon**: Diseño moderno con colores dinámicos para cada tipo de objeto.
*   **Privacidad**: Todo el procesamiento se realiza localmente en el navegador. Ninguna imagen se envía a servidores externos.

## Instalación y Uso

1.  **Clonar el repositorio**:
    ```bash
    git clone https://github.com/tu-usuario/yolo-web-tracking.git
    cd yolo-web-tracking
    ```

2.  **Ejecutar servidor local**:
    Para probarlo localmente, necesitas un servidor web simple (debido a las políticas de seguridad del navegador para la cámara y WASM).
    ```bash
    python -m http.server 8000
    ```

3.  **Abrir en el navegador**:
    Visita `http://localhost:8000` en tu navegador.

## Estructura del Proyecto

*   `index.html`: Interfaz principal.
*   `yolo.js`: Lógica de detección, seguimiento y renderizado.
*   `style.css`: Estilos de la aplicación.
*   `model/`: Carpeta para los modelos ONNX (se deben generar o descargar).
    *   `yolo11n.onnx`: Modelo estándar (640x640).
    *   `yolo11n-320.onnx`: Modelo ligero (320x320).

## Créditos

Creado por **Alejandro Rebolledo** (arebolledo@udd.cl).

Basado en la arquitectura YOLO de Ultralytics y ONNX Runtime Web.

## Licencia

Este proyecto está bajo la licencia **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**.
Ver el archivo `LICENSE` para más detalles.

## Descargo de Responsabilidad

ESTE SOFTWARE SE PROPORCIONA "TAL CUAL", SIN GARANTÍA DE NINGÚN TIPO, EXPRESA O IMPLÍCITA. EL USO DE ESTE CÓDIGO ES BAJO SU PROPIO RIESGO. EL AUTOR NO SE HACE RESPONSABLE DE NINGÚN DAÑO O PÉRDIDA QUE PUEDA SURGIR DEL USO DE ESTE SOFTWARE.
