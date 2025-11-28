# Human Pose Tracking & Object Detection - Web

Sistema web de **detección de postura humana** y **detección de objetos** en tiempo real utilizando **YOLOv11** y **ONNX Runtime**. Funciona completamente en el navegador usando la cámara web.

## � Demo en Vivo

**Prueba la aplicación aquí:** [https://alereb.github.io/yolotest/](https://alereb.github.io/yolotest/)

> **Nota:** Necesitarás permitir el acceso a la cámara web en tu navegador.

## �🎯 Características

### Dos Modos de Detección

1. **Modo Pose (Postura)**
   - Detecta personas y muestra 17 puntos clave del cuerpo (hombros, codos, rodillas, etc.)
   - Dibuja el esqueleto completo con conexiones entre articulaciones
   - Opción "Solo Esqueleto" para visualización minimalista

2. **Modo Object Detection (Objetos)**
   - Detecta 80 clases de objetos COCO (personas, vehículos, animales, objetos cotidianos)
   - Muestra cajas delimitadoras con etiquetas y confianza

### Funcionalidades Comunes

- **Seguimiento Multi-Objeto**: Asigna IDs únicos y mantiene el rastro de cada detección
- **Visualización de Trayectorias**: Muestra estelas de movimiento
- **Detección de Dirección**: Indica si los objetos se mueven (arriba, abajo, izquierda, derecha)
- **Modo Rendimiento**: Alterna entre modelos de 320x320 (rápido) y 640x640 (preciso)
- **Privacidad Total**: Todo el procesamiento se realiza localmente en el navegador

## 🚀 Instalación y Uso

### Opción 1: Uso Web (Sin Python)

Si solo quieres usar la aplicación web, **no necesitas Python**. Solo necesitas un servidor web simple:

1. **Clonar el repositorio**:
   ```bash
   git clone https://github.com/tu-usuario/yolo-web-tracking.git
   cd yolo-web-tracking
   ```

2. **Ejecutar servidor local**:
   
   **Con Python (si lo tienes):**
   ```bash
   python -m http.server 8000
   ```
   
   **Con Node.js:**
   ```bash
   npx http-server -p 8000
   ```
   
   **Windows (con Python):**
   ```powershell
   .\run_server.bat
   ```

3. **Abrir en el navegador**:
   Visita `http://localhost:8000`

### Opción 2: Setup Completo con Python (Para Testing y Export)

Si quieres usar el script de testing (`test_model.py`) o exportar tus propios modelos:

1. **Crear entorno virtual**:
   ```bash
   # Windows
   python -m venv .venv
   .\.venv\Scripts\Activate
   
   # Linux/Mac
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ejecutar script de prueba**:
   ```bash
   python test_model.py --model pose    # Para detección de postura
   python test_model.py --model object  # Para detección de objetos
   ```

4. **Exportar modelos personalizados** (opcional):
   ```bash
   python export_model.py
   ```

## 📁 Estructura del Proyecto

```
yolotest/
├── index.html              # Interfaz web principal
├── yolo.js                 # Lógica de detección y renderizado
├── style.css               # Estilos de la aplicación
├── test_model.py           # Script de prueba en Python
├── export_model.py         # Script para exportar modelos a ONNX
├── run_server.bat          # Servidor web rápido (Windows)
├── yolo11n-pose.onnx       # Modelo de postura (640x640)
├── yolo11n-pose-320.onnx   # Modelo de postura ligero (320x320)
├── yolo11n.onnx            # Modelo de objetos (640x640)
├── yolo11n-320.onnx        # Modelo de objetos ligero (320x320)
└── LICENSE                 # Licencia CC BY-NC 4.0
```

## 🎮 Controles de la Interfaz

- **Modelo**: Selector para cambiar entre Pose y Object Detection
- **Solo Esqueleto**: (Solo en modo Pose) Muestra únicamente los puntos y líneas del esqueleto
- **Mostrar Estelas**: Activa/desactiva las trayectorias de movimiento
- **Modo Rendimiento**: Alterna entre modelos de 320px (rápido) y 640px (preciso)

## 🔧 Exportar Modelos Personalizados

Si deseas usar otros modelos YOLO:

```bash
# Activar entorno virtual
.\.venv\Scripts\Activate

# Instalar ultralytics
pip install ultralytics

# Exportar modelo
python export_model.py
```

Edita `export_model.py` para cambiar el modelo base o el tamaño de entrada.

## 🌐 Navegadores Compatibles

- Chrome/Edge (Recomendado)
- Firefox
- Safari (macOS/iOS)

**Nota**: Se requiere HTTPS o localhost para acceso a la cámara web.

## 📝 Notas Técnicas

- **ONNX Runtime Web**: Usa WebAssembly para inferencia rápida en el navegador
- **Formato de Salida**: Los modelos YOLO v8/v11 devuelven tensores en formato `[1, channels, N]` que se transponen a `[N, channels]` para procesamiento
- **NMS (Non-Maximum Suppression)**: Implementado con IoU threshold de 0.7
- **Confianza Mínima**: 0.25 (configurable en `yolo.js`)

## 👤 Autor

**Alejandro Rebolledo**  
📧 arebolledo@udd.cl

Basado en la arquitectura YOLO de Ultralytics y ONNX Runtime Web.

## 📄 Licencia

Este proyecto está bajo la licencia **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**.

Ver el archivo `LICENSE` para más detalles.

## ⚠️ Descargo de Responsabilidad

ESTE SOFTWARE SE PROPORCIONA "TAL CUAL", SIN GARANTÍA DE NINGÚN TIPO, EXPRESA O IMPLÍCITA. EL USO DE ESTE CÓDIGO ES BAJO SU PROPIO RIESGO. EL AUTOR NO SE HACE RESPONSABLE DE NINGÚN DAÑO O PÉRDIDA QUE PUEDA SURGIR DEL USO DE ESTE SOFTWARE.
