# YOLO Object Tracking & Depth Estimation

Este proyecto implementa un sistema de seguimiento de objetos en tiempo real utilizando **YOLOv11** y visualización de estelas de movimiento. Además, incluye una funcionalidad experimental para la estimación de profundidad utilizando el modelo **MiDaS (DPT_Hybrid)**.

## Características

- **Detección y Seguimiento**: Utiliza YOLOv11 para detectar y seguir objetos.
- **Estelas de Movimiento**: Dibuja una línea que muestra la trayectoria reciente de los objetos.
- **Estimación de Profundidad (Opcional)**: Estima la distancia relativa de los objetos utilizando MiDaS.
- **Dirección de Movimiento**: Indica si el objeto se mueve hacia arriba, abajo, izquierda o derecha.

## Requisitos

- Python 3.8+
- Cámara web

## Instalación

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

## Uso

Ejecutar el script principal:

```bash
python test.py
```

### Configuración

Para activar o desactivar la estimación de profundidad, edita la variable `ENABLE_DEPTH` en `test.py`:

```python
# test.py
ENABLE_DEPTH = True  # Para activar
ENABLE_DEPTH = False # Para desactivar (más rápido)
```

## Autor

**Alejandro Rebolledo**  
Correo: [arebolledo@udd.cl](mailto:arebolledo@udd.cl)

## Licencia

Este proyecto está bajo la licencia **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**.
Ver el archivo `LICENSE` para más detalles.

## Descargo de Responsabilidad

ESTE SOFTWARE SE PROPORCIONA "TAL CUAL", SIN GARANTÍA DE NINGÚN TIPO, EXPRESA O IMPLÍCITA. EL USO DE ESTE CÓDIGO ES BAJO SU PROPIO RIESGO. EL AUTOR NO SE HACE RESPONSABLE DE NINGÚN DAÑO O PROBLEMA DERIVADO DE SU USO.
