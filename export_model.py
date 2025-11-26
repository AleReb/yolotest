#!/usr/bin/env python3
"""
Export YOLOv11n to ONNX format for web browser use
"""

from ultralytics import YOLO
import os
import sys
print(os.getcwd())
print(os.listdir())

def main():
    print("="*60)
    print("YOLO11n ONNX Export para Web")
    print("="*60)

    try:
        print("\n1. Descargando y cargando modelo YOLOv11n...")
        # YOLO descargará automáticamente si no existe
        model = YOLO("yolo11n.pt")  
        print("   OK: Modelo cargado/descargado")

        print("\n2. Exportando a formato ONNX...")
        print("   Configuracion:")
        print("   - Input size: 640x640")
        print("   - opset: 12")
        print("   - Simplify: True")

        onnx_path = model.export(
            format="onnx",
            imgsz=640,
            opset=12,
            simplify=True,
            dynamic=False,
            half=False
        )

        print(f"   OK: Exportado a: {onnx_path}")

        # Verificar archivo
        if os.path.exists(onnx_path):
            size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
            print(f"\n3. Verificando archivo...")
            print(f"   OK: Tamaño: {size_mb:.1f} MB")
            print(f"   OK: Ruta: {os.path.abspath(onnx_path)}")

            # Mover a yolo11n.onnx si es diferente
            target = "yolo11n.onnx"
            if onnx_path != target:
                import shutil
                print(f"\n4. Moviendo a {target}...")
                if os.path.exists(target):
                    os.remove(target)
                    print(f"   OK: Archivo anterior eliminado")

                shutil.move(onnx_path, target)
                print(f"   OK: Guardado como {target}")

            print("\n" + "="*60)
            print("EXITO: Modelo exportado correctamente")
            print("="*60)
            print(f"\nEl archivo 'yolo11n.onnx' esta listo para usar en web")
            print(f"Tamaño: {size_mb:.1f} MB")
            return 0

        else:
            print(f"ERROR: Archivo no encontrado")
            return 1

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
