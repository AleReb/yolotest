#!/usr/bin/env python3
"""
Export YOLOv11n to ONNX format for web browser use
"""

"""
YOLO Model Export Script

Exports YOLOv11 models to ONNX format for web deployment.
Supports both Pose Estimation and Object Detection models.

Usage:
    python export_model.py

Modify the MODELS list below to export different models or sizes.

Author: Alejandro Rebolledo (arebolledo@udd.cl)
License: CC BY-NC 4.0
"""

from ultralytics import YOLO
import os
import sys
import shutil

def main():
    print("="*60)
    print("YOLO11n ONNX Export para Web (Robust)")
    print("="*60)
    
    cwd = os.getcwd()
    print(f"Working directory: {cwd}")

    try:
        print("\n1. Loading YOLOv11n-pose model...")
        model_path = os.path.join(cwd, "yolo11n-pose.pt")
        if not os.path.exists(model_path):
            print(f"   Downloading model to {model_path}...")
        
        model = YOLO("yolo11n-pose.pt")  
        print("   OK: Model loaded")

        configs = [
            {"name": "yolo11n-pose-320.onnx", "size": 320},
            {"name": "yolo11n-pose.onnx", "size": 640}
        ]

        for config in configs:
            print(f"\n--- Exporting {config['name']} ({config['size']}x{config['size']}) ---")
            
            # Export
            # Ultralytics exports to the same dir as the .pt file usually
            exported_path = model.export(
                format="onnx",
                imgsz=config['size'],
                opset=12,
                simplify=True,
                dynamic=False,
                half=False
            )
            
            print(f"   Export returned path: {exported_path}")
            
            # Ensure we have an absolute path
            if not os.path.isabs(exported_path):
                exported_path = os.path.abspath(exported_path)
            
            print(f"   Looking for file at: {exported_path}")

            if not os.path.exists(exported_path):
                print(f"   ERROR: Exported file not found at {exported_path}")
                # Try looking in current dir just in case
                local_path = os.path.join(cwd, os.path.basename(exported_path))
                if os.path.exists(local_path):
                    print(f"   Found it at {local_path} instead")
                    exported_path = local_path
                else:
                    return 1

            # Target path
            target_path = os.path.join(cwd, config['name'])
            
            # If the exported file IS the target file, we are done
            if os.path.abspath(exported_path) == os.path.abspath(target_path):
                print(f"   File is already in place: {target_path}")
            else:
                print(f"   Moving/Renaming to {target_path}...")
                if os.path.exists(target_path):
                    os.remove(target_path)
                shutil.move(exported_path, target_path)
                print(f"   OK: Moved")
            
            # Verify
            if os.path.exists(target_path):
                size_mb = os.path.getsize(target_path) / (1024 * 1024)
                print(f"   OK: Verified Size: {size_mb:.1f} MB")
            else:
                print(f"   ERROR: Target file missing after move!")
                return 1

        print("\n" + "="*60)
        print("EXITO: Modelos exportados correctamente")
        print("="*60)
        return 0

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
