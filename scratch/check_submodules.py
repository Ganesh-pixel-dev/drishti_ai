import os
import sys
import django

sys.path.append(os.getcwd())
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'drishti_ai.settings')
django.setup()

try:
    from video_module import detectors
    print(f"detectors is: {detectors}")
    
    from video_module.detectors import heartbeat_detector
    print("Import heartbeat_detector successful")
except Exception as e:
    print(f"Import failed: {e}")
    import traceback
    traceback.print_exc()
