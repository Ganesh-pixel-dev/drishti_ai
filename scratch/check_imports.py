import os
import sys
import django

# Add current directory to sys.path
sys.path.append(os.getcwd())

# Setup django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'drishti_ai.settings')
django.setup()

try:
    from video_module.analyzer import evaluate_video_final
    print("Import successful")
except Exception as e:
    print(f"Import failed: {e}")
    import traceback
    traceback.print_exc()
