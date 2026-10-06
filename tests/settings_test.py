import os

os.environ.setdefault("DJANGO_DEBUG", "1")
os.environ.setdefault("DJANGO_SECRET_KEY", "test-key")
os.environ["SERPER_API_KEY"] = ""

from drishti_ai.settings import *  # noqa: E402,F401,F403
