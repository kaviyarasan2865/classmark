import sys
from pathlib import Path


# Streamlit Cloud runs this file with its own folder on the path, so the
# package import fails unless the repo root is added first.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
   sys.path.insert(0, str(_ROOT))


_UNDER_STREAMLIT = "streamlit" in sys.modules or any(
   "streamlit" in str(arg) for arg in sys.argv
)


if _UNDER_STREAMLIT:
   import runpy


   runpy.run_path(str(_ROOT / "app.py"), run_name="__main__")
else:
   from class_assessor.cli import main


   if __name__ == "__main__":
       main()



