import sys
from pathlib import Path


current_file = Path(__file__).resolve()
project_root = current_file.parent.parent 

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
