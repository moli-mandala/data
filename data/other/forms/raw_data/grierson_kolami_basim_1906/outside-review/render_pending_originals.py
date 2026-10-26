"""Small selected outside leads, sequential original rendering only."""
from pathlib import Path
import subprocess
import sys
for printed in (349, 480, 626, 627):
    subprocess.run([sys.executable, str(Path(__file__).with_name('render_one.py')), str(printed)], check=True)
