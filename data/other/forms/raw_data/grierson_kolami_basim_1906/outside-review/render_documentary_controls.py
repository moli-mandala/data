"""Sequential small originals for the finite documentary retrieval queue."""
from pathlib import Path
import subprocess
import sys
for printed in (8, 9, 167, 185, 186, 283, 284, 290, 473, 474, 478, 577):
    physical = printed + (19 if printed <= 186 else 20)
    subprocess.run([sys.executable, str(Path(__file__).with_name('render_one.py')), str(printed), str(physical)], check=True)
