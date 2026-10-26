from pathlib import Path
import subprocess
p=Path("tmp/jambu-next-source/kolami-specimen-originals")
for leaf,page in [(586,565),(589,568),(590,569)]:
 f=p/f"alternate-leaf{leaf}-unverified.jpg"
 if not f.exists(): subprocess.run(["curl","-L","--fail","--max-time","60",f"https://archive.org/download/in.ernet.dli.2015.42222/page/n{leaf}.jpg","-o",str(f)],check=True)
 print(leaf,page,f.stat().st_size,flush=True)
