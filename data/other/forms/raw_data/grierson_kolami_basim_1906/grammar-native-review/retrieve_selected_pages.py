from pathlib import Path
import subprocess
p=Path("tmp/jambu-next-source/kolami-grammar-originals");p.mkdir(exist_ok=True,parents=True)
for leaf,page in [(583,562),(584,563),(587,566),(588,567)]:
 f=p/f"alternate-leaf{leaf}-unverified.jpg"
 if not f.exists(): subprocess.run(["curl","-L","--fail","--max-time","60",f"https://archive.org/download/in.ernet.dli.2015.42222/page/n{leaf}.jpg","-o",str(f)],check=True)
 print(leaf,page,f.stat().st_size,flush=True)
