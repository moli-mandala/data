"""High-resolution, column-separated OCR aid for the printed Kor–English index.

The output is deliberately source-local review material, not an installed inventory.
"""

import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image


ROOT = Path(__file__).resolve().parent
PDF = Path("/private/tmp/cust-norton-korku-1884-google.pdf")


def main():
    out = []
    with TemporaryDirectory(prefix="norton-reverse-") as temporary:
        temp = Path(temporary)
        for printed_page in range(172, 178):
            pdf_page = printed_page + 19
            png = temp / f"p{printed_page}.png"
            subprocess.run(
                ["pdftoppm", "-f", str(pdf_page), "-l", str(pdf_page),
                 "-r", "260", "-png", "-singlefile", str(PDF), str(png.with_suffix(""))],
                check=True,
            )
            image = Image.open(png)
            width, height = image.size
            top = .305 if printed_page == 172 else .09
            bottom = .73 if printed_page == 177 else .90
            bounds = {
                172: (("left", .17, .54), ("right", .54, .94)),
                173: (("left", .07, .46), ("right", .42, .91)),
                174: (("left", .17, .54), ("right", .54, .94)),
                175: (("left", .06, .42), ("right", .42, .89)),
                176: (("left", .17, .54), ("right", .54, .94)),
                177: (("left", .06, .43), ("right", .38, .89)),
            }[printed_page]
            for column, left, right in bounds:
                crop = temp / f"p{printed_page}-{column}.png"
                image.crop((int(width * left), int(height * top),
                            int(width * right), int(height * bottom))).save(crop)
                result = subprocess.run(
                    ["tesseract", str(crop), "stdout", "--psm", "6"],
                    check=True, text=True, capture_output=True,
                )
                out.append(f"# printed p{printed_page}, {column} column\n{result.stdout.strip()}\n")
                crop.unlink()
            image.close()
            png.unlink()
    (ROOT / "reverse_column_ocr.txt").write_text("\n".join(out), encoding="utf-8")
    print("OCR aid saved; visually check every entry and each column boundary")


if __name__ == "__main__":
    main()
