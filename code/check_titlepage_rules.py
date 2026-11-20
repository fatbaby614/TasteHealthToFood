r"""Inspect main.pdf title pages for rules/drawings overflowing the text area.

Diagnostic for the persistent pdflatex warning:
  Overfull \hbox (123.62721pt too wide) detected at line 157 (\maketitle)

Renders pages 1-3 to PNG and reports every vector drawing bbox against the
cas-dc geometry margins (hmargin=18.1mm, vmargin top=19.5mm / bottom=18.2mm).
Read-only: does not modify the paper or results.
"""
import sys

import fitz

PDF = r"d:\TanHuangWork\TasteHealthToFood\paper\main.pdf"
OUT = r"d:\TanHuangWork\TasteHealthToFood\outputs\figures\titlepage_check"

MM = 72 / 25.4


def main():
    doc = fitz.open(PDF)
    print("pages:", doc.page_count)
    for pno in range(min(4, doc.page_count)):
        page = doc[pno]
        r = page.rect
        ml = 18.1 * MM
        mr = r.width - ml
        mt = 19.5 * MM
        mb = r.height - 18.2 * MM
        print(f"--- page {pno + 1}: {r.width:.0f}x{r.height:.0f} pt; "
              f"text area x[{ml:.1f},{mr:.1f}] y[{mt:.1f},{mb:.1f}]")
        n_out = 0
        for d in page.get_drawings():
            x0, y0, x1, y1 = d["rect"]
            w = x1 - x0
            h = y1 - y0
            outside = (x0 < ml - 2 or x1 > mr + 2 or y0 < mt - 2 or y1 > mb + 2)
            if abs(h) <= 6 and w >= 80:
                tag = "H-RULE"
            elif abs(w) <= 6 and h >= 80:
                tag = "V-RULE"
            else:
                tag = f"{d['type']}({w:.0f}x{h:.0f})"
            if outside:
                n_out += 1
                print(f"  [{tag}] OUTSIDE bbox=({x0:.1f},{y0:.1f})-({x1:.1f},{y1:.1f}) "
                      f"stroke_w={d.get('width')} color={d.get('color')}")
        if n_out == 0:
            print("  no drawing outside the text area")
        pix = page.get_pixmap(dpi=150)
        import os
        os.makedirs(OUT, exist_ok=True)
        out_png = os.path.join(OUT, f"page{pno + 1}.png")
        pix.save(out_png)
        print("  rendered ->", out_png)


if __name__ == "__main__":
    sys.exit(main())
