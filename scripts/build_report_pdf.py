"""Render report.md to report.pdf via markdown -> styled HTML -> weasyprint."""
from pathlib import Path

import markdown
from weasyprint import HTML, CSS

MD_PATH = Path("report.md")
PDF_PATH = Path("report.pdf")

CSS_TEXT = """
@page { size: A4; margin: 18mm 16mm; @bottom-right { content: counter(page) " / " counter(pages); font-family: 'Helvetica', sans-serif; font-size: 9pt; color: #888; } }
body { font-family: 'Helvetica', 'Arial', sans-serif; font-size: 10pt; line-height: 1.45; color: #222; }
h1 { font-size: 20pt; color: #1a1a1a; margin: 0 0 4pt; }
h2 { font-size: 14pt; color: #1a1a1a; margin: 18pt 0 6pt; border-bottom: 1px solid #ccc; padding-bottom: 3pt; }
h3 { font-size: 11pt; color: #333; margin: 12pt 0 4pt; }
h4 { font-size: 10pt; color: #555; margin: 8pt 0 2pt; font-weight: 600; }
p { margin: 4pt 0 6pt; }
ul, ol { margin: 4pt 0 6pt 14pt; padding-left: 8pt; }
li { margin: 2pt 0; }
code { font-family: 'Menlo', 'Courier New', monospace; font-size: 9pt; background: #f4f4f4; padding: 1pt 3pt; border-radius: 2pt; }
pre { font-family: 'Menlo', 'Courier New', monospace; font-size: 8.5pt; background: #f8f8f8; padding: 6pt 8pt; border-left: 3pt solid #4a90e2; overflow-x: auto; line-height: 1.35; page-break-inside: avoid; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; margin: 8pt 0; font-size: 9pt; width: 100%; page-break-inside: avoid; }
th, td { border: 1px solid #ccc; padding: 4pt 6pt; text-align: left; vertical-align: top; }
th { background: #f0f0f0; font-weight: 600; }
strong { color: #1a1a1a; }
blockquote { margin: 6pt 0; padding: 4pt 10pt; border-left: 3pt solid #ccc; color: #555; font-style: italic; }
hr { border: none; border-top: 1px solid #ddd; margin: 12pt 0; }
a { color: #1a73e8; text-decoration: none; }
"""


def main() -> None:
    md_text = MD_PATH.read_text()
    html_body = markdown.markdown(
        md_text,
        extensions=["extra", "tables", "fenced_code", "codehilite"],
        output_format="html5",
    )
    html_doc = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>Pop Lyrics — Final Report</title></head>
<body>{html_body}</body></html>"""
    HTML(string=html_doc).write_pdf(str(PDF_PATH), stylesheets=[CSS(string=CSS_TEXT)])
    print(f"Wrote {PDF_PATH} ({PDF_PATH.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
