# OAS OMR Marker — GitHub Pages (Client-side)

This version runs entirely in the browser using:
- **pdf.js** (render PDFs to canvas)
- **OpenCV.js** (alignment + circle detection + fill scoring)
- **pdf-lib** (optional annotated PDF output)

## Why this works on GitHub Pages
Everything runs client-side (static HTML/JS). GitHub Pages only serves the files.

## Folder workflow (recommended)
Because browsers can struggle with one huge PDF, use **one student per PDF** (2 pages per student).

1) Upload your student PDFs into:
- `data/students/`

2) Edit:
- `data/manifest.json`

Example:
```json
{
  "template": "data/template.pdf",
  "key": "data/key.pdf",
  "students": [
    "data/students/student_001.pdf",
    "data/students/student_002.pdf"
  ]
}
```

3) Put your teacher key at:
- `data/key.pdf`

4) Commit/push.

## Deploy on GitHub Pages
- Repo Settings -> Pages
- Source: Deploy from a branch
- Branch: main (root)

Then open your Pages site and run marking.

## Outputs
- `results.csv` with **exactly 2 columns**: student_number, score (out of 45)
- `annotated.pdf` (optional) with highlighted bubbles (green correct / red incorrect)

## Notes
- Scan around 300 dpi if possible.
- Avoid heavy shadows and page cropping.
- If a student number has multiple/blank marks, it may become UNKNOWN.
