# OAS OMR Marker (Batch PDF → CSV + Annotated PDF)

Marks a custom optical answer sheet (OAS) from scanned PDFs.

## What it does
- Reads a **teacher key** OAS PDF (2 pages) with correct answers shaded
- Reads a **batch** PDF containing many students:
  - 2 pages per student (page 1 then page 2)
- Decodes:
  - Student number bubble grid (page 1)
  - Q1–40 on page 1
  - Q41–45 on page 2 (first 5 questions in the first column)
- Outputs:
  - CSV with exactly 2 columns: `student_number, score` (out of 45)
  - Annotated PDF to visually verify detection

## Install
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Prepare files
1) Put your blank OAS template here:
- `assets/template.pdf`

(If you don't have a blank template yet, you can temporarily use a clean scan. You can replace it later.)

2) Prepare:
- `key.pdf` (teacher key OAS, 2 pages)
- `students.pdf` (batch scan, 2 pages per student; e.g., 42 students → 84 pages)

## Run
```bash
python mark_batch.py --template assets/template.pdf --key key.pdf --batch students.pdf --out output/results.csv --annotated output/annotated.pdf
```

## Output
- `output/results.csv` → 2 columns: student_number, score
- `output/annotated.pdf` → annotated pages for checking

## If it doesn't detect bubbles on the first run
Different scanners sometimes shift/crop pages slightly. If you see errors like “too few circles detected”:
1) Run with a forced rebuild:
```bash
python mark_batch.py --template assets/template.pdf --key key.pdf --batch students.pdf --out output/results.csv --annotated output/annotated.pdf --rebuild-layout
```
2) If it still fails, adjust the regions in `omr/config.py` (3 rectangles) and rerun.

## Notes for best accuracy
- Scan around 300 dpi if possible
- Avoid heavy shadows / page cropping
- Have students shade firmly (2B pencil is great)

## License
MIT
