from dataclasses import dataclass

@dataclass(frozen=True)
class OMRConfig:
    # Rendering DPI for PDF -> image
    dpi: int = 300

    # --- Relative regions (x1,y1,x2,y2) in [0..1] of (width,height) ---
    # If your scans are cropped/shifted, adjust these slightly.
    p1_answers_region = (0.05, 0.70, 0.95, 0.97)          # Q1–40 area
    p1_student_number_region = (0.40, 0.23, 0.86, 0.53)   # Student number grid area
    p2_first_column_region = (0.05, 0.12, 0.28, 0.58)     # Q41–80 block; we read Q41–45 only

    # Hough circle detection tuning
    hough_dp: float = 1.2
    hough_min_dist: int = 18
    hough_param1: int = 120
    hough_param2: int = 22
    hough_min_radius: int = 10
    hough_max_radius: int = 26

    # Fill detection thresholds (tweak if needed)
    answer_min_fill: float = 0.18     # below this treat as blank
    digit_min_fill: float = 0.12
    multi_delta: float = 0.06         # if best-second < delta => ambiguous/multiple

    # Student number digits order (top->bottom). Most grids are 0..9 top-down.
    digit_order = tuple(range(0, 10))

    # Expected structure
    q1_rows: int = 10
    q1_blocks: int = 4
    options: str = "ABCDE"

    # Page2 questions read
    q2_start: int = 41
    q2_count: int = 5
