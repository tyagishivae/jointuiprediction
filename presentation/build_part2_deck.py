from pathlib import Path
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymupdf
from PIL import Image, ImageOps
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent
ASSET_DIR = OUT_DIR / ".part2_assets"
ASSET_DIR.mkdir(exist_ok=True)
PDF_PATH = ROOT.parent / "UG_Presentation.pdf"
HARDNESS_PATH = ROOT / "hardness_experimental.csv"
UTS_DENSITY_PATH = ROOT / "uts_density_experimental.csv"
LOOCV_PATH = ROOT / "joint_train_loocv_metrics.csv"
HOLDOUT_PATH = ROOT / "joint_holdout_validation_results.csv"
INVERSE_PATH = ROOT / "joint_inverse_holdout_recovery_with_percent_errors.csv"
CANDIDATE_PATH = ROOT / "joint_inverse_pareto_candidates.csv"
OUTPUT_PATH = OUT_DIR / "UG_Presentation_Part2_Experimental_Validation_Expanded.pptx"
SETUP_DIR = OUT_DIR / "experiental setup"
PRINTING_SOURCE = OUT_DIR / "WhatsApp Image 2026-11-08 at 08.34.25.jpeg"

NAVY = "173B5E"
BLUE = "2C6E91"
TEAL = "3D8791"
GOLD = "B88936"
BURGUNDY = "8C3A4B"
INK = "202A34"
MID = "586775"
PALE = "EAF0F4"
PALE_BLUE = "EAF3F7"
PALE_GOLD = "F7F1E5"
PALE_RED = "F7ECEE"
WHITE = "FFFFFF"
GRID = "D7E0E7"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 15,
    "axes.labelsize": 11,
    "axes.edgecolor": "#586775",
    "axes.labelcolor": "#202A34",
    "xtick.color": "#586775",
    "ytick.color": "#586775",
    "text.color": "#202A34",
    "axes.grid": True,
    "grid.color": "#D7E0E7",
    "grid.linewidth": 0.7,
})

hardness = pd.read_csv(HARDNESS_PATH)
uts_density = pd.read_csv(UTS_DENSITY_PATH)
metrics = pd.read_csv(LOOCV_PATH).set_index("Property")
holdout = pd.read_csv(HOLDOUT_PATH)
inverse = pd.read_csv(INVERSE_PATH)
candidates = pd.read_csv(CANDIDATE_PATH)
data = hardness.merge(uts_density, on="Sample", validate="one_to_one")


# Extract the original IIT (BHU) mark and two project-owned process photos.
doc = pymupdf.open(PDF_PATH)
cover = doc[0]
cover_images = cover.get_images(full=True)
if cover_images:
    extracted = doc.extract_image(cover_images[0][0])
    LOGO_PATH = ASSET_DIR / f"iit_bhu_logo.{extracted['ext']}"
    LOGO_PATH.write_bytes(extracted["image"])
else:
    LOGO_PATH = None

process_page = doc[16]
printer_path = ASSET_DIR / "eos_m290_photo.png"
coupons_path = ASSET_DIR / "printed_coupons_photo.png"
pymupdf.Matrix(2, 2)
process_page.get_pixmap(
    matrix=pymupdf.Matrix(2, 2),
    clip=pymupdf.Rect(468, 42, 695, 238),
    alpha=False,
).save(printer_path)
process_page.get_pixmap(
    matrix=pymupdf.Matrix(2, 2),
    clip=pymupdf.Rect(92, 264, 450, 395),
    alpha=False,
).save(coupons_path)
doc.close()


def normalize_photo(source, destination):
    with Image.open(source) as image:
        ImageOps.exif_transpose(image).convert("RGB").save(destination, quality=94)
    return destination


printing_photo_path = normalize_photo(
    PRINTING_SOURCE,
    ASSET_DIR / "actual_lpbf_printing.jpg",
)
uts_setup_path = normalize_photo(
    SETUP_DIR / "IMG_20261106_113041843_HDR.jpg.jpeg",
    ASSET_DIR / "tinius_olsen_uts_setup.jpg",
)
uts_full_path = normalize_photo(
    SETUP_DIR / "IMG_20261106_113114545_HDR.jpg.jpeg",
    ASSET_DIR / "tinius_olsen_full_frame.jpg",
)
hardness_setup_path = normalize_photo(
    SETUP_DIR / "IMG_20261106_113944611_HDR.jpg.jpeg",
    ASSET_DIR / "mitutoyo_hm200_hardness_setup.jpg",
)
hardness_microscope_path = normalize_photo(
    SETUP_DIR / "IMG_20261106_113502034_HDR.jpg.jpeg",
    ASSET_DIR / "hardness_optical_microscope.jpg",
)


# Charts for the experiment and validation slides.
def chart_path(name):
    return ASSET_DIR / name


fig, ax = plt.subplots(figsize=(11.2, 4.2))
for split, color, marker in (("train", "#2C6E91", "o"), ("validation", "#8C3A4B", "D")):
    subset = hardness.loc[hardness["Split"] == split]
    ax.scatter(
        subset["Sample"],
        subset["Hardness_HV"],
        s=68 if split == "train" else 86,
        c=color,
        marker=marker,
        edgecolor="white",
        linewidth=0.8,
        label=f"{split.title()} sample",
        zorder=3,
    )
ax.axhline(hardness.loc[hardness["Split"] == "train", "Hardness_HV"].mean(), color="#B88936", linestyle="--", linewidth=1.5, label="Training mean")
ax.set(title="Measured composite hardness by sample", xlabel="Sample", ylabel="Hardness (HV)")
ax.set_xticks(range(1, 21))
ax.set_ylim(300, 360)
ax.legend(frameon=False, ncol=3, loc="upper right")
fig.tight_layout()
fig.savefig(chart_path("hardness_by_sample.png"), dpi=190, facecolor="white")
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.25))
for ax, target, title, ylabel in (
    (axes[0], "UTS_MPa", "Ultimate tensile strength", "UTS (MPa)"),
    (axes[1], "RelativeDensity_pct", "Relative density", "Density (%)"),
):
    for split, color, marker in (("train", "#2C6E91", "o"), ("validation", "#8C3A4B", "D")):
        samples = data.loc[data["Split"] == split]
        ax.scatter(samples["Sample"], samples[target], c=color, marker=marker, s=60 if split == "train" else 82, edgecolor="white", linewidth=0.8, label=split.title(), zorder=3)
    ax.set(title=title, xlabel="Sample", ylabel=ylabel)
    ax.set_xticks([1, 4, 7, 10, 11, 14, 17, 20])
    ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(chart_path("uts_density_by_sample.png"), dpi=190, facecolor="white")
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.1))
for ax, target, title, ylabel in (
    (axes[0], "Hardness_HV", "Hardness", "HV"),
    (axes[1], "UTS_MPa", "UTS", "MPa"),
    (axes[2], "RelativeDensity_pct", "Relative density", "%"),
):
    subset = holdout.loc[holdout["Property"] == target]
    x = np.arange(len(subset))
    width = 0.34
    ax.bar(x - width / 2, subset["Actual_Mean"], width, color="#173B5E", label="Measured")
    ax.bar(x + width / 2, subset["Predicted"], width, color="#B88936", label="Predicted")
    ax.set_xticks(x, ["6 repeats", "Sample 11"])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, fontsize=9)
    for i, err in enumerate(subset["ErrorPercent"]):
        y_top = max(subset["Actual_Mean"].iloc[i], subset["Predicted"].iloc[i])
        ax.annotate(f"{err:+.2f}%", (i, y_top), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=9, color="#8C3A4B")
fig.tight_layout()
fig.savefig(chart_path("holdout_actual_predicted.png"), dpi=190, facecolor="white")
plt.close(fig)

fig, ax = plt.subplots(figsize=(10.8, 3.8))
properties = ["Hardness_HV", "UTS_MPa", "RelativeDensity_pct"]
labels = ["Hardness", "UTS", "Density"]
colors = ["#2C6E91", "#B88936", "#3D8791"]
x = np.arange(len(properties))
bar_width = 0.29
for offset, (condition, color) in enumerate(((1, "#173B5E"), (2, "#8C3A4B"))):
    row = inverse.loc[inverse["ValidationCondition"] == condition].iloc[0]
    values = [row[f"PropertyErrorPercent_{prop}"] for prop in properties]
    ax.bar(x + (offset - 0.5) * bar_width, values, bar_width, color=color, label="6-repeat condition" if condition == 1 else "Sample 11")
ax.axhline(0, color="#586775", linewidth=1)
ax.set_xticks(x, labels)
ax.set_ylabel("Signed property error (%)")
ax.set_title("Inverse recovery: property error at recovered settings")
ax.legend(frameon=False, ncol=2)
fig.tight_layout()
fig.savefig(chart_path("inverse_property_errors.png"), dpi=190, facecolor="white")
plt.close(fig)


prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
blank = prs.slide_layouts[6]


def rgb(hex_value):
    return RGBColor.from_string(hex_value)


def set_background(slide, color=WHITE):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = rgb(color)


def add_text(slide, text, x, y, w, h, size=18, color=INK, bold=False, align=PP_ALIGN.LEFT, font="Aptos", margin=0.04):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = shape.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = Inches(margin)
    frame.margin_right = Inches(margin)
    frame.margin_top = Inches(margin)
    frame.margin_bottom = Inches(margin)
    paragraph = frame.paragraphs[0]
    paragraph.alignment = align
    run = paragraph.add_run()
    run.text = text
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = rgb(color)
    return shape


def add_title(slide, title, subtitle=None, number=None):
    add_text(slide, title, 0.62, 0.28, 12.0, 0.52, size=26, color=NAVY, bold=True)
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.65), Inches(0.91), Inches(12.0), Inches(0.045))
    line.fill.solid()
    line.fill.fore_color.rgb = rgb(GOLD)
    line.line.fill.background()
    if subtitle:
        add_text(slide, subtitle, 0.68, 1.03, 11.9, 0.34, size=11.5, color=MID)
    if number is not None:
        add_footer(slide, number)


def add_footer(slide, number):
    add_text(slide, "ME-392  |  Haynes 282  |  Part II", 0.62, 7.18, 8.5, 0.18, size=8.5, color=MID)
    add_text(slide, str(number), 12.25, 7.14, 0.35, 0.2, size=9, color=MID, align=PP_ALIGN.RIGHT)


def add_card(slide, x, y, w, h, title, body, accent=BLUE, body_size=15):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = rgb(PALE)
    shape.line.color.rgb = rgb(GRID)
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(0.07), Inches(h))
    bar.fill.solid()
    bar.fill.fore_color.rgb = rgb(accent)
    bar.line.fill.background()
    add_text(slide, title, x + 0.22, y + 0.18, w - 0.38, 0.34, size=16, color=accent, bold=True)
    add_text(slide, body, x + 0.22, y + 0.62, w - 0.38, h - 0.76, size=body_size, color=INK)


def add_picture(slide, path, x, y, w, h):
    shape = slide.shapes.add_picture(str(path), Inches(x), Inches(y))
    box_width = Inches(w)
    box_height = Inches(h)
    scale = min(box_width / shape.width, box_height / shape.height)
    shape.width = int(shape.width * scale)
    shape.height = int(shape.height * scale)
    shape.left = Inches(x) + int((box_width - shape.width) / 2)
    shape.top = Inches(y) + int((box_height - shape.height) / 2)
    return shape


def format_cell(cell, text, size=11, color=INK, bold=False, fill=WHITE, align=PP_ALIGN.CENTER):
    cell.text = str(text)
    cell.fill.solid()
    cell.fill.fore_color.rgb = rgb(fill)
    cell.margin_left = Inches(0.05)
    cell.margin_right = Inches(0.05)
    cell.margin_top = Inches(0.04)
    cell.margin_bottom = Inches(0.04)
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    for paragraph in cell.text_frame.paragraphs:
        paragraph.alignment = align
        for run in paragraph.runs:
            run.font.name = "Aptos"
            run.font.size = Pt(size)
            run.font.bold = bold
            run.font.color.rgb = rgb(color)


def add_table(slide, x, y, w, h, headers, rows, widths=None, font_size=11):
    table_shape = slide.shapes.add_table(len(rows) + 1, len(headers), Inches(x), Inches(y), Inches(w), Inches(h))
    table = table_shape.table
    if widths:
        for col, width in enumerate(widths):
            table.columns[col].width = Inches(width)
    for col, header in enumerate(headers):
        format_cell(table.cell(0, col), header, size=font_size, color=WHITE, bold=True, fill=NAVY)
    for row_index, row in enumerate(rows, start=1):
        fill = PALE if row_index % 2 == 0 else WHITE
        for col, value in enumerate(row):
            format_cell(table.cell(row_index, col), value, size=font_size, fill=fill)
    return table_shape


def add_bullet(slide, text, x, y, w, size=16, color=INK, bullet_color=GOLD):
    add_text(slide, "•", x, y, 0.24, 0.36, size=size + 2, color=bullet_color, bold=True)
    add_text(slide, text, x + 0.27, y, w - 0.27, 0.54, size=size, color=color)


# Slide 1: matching project identity and cover structure.
slide = prs.slides.add_slide(blank)
set_background(slide)
add_text(slide, "UNDERGRADUATE PROJECT (ME-392)", 0.7, 0.32, 11.95, 0.28, size=15, color=NAVY, bold=True, align=PP_ALIGN.CENTER)
add_text(slide, "Prediction of Properties of Additively\nManufactured Parts for Haynes 282", 0.72, 0.78, 11.9, 0.9, size=28, color=INK, bold=True, align=PP_ALIGN.CENTER)
add_text(slide, "PART II  |  EXPERIMENTAL VALIDATION AND JOINT INVERSE DESIGN", 0.82, 1.72, 11.7, 0.35, size=14, color=BLUE, bold=True, align=PP_ALIGN.CENTER)
if LOGO_PATH:
    add_picture(slide, LOGO_PATH, 5.72, 2.12, 1.9, 1.82)
add_text(slide, "Taha Hassan Khan (23135147)  |  Shivarpan Tyagi (23135151)  |  Shivam Kumar (23135152)", 0.8, 4.18, 11.75, 0.3, size=12, color=INK, bold=True, align=PP_ALIGN.CENTER)
add_text(slide, "Prachi Singh (23135158)  |  Satyam Kumar Mishra (23135162)", 0.8, 4.52, 11.75, 0.3, size=12, color=INK, bold=True, align=PP_ALIGN.CENTER)
rule = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(2.15), Inches(5.0), Inches(9.05), Inches(0.025))
rule.fill.solid(); rule.fill.fore_color.rgb = rgb(GOLD); rule.line.fill.background()
add_text(slide, "Under the guidance of", 1.0, 5.25, 11.3, 0.28, size=14, color=MID, align=PP_ALIGN.CENTER)
add_text(slide, "Dr. Pawan Sharma", 1.0, 5.56, 11.3, 0.34, size=19, color=NAVY, bold=True, align=PP_ALIGN.CENTER)
add_text(slide, "Department of Mechanical Engineering  |  Indian Institute of Technology (BHU), Varanasi", 1.0, 5.96, 11.3, 0.32, size=13, color=INK, align=PP_ALIGN.CENTER)
add_text(slide, "NOVEMBER 2026", 1.0, 6.78, 11.3, 0.22, size=9, color=MID, align=PP_ALIGN.CENTER)

# Slide 2: clear continuation from the first presentation.
slide = prs.slides.add_slide(blank); set_background(slide); add_title(slide, "Project extension", "From a broad process-property workflow to a mechanically validated, multi-property study", 2)
add_card(slide, 0.72, 1.72, 5.75, 2.05, "PART I  |  FOUNDATION", "Four LPBF inputs\nRelative-density prediction and process mapping\nInitial forward and inverse application", accent=BLUE, body_size=17)
add_card(slide, 6.83, 1.72, 5.75, 2.05, "PART II  |  THIS STUDY", "Measured hardness and UTS added to relative density\nGrouped holdout checks at two process settings\nJoint inverse search for diverse candidates", accent=TEAL, body_size=17)
add_bullet(slide, "20 experimental records cover 15 unique process settings.", 1.0, 4.22, 11.4, 17)
add_bullet(slide, "The 500-point set is synthetic augmentation generated from training-only models; it is not 500 additional independent experiments.", 1.0, 4.86, 11.4, 16)
add_bullet(slide, "This deck reports the new mechanical-property checks separately from the earlier broad-dataset model scores.", 1.0, 5.62, 11.4, 16)

# Slide 3: experiment context and split, with original project-owned images.
slide = prs.slides.add_slide(blank); set_background(slide); add_title(slide, "Experimental campaign and validation design", "Real measurements are grouped by process setting before training or validation", 3)
add_picture(slide, printing_photo_path, 0.72, 1.65, 4.1, 2.4)
add_picture(slide, coupons_path, 0.72, 4.55, 4.1, 1.68)
add_card(slide, 5.25, 1.62, 3.25, 1.42, "20 RECORDS", "15 unique settings\nAll layer thicknesses: 0.04 mm", accent=BLUE, body_size=15)
add_card(slide, 8.85, 1.62, 3.65, 1.42, "13 TRAIN SETTINGS", "Used for training-only LOOCV and the validation-fold surrogate", accent=TEAL, body_size=14)
add_card(slide, 5.25, 3.35, 3.25, 1.72, "6-REPLICATE HOLDOUT", "270 W | 1500 mm/s\n0.08 mm hatch\n6 repeated samples = 1 setting", accent=BURGUNDY, body_size=14)
add_card(slide, 8.85, 3.35, 3.65, 1.72, "SAMPLE 11 HOLDOUT", "280 W | 1550 mm/s\n0.085 mm hatch\n1 independent setting", accent=GOLD, body_size=14)
add_text(slide, "Hardness composite: mean of the perpendicular-group mean and parallel-group mean; each orientation has equal weight.", 5.3, 5.48, 7.1, 0.55, size=13, color=INK, bold=True)
add_text(slide, "Hardness readings are in HV; UTS in MPa; relative density in percent.", 5.3, 6.13, 7.1, 0.35, size=11, color=MID)

add_text(slide, "LIVE LPBF BUILD", 0.78, 4.08, 4.0, 0.22, size=10, color=TEAL, bold=True, align=PP_ALIGN.CENTER)

# Slide 4: UTS test setup.
slide = prs.slides.add_slide(blank); set_background(slide); add_title(slide, "UTS experimental setup", "Tensile testing on the Tinius Olsen frame", 4)
add_picture(slide, uts_setup_path, 0.72, 1.52, 3.8, 4.95)
add_picture(slide, uts_full_path, 4.72, 1.52, 3.1, 4.95)
add_card(slide, 8.15, 1.65, 4.25, 1.12, "TEST MACHINE", "Tinius Olsen H25KL tensile testing frame", accent=BLUE, body_size=15)
add_card(slide, 8.15, 3.02, 4.25, 1.45, "MEASUREMENT", "Each specimen is loaded in tension while force and extension are recorded. UTS is derived from peak load divided by the original cross-sectional area.", accent=GOLD, body_size=14)
add_card(slide, 8.15, 4.72, 4.25, 1.15, "DATA LINK", "UTS measurements are matched to the same laser power, scan speed, hatch distance, and layer thickness as hardness and density.", accent=TEAL, body_size=13)

# Slide 5: hardness test setup.
slide = prs.slides.add_slide(blank); set_background(slide); add_title(slide, "Hardness experimental setup", "Mitutoyo HM-200 with optical indentation measurement", 5)
add_picture(slide, hardness_setup_path, 0.75, 1.52, 3.55, 4.92)
add_picture(slide, hardness_microscope_path, 4.48, 1.52, 3.55, 4.92)
add_card(slide, 8.35, 1.65, 4.0, 1.18, "HARDNESS INSTRUMENT", "Mitutoyo HM-200 hardness tester", accent=BLUE, body_size=16)
add_card(slide, 8.35, 3.08, 4.0, 1.55, "READING STRATEGY", "Multiple indent readings were taken in two orientations: six perpendicular and three parallel readings per sample.", accent=GOLD, body_size=14)
add_card(slide, 8.35, 4.88, 4.0, 1.2, "COMPOSITE VALUE", "Hardness_HV = mean of the two orientation means, giving each orientation equal weight.", accent=TEAL, body_size=13)

# Slide 6: reserved area for an actual defective-part image.
slide = prs.slides.add_slide(blank); set_background(slide); add_title(slide, "Post-build inspection", "Reserved for a representative defective part or defect micrograph", 6)
placeholder = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.9), Inches(1.55), Inches(11.55), Inches(4.85))
placeholder.fill.solid(); placeholder.fill.fore_color.rgb = rgb(PALE_BLUE)
placeholder.line.color.rgb = rgb(BLUE); placeholder.line.width = Pt(2)
add_text(slide, "ADD DEFECTIVE PART / MICROGRAPH IMAGE", 1.35, 3.08, 10.65, 0.5, size=24, color=BLUE, bold=True, align=PP_ALIGN.CENTER)
add_text(slide, "Replace this area with a post-build photo or microscopy image. Add sample ID, process setting, and observed defect type in the caption.", 1.75, 3.75, 9.85, 0.75, size=15, color=MID, align=PP_ALIGN.CENTER)

# Slide 7: reproducible modeling pipeline.
slide = prs.slides.add_slide(blank); set_background(slide); add_title(slide, "Methodology | leakage-aware training pipeline", "Real process settings are kept separate from synthetic augmentation", 7)
method_cards = [
    (0.78, 1.62, "1  |  ALIGN", "20 paired records\n15 unique settings\nFour process inputs + three measured properties", BLUE),
    (4.72, 1.62, "2  |  GROUP SPLIT", "13 training settings\n2 held-out settings\nAll repeated settings stay together", BURGUNDY),
    (8.66, 1.62, "3  |  TEACHER", "Ridge models\nLeave-one-condition-out residuals\nTraining fold only", GOLD),
    (0.78, 3.72, "4  |  AUGMENT", "Generate 500 process points\nWithin training support\nBootstrap centered residuals", TEAL),
    (4.72, 3.72, "5  |  STUDENT", "Extra Trees regressors\nSynthetic coverage + weighted real anchors\nOne model per property", BLUE),
    (8.66, 3.72, "6  |  EVALUATE", "Predict held-out measured settings\nReport RMSE, MAE, R², and % error\nThen refit on all 15 settings", BURGUNDY),
]
for x, y, title, body, accent in method_cards:
    add_card(slide, x, y, 3.72, 1.65, title, body, accent=accent, body_size=12)
add_text(slide, "The 500 points are model-generated augmentation, not 500 new experiments.", 1.0, 6.12, 11.25, 0.36, size=14, color=NAVY, bold=True, align=PP_ALIGN.CENTER)

# Slide 8: rationale for the method and its limits.
slide = prs.slides.add_slide(blank); set_background(slide); add_title(slide, "Methodology | why this design", "A practical compromise for sparse experiments, with limits stated explicitly", 8)
add_card(slide, 0.8, 1.65, 5.75, 1.62, "GROUP BY PROCESS SETTING", "Replicate coupons at identical settings are not split across training and validation; this prevents an overly optimistic test.", accent=BLUE, body_size=14)
add_card(slide, 6.8, 1.65, 5.75, 1.62, "HOLD OUT BEFORE SYNTHESIS", "The two validation settings are excluded before the 500 synthetic rows are generated, so their measurements do not seed the validation model.", accent=BURGUNDY, body_size=14)
add_card(slide, 0.8, 3.65, 5.75, 1.62, "AUGMENT COVERAGE, NOT EVIDENCE", "Synthetic points help the student model cover the measured parameter region; only real measurements can establish experimental performance.", accent=GOLD, body_size=14)
add_card(slide, 6.8, 3.65, 5.75, 1.62, "REPORT WEAK RESULTS HONESTLY", "Negative LOOCV R² and a weak UTS miss at sample 11 indicate that more unique experimental settings are needed before strong claims.", accent=TEAL, body_size=14)
add_text(slide, "Inverse optimization proposes experiments; it does not experimentally certify the recommended parameters.", 0.95, 5.82, 11.5, 0.48, size=15, color=NAVY, bold=True, align=PP_ALIGN.CENTER)

# Slide 9: hardness distribution across all sample records.
slide = prs.slides.add_slide(blank); set_background(slide); add_title(slide, "Hardness measurements", "Orientation-averaged composite hardness; validation settings are highlighted", 9)
add_picture(slide, chart_path("hardness_by_sample.png"), 0.75, 1.55, 8.15, 4.65)
add_card(slide, 9.18, 1.75, 3.4, 1.22, "TRAIN MEAN", f"{hardness.loc[hardness['Split'] == 'train', 'Hardness_HV'].mean():.1f} HV", accent=BLUE, body_size=23)
rep = hardness.loc[hardness["Split"] == "validation"].groupby(["LaserPower_W", "ScanSpeed_mm_s", "HatchDistance_mm"])["Hardness_HV"].agg(["mean", "std", "count"]).reset_index().iloc[0]
add_card(slide, 9.18, 3.2, 3.4, 1.42, "6-REPEAT HOLDOUT", f"Mean {rep['mean']:.2f} HV\nSD {rep['std']:.2f} HV", accent=BURGUNDY, body_size=16)
add_card(slide, 9.18, 4.85, 3.4, 1.16, "SAMPLE 11", f"{hardness.loc[hardness['Sample'] == 11, 'Hardness_HV'].iloc[0]:.2f} HV", accent=GOLD, body_size=21)
add_text(slide, "One source reading was confirmed as 320 HV before the composite was calculated.", 0.9, 6.36, 11.7, 0.3, size=10.5, color=MID)

# Slide 10: measured UTS and density over sample IDs.
slide = prs.slides.add_slide(blank); set_background(slide); add_title(slide, "Measured UTS and relative density", "The same experimental settings link the mechanical targets", 10)
add_picture(slide, chart_path("uts_density_by_sample.png"), 0.8, 1.62, 11.8, 4.4)
rep_data = data.loc[data["Split"] == "validation"].groupby(["LaserPower_W", "ScanSpeed_mm_s", "HatchDistance_mm"])[["UTS_MPa", "RelativeDensity_pct"]].mean().reset_index().iloc[0]
sample11 = data.loc[data["Sample"] == 11].iloc[0]
add_text(slide, f"Six-repeat condition mean: UTS {rep_data['UTS_MPa']:.1f} MPa  |  density {rep_data['RelativeDensity_pct']:.3f}%", 1.0, 6.18, 11.3, 0.25, size=13, color=NAVY, bold=True, align=PP_ALIGN.CENTER)
add_text(slide, f"Sample 11: UTS {sample11['UTS_MPa']:.0f} MPa  |  density {sample11['RelativeDensity_pct']:.2f}%", 1.0, 6.48, 11.3, 0.25, size=13, color=BURGUNDY, bold=True, align=PP_ALIGN.CENTER)

# Slide 11: exact forward holdout results.
slide = prs.slides.add_slide(blank); set_background(slide); add_title(slide, "Forward-model holdout results", "Predictions at measured settings; error is relative to the measured property", 11)
add_picture(slide, chart_path("holdout_actual_predicted.png"), 0.75, 1.55, 7.15, 4.45)
rows = []
for condition, label in ((holdout["LaserPower_W"].eq(270) & holdout["HatchDistance_mm"].eq(0.08), "6 repeats"), (holdout["LaserPower_W"].eq(280) & holdout["HatchDistance_mm"].eq(0.085), "Sample 11")):
    for _, row in holdout.loc[condition].iterrows():
        prop = {"Hardness_HV": "Hardness", "UTS_MPa": "UTS", "RelativeDensity_pct": "Density"}[row["Property"]]
        rows.append([label, prop, f"{row['Actual_Mean']:.2f}", f"{row['Predicted']:.2f}", f"{row['ErrorPercent']:+.2f}%"])
add_table(slide, 8.1, 1.72, 4.45, 4.85, ["Setting", "Property", "Measured", "Predicted", "Error"], rows, widths=[1.0, 0.95, 0.9, 0.9, 0.7], font_size=9.5)
add_text(slide, "Sample 11 UTS is the main miss (-5.19%); additional unique-condition experiments are needed.", 0.95, 6.24, 11.7, 0.34, size=13, color=BURGUNDY, bold=True)

# Slide 12: cross-validation diagnostics with correct, property-specific units.
slide = prs.slides.add_slide(blank); set_background(slide); add_title(slide, "Training-only LOOCV diagnostics", "Ridge teacher predictions across 13 unique training settings", 12)
metric_rows = []
for target, label, unit in (("UTS_MPa", "UTS", "MPa"), ("RelativeDensity_pct", "Density", "percentage points"), ("Hardness_HV", "Hardness", "HV")):
    row = metrics.loc[target]
    metric_rows.append([label, "13", f"{row['R2']:.3f}", f"{row['RMSE']:.2f} {unit}", f"{row['MAE']:.2f} {unit}", f"{row['MAPE_Percent']:.2f}%"])
add_table(slide, 0.9, 1.78, 11.55, 2.55, ["Property", "N", "R2", "RMSE", "MAE", "MAPE"], metric_rows, widths=[2.0, 0.7, 1.3, 2.7, 2.6, 2.25], font_size=14)
add_card(slide, 0.95, 4.78, 11.45, 1.18, "INTERPRETATION", "All three LOOCV R2 values are below zero: on these 13 unique settings, out-of-fold predictions did not beat the mean-target baseline.", accent=BURGUNDY, body_size=16)
add_text(slide, "These are diagnostics for the real-data Ridge teacher, not a claim that the 500 synthetic rows are independent measurements.", 1.0, 6.28, 11.2, 0.38, size=11, color=MID)

# Slide 13: inverse recovery and property-match errors.
slide = prs.slides.add_slide(blank); set_background(slide); add_title(slide, "Inverse recovery: matching properties is not recovering the same setting", "The validation-fold inverse uses models trained on the other 13 settings", 13)
add_picture(slide, chart_path("inverse_property_errors.png"), 0.75, 1.62, 7.1, 3.95)
inv_rows = []
for _, row in inverse.iterrows():
    name = "6-repeat" if int(row["ValidationCondition"]) == 1 else "Sample 11"
    actual = f"{row['Actual_LaserPower_W']:.0f}/{row['Actual_ScanSpeed_mm_s']:.0f}/{row['Actual_HatchDistance_mm']:.3f}"
    found = f"{row['Inverse_LaserPower_W']:.1f}/{row['Inverse_ScanSpeed_mm_s']:.1f}/{row['Inverse_HatchDistance_mm']:.4f}"
    inv_rows.append([name, actual, found])
add_table(slide, 8.05, 1.9, 4.55, 2.35, ["Target", "Actual P/v/h", "Recovered P/v/h"], inv_rows, widths=[1.1, 1.7, 1.75], font_size=10)
add_text(slide, "First condition: properties matched closely at a different setting. This demonstrates non-unique inverse solutions.", 8.12, 4.55, 4.4, 0.85, size=13, color=NAVY, bold=True)
add_text(slide, f"Sample 11 inverse property errors: hardness {inverse.loc[inverse['ValidationCondition'] == 2, 'PropertyErrorPercent_Hardness_HV'].iloc[0]:+.2f}%, UTS {inverse.loc[inverse['ValidationCondition'] == 2, 'PropertyErrorPercent_UTS_MPa'].iloc[0]:+.2f}%, density {inverse.loc[inverse['ValidationCondition'] == 2, 'PropertyErrorPercent_RelativeDensity_pct'].iloc[0]:+.2f}%.", 8.12, 5.52, 4.4, 0.85, size=12, color=BURGUNDY)

# Slide 14: six diverse Pareto candidates.
slide = prs.slides.add_slide(blank); set_background(slide); add_title(slide, "Joint inverse search: six Pareto candidates", "All outputs are forward-model estimates; separate candidates lead on separate properties", 14)
candidate_rows = []
for _, row in candidates.iterrows():
    candidate_rows.append([
        f"{int(row['Candidate'])}",
        f"{row['LaserPower_W']:.1f}",
        f"{row['ScanSpeed_mm_s']:.0f}",
        f"{row['HatchDistance_mm']:.4f}",
        f"{row['Hardness_HV']:.2f}",
        f"{row['UTS_MPa']:.1f}",
        f"{row['RelativeDensity_pct']:.3f}",
    ])
add_table(slide, 0.75, 1.58, 11.85, 3.7, ["#", "Power W", "Speed mm/s", "Hatch mm", "Hardness HV", "UTS MPa", "Density %"], candidate_rows, widths=[0.6, 1.35, 1.65, 1.45, 2.0, 1.75, 2.0], font_size=12)
add_card(slide, 0.85, 5.62, 3.65, 0.88, "MAX HARDNESS", "Candidate 1  |  331.47 HV", accent=BLUE, body_size=13)
add_card(slide, 4.82, 5.62, 3.65, 0.88, "MAX UTS", "Candidate 3  |  931.16 MPa", accent=GOLD, body_size=13)
add_card(slide, 8.78, 5.62, 3.65, 0.88, "MAX DENSITY", "Candidate 2  |  99.805%", accent=TEAL, body_size=13)
add_text(slide, "The six candidates are a screening shortlist, not experimentally validated settings. Confirm machine increments and build feasibility before printing.", 0.9, 6.68, 11.7, 0.28, size=10.5, color=MID)

# Slide 15: concise close and next experiments.
slide = prs.slides.add_slide(blank); set_background(slide); add_title(slide, "Conclusions and next experimental step", "A practical result: use the models to choose informative builds, then measure them", 15)
add_bullet(slide, "Hardness, UTS, and density now share one measured-condition table and one grouped validation design.", 0.95, 1.62, 11.4, 16)
add_bullet(slide, "The six-replicate holdout checks repeatability at one setting; sample 11 checks transfer to a second setting.", 0.95, 2.35, 11.4, 16)
add_bullet(slide, "Current LOOCV R2 is negative for all three targets, so more unique real settings are the highest-value next data.", 0.95, 3.08, 11.4, 16)
add_bullet(slide, "Inverse search shows trade-offs and non-uniqueness; recommended parameters must be verified by printing and testing.", 0.95, 3.81, 11.4, 16)
add_card(slide, 1.05, 4.72, 5.35, 1.18, "PROPOSED FIRST BUILDS", "Choose candidates 1, 2, and 3 to probe balanced, density-led, and UTS-led trade-offs.", accent=BLUE, body_size=14)
add_card(slide, 6.83, 4.72, 5.35, 1.18, "MEASURE AND UPDATE", "Record actual settings, UTS, density, and orientation-specific hardness; append the results and retrain.", accent=GOLD, body_size=14)
add_text(slide, "Thank you", 1.0, 6.35, 11.2, 0.42, size=23, color=NAVY, bold=True, align=PP_ALIGN.CENTER)

# Slide 16: UI testing workflow with a reserved QR-code area.
slide = prs.slides.add_slide(blank); set_background(slide); add_title(slide, "UI testing | forward and inverse workflows", "Test the input, prediction, constraint, and recommendation paths", 16)
add_card(slide, 0.82, 1.62, 5.15, 1.05, "1  |  FORWARD TEST", "Enter a known setting and compare predicted properties with the measured record.", accent=BLUE, body_size=13)
add_card(slide, 0.82, 2.9, 5.15, 1.05, "2  |  TARGET TEST", "Enter target hardness, UTS, and density; check the returned setting and property errors.", accent=GOLD, body_size=13)
add_card(slide, 0.82, 4.18, 5.15, 1.05, "3  |  FEASIBILITY TEST", "Confirm parameter bounds, layer thickness, and volumetric energy-density window.", accent=TEAL, body_size=13)
add_card(slide, 0.82, 5.46, 5.15, 0.85, "4  |  REVIEW", "Show trade-offs and label all outputs as model estimates pending experimental checks.", accent=BURGUNDY, body_size=12)
qr_box = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(7.25), Inches(1.72), Inches(4.0), Inches(4.0))
qr_box.fill.solid(); qr_box.fill.fore_color.rgb = rgb(WHITE); qr_box.line.color.rgb = rgb(NAVY); qr_box.line.width = Pt(2)
add_text(slide, "ADD UI QR CODE", 7.6, 3.08, 3.3, 0.48, size=22, color=NAVY, bold=True, align=PP_ALIGN.CENTER)
add_text(slide, "Replace with QR code linked to the live interface or demo.", 7.72, 3.67, 3.05, 0.64, size=13, color=MID, align=PP_ALIGN.CENTER)
add_text(slide, "Demo / interface link", 7.55, 5.98, 3.4, 0.26, size=12, color=BLUE, bold=True, align=PP_ALIGN.CENTER)

# Slide 17: bibliography space for the team to complete.
slide = prs.slides.add_slide(blank); set_background(slide); add_title(slide, "Bibliography", "References", 17)
for index in range(6):
    line_y = 1.72 + index * 0.78
    add_text(slide, f"{index + 1}.", 0.95, line_y, 0.45, 0.28, size=15, color=NAVY, bold=True)
    rule = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.5), Inches(line_y + 0.28), Inches(10.75), Inches(0.012))
    rule.fill.solid(); rule.fill.fore_color.rgb = rgb(GRID); rule.line.fill.background()

# Slide 18: final thank-you and questions.
slide = prs.slides.add_slide(blank); set_background(slide)
accent = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.0), Inches(0.0), Inches(13.333), Inches(0.18))
accent.fill.solid(); accent.fill.fore_color.rgb = rgb(GOLD); accent.line.fill.background()
add_text(slide, "THANK YOU", 1.0, 2.18, 11.3, 0.82, size=38, color=NAVY, bold=True, align=PP_ALIGN.CENTER)
add_text(slide, "We are open to questions", 1.0, 3.18, 11.3, 0.48, size=22, color=INK, align=PP_ALIGN.CENTER)
add_text(slide, "Prediction of Properties of Additively Manufactured Parts for Haynes 282", 1.0, 4.05, 11.3, 0.42, size=15, color=MID, align=PP_ALIGN.CENTER)
if LOGO_PATH:
    add_picture(slide, LOGO_PATH, 6.08, 5.0, 1.15, 1.15)
add_text(slide, "ME-392  |  IIT (BHU), Varanasi", 1.0, 6.6, 11.3, 0.26, size=11, color=MID, align=PP_ALIGN.CENTER)

prs.save(OUTPUT_PATH)
print(f"Created {OUTPUT_PATH}")
