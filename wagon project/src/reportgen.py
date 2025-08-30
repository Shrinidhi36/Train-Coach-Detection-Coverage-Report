# src/reportgen.py
import os
from reportlab.platypus import SimpleDocTemplate, Image, Paragraph, Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from html import escape

def choose_representative_images_for_coach(coach_folder, max_per_coach=3):
    imgs = sorted([f for f in os.listdir(coach_folder) if f.lower().endswith(".jpg")])
    imgs = [os.path.join(coach_folder, f) for f in imgs]
    if not imgs:
        return []
    n = len(imgs)
    indices = []
    if max_per_coach >= 1:
        indices.append(0)
    if max_per_coach >= 2 and n > 2:
        indices.append(n//2)
    if max_per_coach >= 3 and n > 1:
        indices.append(n-1)
    indices = sorted(set([min(n-1, max(0, i)) for i in indices]))
    return [imgs[i] for i in indices]

def build_pdf_report(train_number, coach_meta, output_root, out_filename=None):
    out_folder = os.path.join(output_root, train_number)
    os.makedirs(out_folder, exist_ok=True)
    if out_filename is None:
        out_filename = os.path.join(out_folder, f"{train_number}_report.pdf")
    doc = SimpleDocTemplate(out_filename, pagesize=A4)
    story = []
    styles = getSampleStyleSheet()
    story.append(Paragraph(f"Train {train_number} Coverage Report", styles['Title']))
    story.append(Spacer(1,16))
    for meta in coach_meta:
        coach_folder = meta["folder"]
        chosen = choose_representative_images_for_coach(coach_folder, max_per_coach=3)
        if not chosen:
            continue
        story.append(Paragraph(f"Coach {meta['coach_index']}", styles['Heading2']))
        for img in chosen:
            try:
                story.append(Image(img, width=450, height=200))
            except Exception as e:
                print("Could not include image in PDF:", img, e)
            story.append(Spacer(1,8))
        story.append(Spacer(1,12))
    doc.build(story)
    return out_filename

def build_html_report(train_number, coach_meta, output_root, out_filename=None):
    out_folder = os.path.join(output_root, train_number)
    os.makedirs(out_folder, exist_ok=True)
    if out_filename is None:
        out_filename = os.path.join(out_folder, f"{train_number}_report.html")
    lines = []
    lines.append("<html><head><meta charset='utf-8'><title>Coverage Report</title></head><body>")
    lines.append(f"<h1>Train {escape(train_number)} - Coverage Report</h1>")
    for meta in coach_meta:
        lines.append(f"<h2>Coach {meta['coach_index']}</h2>")
        chosen = choose_representative_images_for_coach(meta['folder'], max_per_coach=3)
        for img in chosen:
            rel = os.path.relpath(img, out_folder)
            lines.append(f"<div style='margin:8px'><img src='{rel}' style='max-width:600px;display:block'/></div>")
    lines.append("</body></html>")
    with open(out_filename, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_filename