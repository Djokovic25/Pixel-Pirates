from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

def generate_pdf(report_data, summary, filename="report.pdf"):

    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()

    elements = []

    # Title
    elements.append(Paragraph("Microplastic Analysis Report", styles["Title"]))
    elements.append(Spacer(1, 12))

    # Summary
    elements.append(Paragraph(f"Overall Risk Score: {summary}", styles["Normal"]))
    elements.append(Spacer(1, 12))

    # Table
    table_data = [["Image", "Label", "Size (µm)", "Confidence"]]

    for row in report_data:
        table_data.append([
            row["image"],
            row["label"],
            f"{row['size']:.2f}",
            f"{row['confidence']:.2f}"
        ])

    table = Table(table_data)
    table.setStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.grey),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), 1, colors.black)
    ])

    elements.append(table)

    doc.build(elements)

    return filename