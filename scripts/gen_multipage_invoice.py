#!/usr/bin/env python3
"""Generate a 3‑page test PDF with invoice‑like content."""

from fpdf import FPDF
from fpdf.enums import XPos, YPos

pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)

# Page 1
pdf.add_page()
pdf.set_font("Helvetica", "B", 16)
pdf.cell(0, 10, "INVOICE", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
pdf.ln(10)
pdf.set_font("Helvetica", "", 12)
pdf.cell(0, 8, "Invoice Number: INV-2024-0801", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.cell(0, 8, "Date: 2024-08-01", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.cell(0, 8, "Vendor: Alibaba Cloud Ltd.", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.cell(0, 8, "Total Amount: USD 4,500.00", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.ln(5)
pdf.cell(0, 8, "Item 1: Cloud Hosting (6 months) - $3,000.00", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.cell(0, 8, "Item 2: Technical Support - $1,500.00", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

# Page 2
pdf.add_page()
pdf.set_font("Helvetica", "B", 16)
pdf.cell(0, 10, "INVOICE", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
pdf.ln(10)
pdf.set_font("Helvetica", "", 12)
pdf.cell(0, 8, "Invoice Number: INV-2024-0802", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.cell(0, 8, "Date: 2024-08-02", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.cell(0, 8, "Vendor: Global Supplies Inc.", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.cell(0, 8, "Total Amount: EUR 2,350.75", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.ln(5)
pdf.cell(0, 8, "Item 1: Office Supplies - EUR 1,200.00", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.cell(0, 8, "Item 2: Shipping - EUR 1,150.75", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

# Page 3
pdf.add_page()
pdf.set_font("Helvetica", "B", 16)
pdf.cell(0, 10, "INVOICE", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
pdf.ln(10)
pdf.set_font("Helvetica", "", 12)
pdf.cell(0, 8, "Invoice Number: INV-2024-0803", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.cell(0, 8, "Date: 2024-08-03", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.cell(0, 8, "Vendor: TechSource HK", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.cell(0, 8, "Total Amount: HKD 18,500.00", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.ln(5)
pdf.cell(0, 8, "Item 1: Server Equipment - HKD 15,000.00", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.cell(0, 8, "Item 2: Networking Cables - HKD 3,500.00", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

pdf.output("test_multipage_invoice.pdf")
print("Created test_multipage_invoice.pdf with 3 pages.")
