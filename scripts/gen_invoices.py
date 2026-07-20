#!/usr/bin/env python3
"""Generate 5 sample invoice images for testing the pipeline."""
from PIL import Image, ImageDraw, ImageFont
import os

output_dir = os.path.expanduser("~/ocr")
os.makedirs(output_dir, exist_ok=True)

invoices = [
    {"num": "INV-2024-0720", "date": "2024-07-20", "vendor": "Alibaba Cloud Ltd.", "total": "USD 3,450.00"},
    {"num": "INV-2024-0721", "date": "2024-07-21", "vendor": "Global Supplies Inc.", "total": "EUR 1,200.50"},
    {"num": "INV-2024-0722", "date": "2024-07-22", "vendor": "TechSource HK", "total": "HKD 15,800.00"},
    {"num": "INV-2024-0723", "date": "2024-07-23", "vendor": "ABC Technologies Ltd.", "total": "USD 890.00"},
    {"num": "INV-2024-0724", "date": "2024-07-24", "vendor": "Alibaba Cloud Ltd.", "total": "USD 2,100.00"},
]

for inv in invoices:
    img = Image.new("RGB", (800, 500), "white")
    draw = ImageDraw.Draw(img)
    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        font_body = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except:
        font_title = ImageFont.load_default()
        font_body = font_title

    draw.text((50, 30), "INVOICE", fill="black", font=font_title)
    draw.text((50, 80), f"Invoice Number: {inv['num']}", fill="black", font=font_body)
    draw.text((50, 120), f"Date: {inv['date']}", fill="black", font=font_body)
    draw.text((50, 160), f"Vendor: {inv['vendor']}", fill="black", font=font_body)
    draw.text((50, 220), "Item 1: Consulting Services", fill="black", font=font_body)
    draw.text((50, 260), "Item 2: Software License", fill="black", font=font_body)
    draw.text((50, 320), f"Total Amount: {inv['total']}", fill="black", font=font_body)
    draw.text((50, 380), "Payment Due: Upon Receipt", fill="black", font=font_body)
    draw.text((50, 430), "Thank you for your business!", fill="black", font=font_body)

    filename = f"sample_{inv['num']}.jpg"
    filepath = os.path.join(output_dir, filename)
    img.save(filepath, "JPEG", quality=85)
    print(f"Created {filepath}")

print("\nDone! Now run the pipeline on each:")
print("cd ~/ocr && source venv/bin/activate")
for inv in invoices:
    print(f"python3 pipeline.py sample_{inv['num']}.jpg")