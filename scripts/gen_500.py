#!/usr/bin/env python3
"""Insert 5,000 random invoices into the existing `invoices` table – fast."""

import random, psycopg2
from faker import Faker

DB_CONFIG = {
    "host": "localhost", "port": 5432,
    "user": "ocr", "password": "***REMOVED***", "dbname": "invoices"
}

fake = Faker()
Faker.seed(42)

VENDORS = [
    "Alibaba Cloud Ltd.", "Global Supplies Inc.", "TechSource HK",
    "ABC Technologies Ltd.", "Pacific Logistics Co.",
    "Digital Innovations Sdn Bhd", "EuroTrade GmbH", "Nordic Paper Mills",
]
CURRENCIES = ["USD", "EUR", "HKD", "GBP", "SGD"]

def random_date():
    return fake.date_between(start_date="-2y", end_date="today").isoformat()

def main():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    print("Inserting 5,000 invoices...")
    batch = []
    for i in range(5000):
        inv_num = f"INV-{random.randint(100000, 999999)}"
        date = random_date()
        vendor = random.choice(VENDORS)
        amount = round(random.uniform(50.0, 50000.0), 2)
        currency = random.choice(CURRENCIES)
        raw = (
            f"INVOICE\n"
            f"Invoice Number: {inv_num}\n"
            f"Date: {date}\n"
            f"Vendor: {vendor}\n"
            f"Item 1: Consulting Services\n"
            f"Item 2: Software License\n"
            f"Total Amount: {currency} {amount:,.2f}\n"
            f"Payment Due: Upon Receipt\n"
            f"Thank you for your business!\n"
        )
        batch.append((inv_num, date, vendor, str(amount), currency, raw, f"gen_{i}"))

        if len(batch) >= 500:
            cur.executemany(
                """INSERT INTO invoices
                   (invoice_number, date, vendor_name, total_amount, currency, raw_text, source_file)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                batch
            )
            conn.commit()
            print(f"  {i+1} done")
            batch.clear()

    if batch:
        cur.executemany(
            """INSERT INTO invoices
               (invoice_number, date, vendor_name, total_amount, currency, raw_text, source_file)
               VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            batch
        )
        conn.commit()

    cur.close()
    conn.close()
    print("5,000 invoices inserted successfully.")

if __name__ == "__main__":
    main()