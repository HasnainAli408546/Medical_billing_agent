import os
import csv

CSV_PATH = r"d:\8th Semester\Billing_Voice_Agent\backend\data\reference\cpt_reference.csv"

new_codes = [
    ["99202", "Office or other outpatient visit for the evaluation and management of a new patient (15-29 mins)", "Evaluation and Management"],
    ["99203", "Office or other outpatient visit for the evaluation and management of a new patient (30-44 mins)", "Evaluation and Management"],
    ["99204", "Office or other outpatient visit for the evaluation and management of a new patient (45-59 mins)", "Evaluation and Management"],
    ["99205", "Office or other outpatient visit for the evaluation and management of a new patient (60-74 mins)", "Evaluation and Management"],
    ["99212", "Office or other outpatient visit for the evaluation and management of an established patient (10-19 mins)", "Evaluation and Management"],
    ["99213", "Office or other outpatient visit for the evaluation and management of an established patient (20-29 mins)", "Evaluation and Management"],
    ["99214", "Office or other outpatient visit for the evaluation and management of an established patient (30-39 mins)", "Evaluation and Management"],
    ["99215", "Office or other outpatient visit for the evaluation and management of an established patient (40-54 mins)", "Evaluation and Management"]
]

def main():
    print(f"Injecting 8 US E/M codes into {CSV_PATH}...")
    
    # Read existing codes to prevent duplicates
    existing_codes = set()
    try:
        with open(CSV_PATH, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None) # skip header
            for row in reader:
                if row:
                    existing_codes.add(row[0])
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    codes_added = 0
    with open(CSV_PATH, 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        for code, desc, cat in new_codes:
            if code not in existing_codes:
                writer.writerow([code, desc, cat])
                codes_added += 1
                
    print(f"Successfully added {codes_added} new codes.")

if __name__ == "__main__":
    main()
