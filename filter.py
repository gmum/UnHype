import csv

input_file = "data/coco_30k.csv"
output_file = "data/coco_30k_filtered.csv"

blocked_words = ["naked", "shirtless", "nude", "sexy"]

with open(input_file, newline='', encoding="utf-8") as infile, \
     open(output_file, "w", newline='', encoding="utf-8") as outfile:

    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    for row in reader:
        prompt = row[2].lower()  # assuming prompt is column index 2
        if not any(word in prompt for word in blocked_words):
            writer.writerow(row)

print("Filtering complete.")