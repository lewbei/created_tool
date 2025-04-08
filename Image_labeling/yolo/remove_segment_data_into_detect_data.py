import os

# Root directory
root_dir = r"C:\Users\lewka\Downloads\halal logo.v5i.yolov11-detect"

# Splits to clean
splits = ['train', 'valid', 'test']

for split in splits:
    label_dir = os.path.join(root_dir, split, 'labels')

    if not os.path.exists(label_dir):
        print(f"⚠️ Folder not found (skipping): {label_dir}")
        continue

    print(f"\n🔍 Cleaning labels in folder: {label_dir}")

    for filename in os.listdir(label_dir):
        if filename.endswith('.txt'):
            filepath = os.path.join(label_dir, filename)

            with open(filepath, 'r') as file:
                lines = file.readlines()

            cleaned_lines = []
            cleaned = False
            for line in lines:
                if len(line.strip().split()) == 5:
                    cleaned_lines.append(line)
                else:
                    cleaned = True

            if cleaned:
                with open(filepath, 'w') as file:
                    file.writelines(cleaned_lines)
                print(f"✅ Cleaned segmentation annotations from: {filename}")

print("\n🎉 Finished cleaning your dataset!")
