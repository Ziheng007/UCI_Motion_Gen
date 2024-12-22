import os

prompt_dir = '/home/haoyum3/momask-codes/editing/sequential_1st_final/prompt' 
group_dir = '/home/haoyum3/momask-codes/editing/sequential_1st_final/raw' 
output_dir = '/home/haoyum3/momask-codes/editing/sequential_1st_final/group'
classified_files = {}

for filename in os.listdir(prompt_dir):
    if filename.endswith('.json'):
        classification = filename.split('_')[0]
        name = filename.split('.')[0].strip()
        file_path = os.path.join(prompt_dir, filename)

        if classification not in classified_files:
            classified_files[classification] = []
        classified_files[classification].append(name)

for classification, names in classified_files.items():
    txt_filename = f"{classification}.txt"
    with open(os.path.join(output_dir,txt_filename), 'w') as txt_file:
        for name in names:
            txt_file.write(name + '\n')

print("finish!")