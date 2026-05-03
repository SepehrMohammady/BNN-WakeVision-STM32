"""
Generate test_images_data.h: 5 person + 5 no-person images from WakeVision test set.
Images stored as uint8 (0-255). Firmware normalizes as (v / 127.5f) - 1.0f,
matching the Python preprocessing in convert_images.py ((x - 127.5) / 127.5).

Usage:
    python generate_test_images.py
"""
import os, random
import numpy as np
from PIL import Image

TEST_DIR   = r'C:\Projects\PhD\NAS-BNN\WakeVision\data\WakeVision_Full\test'
OUT_HEADER = r'C:\Projects\PhD\NAS-BNN\STM32\NAS_BNN_Benchmark\Core\Inc\test_images_data.h'
N_PER_CLASS = 5
SEED        = 42

def pick_files(class_dir, n, seed):
    random.seed(seed)
    exts = ('.jpg', '.jpeg', '.png', '.bmp')
    files = [f for f in os.listdir(class_dir) if f.lower().endswith(exts)]
    return [os.path.join(class_dir, f) for f in random.sample(files, min(n, len(files)))]

imgs_person    = pick_files(os.path.join(TEST_DIR, 'person_present'),    N_PER_CLASS, SEED)
imgs_no_person = pick_files(os.path.join(TEST_DIR, 'no_person_present'), N_PER_CLASS, SEED + 1)
all_imgs = imgs_person + imgs_no_person
labels   = [1] * len(imgs_person) + [0] * len(imgs_no_person)
n_total  = len(all_imgs)

os.makedirs(os.path.dirname(OUT_HEADER), exist_ok=True)

with open(OUT_HEADER, 'w') as f:
    f.write('#ifndef TEST_IMAGES_DATA_H\n')
    f.write('#define TEST_IMAGES_DATA_H\n\n')
    f.write(f'#define NUM_TEST_IMAGES {n_total}  // {len(imgs_person)} person + {len(imgs_no_person)} no_person\n')
    f.write('// label: 1=person, 0=no_person\n')
    f.write(f'const int test_image_labels[NUM_TEST_IMAGES] = {{{", ".join(map(str, labels))}}};\n\n')
    f.write('// uint8 RGB, row-major, 128x128x3. Normalize in firmware: (v/127.5f)-1.0f\n')
    f.write(f'const unsigned char test_image_data[NUM_TEST_IMAGES][128*128*3] = {{\n')

    for i, (path, label) in enumerate(zip(all_imgs, labels)):
        img = Image.open(path).convert('RGB').resize((128, 128))
        arr = np.array(img, dtype=np.uint8).flatten()
        tag = 'person' if label == 1 else 'no_person'
        f.write(f'  /* [{i}] {tag} {os.path.basename(path)} */\n  {{')
        for j, v in enumerate(arr):
            if j % 32 == 0:
                f.write('\n    ')
            f.write(f'{v},')
        f.write('\n  }')
        if i < n_total - 1:
            f.write(',')
        f.write('\n')

    f.write('};\n\n#endif // TEST_IMAGES_DATA_H\n')

# Print expected Flash cost
flash_bytes = n_total * 128 * 128 * 3
print(f'Generated {OUT_HEADER}')
print(f'  {n_total} images ({len(imgs_person)}P + {len(imgs_no_person)}NP)')
print(f'  Flash cost: {flash_bytes:,} bytes ({flash_bytes/1024:.1f} KiB)')
