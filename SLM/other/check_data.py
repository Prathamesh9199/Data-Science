import numpy as np

def about_data(file_path):
    data = np.memmap(file_path, dtype=np.uint16, mode='r')
    print(f"File size      : {len(data):,} tokens")

    CHUNK = 10_000_000
    last_real = 0
    for i in range(0, len(data), CHUNK):
        chunk = data[i:i+CHUNK]
        nonzero = np.nonzero(chunk)[0]
        if len(nonzero) > 0:
            last_real = i + nonzero[-1] + 1
        else:
            break  # hit the zero region

    print(f"Real tokens    : {last_real:,}")
    print(f"Zero padding   : {len(data) - last_real:,}")
    print(f"Real data size : {last_real * 2 / 1e6:.1f} MB")


bins = [
    '/home/prathamesh/Data-Science/SLM/data/cosmopedia/train.bin',
    '/home/prathamesh/Data-Science/SLM/data/cosmopedia/val.bin',
    '/home/prathamesh/Data-Science/SLM/data/dclm-edu/eq_3/train.bin',
    '/home/prathamesh/Data-Science/SLM/data/dclm-edu/eq_3/val.bin',
    '/home/prathamesh/Data-Science/SLM/data/dclm-edu/gt_eq_4/train.bin',
    '/home/prathamesh/Data-Science/SLM/data/dclm-edu/gt_eq_4/val.bin',
    '/home/prathamesh/Data-Science/SLM/data/dolmino/train.bin',
    '/home/prathamesh/Data-Science/SLM/data/dolmino/val.bin',
    '/home/prathamesh/Data-Science/SLM/data/openwebmath/train.bin',
    '/home/prathamesh/Data-Science/SLM/data/openwebmath/val.bin',
    '/home/prathamesh/Data-Science/SLM/data/pes2o/train.bin',
    '/home/prathamesh/Data-Science/SLM/data/pes2o/val.bin',
    '/home/prathamesh/Data-Science/SLM/data/stackexchange/train.bin',
    '/home/prathamesh/Data-Science/SLM/data/stackexchange/val.bin',
    '/home/prathamesh/Data-Science/SLM/data/master_train.bin',
    '/home/prathamesh/Data-Science/SLM/data/master_val.bin',
]

for bin in bins:
    print(f"{bin.split('/')[-3]} | {bin.split('/')[-2]} | {bin.split('/')[-1]}")
    about_data(bin)
    print("=" * 50)