import os
from glob import glob


if __name__ == '__main__':
    srcs = glob(os.path.join('pusht_frames', '*.png'))
    srcs = sorted(srcs, key=lambda x: int(os.path.basename(x).split('.')[0]))
    os.makedirs('pusht_frames/train', exist_ok=True)
    os.makedirs('pusht_frames/test', exist_ok=True)

    for i_src, src in enumerate(srcs):
        if i_src < len(srcs) * 4 // 5:
            dst = os.path.join('pusht_frames', 'train', os.path.basename(src))
        else:
            dst = os.path.join('pusht_frames', 'test', os.path.basename(src))
        print(src, dst)
        os.rename(src, dst)