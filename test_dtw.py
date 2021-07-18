import argparse

import os
import numpy as np

def main():

    # Example options:
    # train ./Configs/Base.yaml
    # test ./Configs/Base.yaml

    ap = argparse.ArgumentParser("Progressive Transformers test DTW score")

    # Choose between Train and Test
    ap.add_argument("test_dir_path",
                    help="test dir path")

    args = ap.parse_args()

    path = args.test_dir_path #'Models_f/Base/test_videos'
    dtws = []
    for v in sorted(os.listdir(path)):
    	score, deci = v.split('.')[0].split('_')[-2:]
    	d = score + '.' + deci
    	dtws.append(float(d))

    print("avg dtw ", np.mean(dtws))

if __name__ == "__main__":
    main()
