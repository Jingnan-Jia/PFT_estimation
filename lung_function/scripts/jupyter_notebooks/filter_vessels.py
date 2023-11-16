import numpy as np
from seg_metrics.seg_metrics import load_itk, save_itk
import pickle




def main():
    # TODO: for loop
    img_fpath = "/home/jjia/data/dataset/lung_function/ori_resolution/SSc_patient_6484444_skeleton_coordinates.pt"
    vs_skeleton = pickle.load(img_fpath)
    
    
    
if __name__ == '__main__':
    main()





