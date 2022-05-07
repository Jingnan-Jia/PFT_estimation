#This file should be run using local interpreter because all data are saved at local PC now
import csv
import glob
import pandas as pd
import numpy as np
import copy
import shutil
from medutils.medutils import get_all_ct_names, load_itk, save_itk
import os
from pathlib import Path
import tqdm



def bbox2_3D(img):

    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return rmin, rmax, cmin, cmax, zmin, zmax

folder = "P:\Databases\SSc\SSc_Database\LUMCAq64"
label_file = "E:\jjia\projects\lung_function\SScBaseline_PFT_anonymized.xlsx"
patients_fdir = glob.glob(folder+"\SSc_patient_*")
patients_dir = sorted([i.split('\\')[-1] for i in patients_fdir])
# patients_dir = np.array(patients_dir)[:, None]

patients_excel = pd.read_excel(label_file, engine='openpyxl')

sub_id = pd.DataFrame(patients_excel, columns=['subjectID']).values
sub_id_sort = sorted([i[0] for i in sub_id])
# confirm that the data in excel and in folder are one-to-one
assert patients_dir==sub_id_sort


patients_excel = patients_excel.drop(patients_excel[np.isnan(patients_excel.DLCO_SB)].index)
patients_excel = patients_excel.drop(patients_excel[patients_excel.DLCO_SB==0].index)
patients_excel = patients_excel.drop(patients_excel[np.isnan(patients_excel['FEV 1'])].index)
patients_excel = patients_excel.drop(patients_excel[patients_excel['FEV 1']==0].index)
patients_excel = patients_excel.drop(patients_excel[np.isnan(patients_excel.DateDF_abs)].index)
patients_excel = patients_excel.drop(patients_excel[patients_excel.DateDF_abs > 10].index)

mha_files = glob.glob(folder + "\SSc_patient_*\*\TLC_HRes\images\mhd\Original_CT.mha")
length = len(mha_files)
lung_masks = glob.glob(folder + "\SSc_patient_*\*\TLC_HRes\masks\lungs\Both_Lung_Mask_Atlas2.mha")
lung_masks_set = set([a.split('SSc_patient_')[-1][:7] for a in lung_masks])



scan_date_dt = {row['subjectID']: row['scandate'] for index, row in patients_excel.iterrows()}

scan_fpath_dt = {}
mha_files_cp = copy.deepcopy(mha_files)
for id, date in scan_date_dt.items():
    flag=False
    for file in mha_files_cp:
        if id in file and str(date) in file:
            scan_fpath_dt[id] = file
            flag=True
            break
    if flag is False:
        print(flag, id, date)


print('------')
print(len(scan_fpath_dt))

for id, fpath in tqdm.tqdm(scan_fpath_dt.items()):
    # shutil.copy(fpath, "E:\jjia\data\lung_function" + '\\' + id + ".mha")

    lung_fpath = fpath.replace('images', 'masks').replace('mhd', 'lungs').replace('Original_CT', 'Both_Lung_Mask_Atlas2')
    lung_fpath_new = "E:\jjia\data\lung_function" + '\\' + id + "_LungMask.mha"
    # shutil.copy(lung_fpath, lung_fpath_new)
    lung_mask_np, ori, sp = load_itk(lung_fpath_new, True)

    zmin, zmax, ymin, ymax, xmin, xmax = bbox2_3D(lung_mask_np)
    z_size = (zmax-zmin)*sp[0]
    y_size = (ymax-ymin)*sp[1]
    x_size = (xmax-xmin)*sp[2]
    data_stastics_fpath = 'BoxOfLung5.csv'
    if not Path(data_stastics_fpath).exists():
        with open(data_stastics_fpath, 'w', newline='') as f:
            writer = csv.writer(f)
            row = ['z_left0', 'z_right0', 'y_left0', 'y_right0', 'x_left0', 'x_right0',
                   'z_sizeLungMM', 'y_sizeLungMM', 'x_sizeLungMM',
                   'z_sizeLung', 'y_sizeLung', 'x_sizeLung',
                   'z_lung/z', 'y_lung/y', 'x_lung/x',
                   'z_size', 'y_size', 'x_size',
                   'z_sp', 'y_sp', 'x_sp']
            writer.writerow(row)

    with open(data_stastics_fpath, 'a', newline='') as f:
        writer = csv.writer(f)
        row = [zmin, lung_mask_np.shape[0]-zmax, ymin, lung_mask_np.shape[1]-ymax, xmin, lung_mask_np.shape[2]-xmax,
               z_size, y_size, x_size,
               zmax-zmin, ymax-ymin, xmax-xmin,
               (zmax - zmin)/lung_mask_np.shape[0],
               (ymax - ymin)/lung_mask_np.shape[1],
               (xmax - xmin)/lung_mask_np.shape[2],
               *lung_mask_np.shape, *sp]
        writer.writerow(row)







