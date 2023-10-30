import random
import os
import subprocess
import sys
import json
import multiprocessing

random.seed(42)

elements = ['p_mean','pet_mean','p_seasonality','frac_snow','aridity','high_prec_freq','high_prec_dur',
               'low_prec_freq','low_prec_dur', 'elev_mean', 'slope_mean', 'area_gages2', 'frac_forest', 'lai_max',
               'lai_diff', 'gvf_max', 'gvf_diff', 'dom_land_cover_frac', 'dom_land_cover', 'root_depth_50',
               'soil_depth_pelletier', 'soil_depth_statsgo', 'soil_porosity', 'soil_conductivity',
               'max_water_content', 'sand_frac', 'silt_frac', 'clay_frac', 'geol_1st_class', 'glim_1st_class_frac',
               'geol_2nd_class', 'glim_2nd_class_frac', 'carbonate_rocks_frac', 'geol_porostiy', 'geol_permeability']

#-----making ensembles-------
# # print(len(attrnewLst))
# random.shuffle(elements)
#
# # Create 10 empty lists
# lists = [[] for _ in range(10)]
#
# # Distribute one instance of each element to a random list
# for i in range(10):
#     element = elements[i]
#     lists[i].append(element)
#
# # Fill up each list to length 10 by randomly sampling from remaining elements
# for i in range(10):
#     while len(lists[i]) < 10:
#         remaining_elements = list(set(elements) - set(lists[i]))
#         element = random.choice(remaining_elements)
#         lists[i].append(element)
#
# # Print the lists
# for i, lst in enumerate(lists):
#     # check = has_duplicates(lst)
#     print(f"List {i+1}: {lst}")

list1 = ['elev_mean', 'pet_mean', 'soil_conductivity', 'glim_1st_class_frac', 'glim_2nd_class_frac', 'aridity', 'geol_permeability', 'geol_1st_class', 'high_prec_freq', 'geol_porostiy']
list2 = ['frac_forest', 'glim_2nd_class_frac', 'sand_frac', 'dom_land_cover_frac', 'pet_mean', 'glim_1st_class_frac', 'lai_diff', 'max_water_content', 'gvf_diff', 'soil_conductivity']
list3 = ['high_prec_freq', 'dom_land_cover', 'silt_frac', 'glim_2nd_class_frac', 'soil_depth_pelletier', 'geol_permeability', 'geol_2nd_class', 'dom_land_cover_frac', 'sand_frac', 'gvf_max']
list4 = ['soil_depth_pelletier', 'geol_permeability', 'soil_depth_statsgo', 'high_prec_freq', 'dom_land_cover_frac', 'glim_1st_class_frac', 'gvf_diff', 'slope_mean', 'geol_1st_class', 'carbonate_rocks_frac']
list5 = ['glim_2nd_class_frac', 'frac_snow', 'silt_frac', 'frac_forest', 'p_mean', 'sand_frac', 'geol_2nd_class', 'slope_mean', 'area_gages2', 'low_prec_dur']
list6 = ['soil_porosity', 'p_seasonality', 'slope_mean', 'geol_porostiy', 'silt_frac', 'low_prec_freq', 'dom_land_cover_frac', 'sand_frac', 'high_prec_freq', 'root_depth_50']
list7 = ['glim_1st_class_frac', 'carbonate_rocks_frac', 'soil_conductivity', 'geol_permeability', 'geol_porostiy', 'gvf_max', 'dom_land_cover', 'root_depth_50', 'gvf_diff', 'sand_frac']
list8 = ['gvf_diff', 'low_prec_dur', 'pet_mean', 'geol_1st_class', 'dom_land_cover', 'lai_diff', 'max_water_content', 'geol_2nd_class', 'slope_mean', 'soil_conductivity']
list9 = ['silt_frac', 'soil_conductivity', 'soil_depth_statsgo', 'low_prec_dur', 'clay_frac', 'pet_mean', 'low_prec_freq', 'root_depth_50', 'carbonate_rocks_frac', 'frac_snow']
list10 = ['max_water_content', 'geol_2nd_class', 'silt_frac', 'lai_diff', 'area_gages2', 'high_prec_dur', 'glim_2nd_class_frac', 'carbonate_rocks_frac', 'elev_mean', 'p_seasonality']


alpha1 = 0.0
alpha2 = 0.25
alpha3 = 0.5
alpha4 = 0.75
alpha5 = 1.0
# use_ensembles = 'True'
# gpuid = '5'
action = 1 # 0: train, 1: test

def trigger_ensemble(lists, gpuid, ensembleno, action=0):
    if action==0:
        # subprocess.call([sys.executable, 'traindPLHBV_ensemble.py', json.dumps(lists), str(gpuid), ensembleno]) # for attributes
        subprocess.call([sys.executable, 'traindPLHBV_ensemble.py', str(lists), str(gpuid), ensembleno])
    else:
        subprocess.call([sys.executable, 'testdPLHBV-Dynamic-ensemble.py', json.dumps(lists), json.dumps(gpuid)])
        # subprocess.call([sys.executable, 'testdPLHBV-Dynamic-ensemble.py', json.dumps(lists), json.dumps(gpuid)])


# for attributes
input1 = [(list1, 5, 'ensemble1'), (list2, 6, 'ensemble2'), (list3, 7, 'ensemble3')]
input2 = [(list4, 5,'ensemble4'), (list5, 6, 'ensemble5'), (list6, 7, 'ensemble6')]
input3 = [(list7, 5, 'ensemble7'), (list8, 6, 'ensemble8'), (list9, 7, 'ensemble9'), (list10, 4, 'ensemble10')]

#for alpha
inputs_alpha = [(alpha1, 7, 'ensemble1_alpha'), (alpha2, 6, 'ensemble2_alpha'), (alpha3, 5, 'ensemble3_alpha'), (alpha4, 4, 'ensemble4_alpha'), (alpha5, 3, 'ensemble5_alpha')]
# inputs = [(alpha1, 7, 'ensemble1_alpha', 0), (alpha2, 6, 'ensemble2_alpha', 0), (alpha3, 5, 'ensemble3_alpha', 0), (alpha4, 4, 'ensemble4_alpha', 0)]
if action==0:
    # inputs = [input1, input2, input3] #for attributes
    if __name__ == '__main__':
        with multiprocessing.Pool() as pool:
            # for input in inputs:
            pool.starmap(trigger_ensemble, inputs_alpha)

else:
    inputs = input1 + input2 + input3 + inputs_alpha
    gpuids = [t[1] for t in inputs]
    # attributeList = [a[0] for a in inputs]
    alphas = [a[0] for a in inputs]
    trigger_ensemble(lists = alphas, gpuid=gpuids, ensembleno= 'all', action=action)