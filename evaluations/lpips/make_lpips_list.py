import os
dirr = '/home/user/scdm/results/ade20k/generated_samples'
name = 'lpips_list.txt'
filelist = os.listdir(dirr)
filelist.sort()
seed_list 
with open(name, 'w') as f:
    f.writelines(dirr + line + '\n' for line in filelist)