

#---------------------------------
# New invocation of recon-all 2023年 04月 11日 星期二 11:11:31 CST 
#-------------------------------------
#@# EM Registration 2023年 04月 11日 星期二 11:11:35 CST

 mri_em_register -rusage /media/nhp/data2/KIZ_data/NMT/free_data/subjects/2021_NMT.T1/touch/rusage.mri_em_register.dat -uns 3 -mask brainmask.mgz nu.mgz /media/nhp/data2/template/RB_all_2016-05-10.vc700.gca transforms/talairach.lta 

#--------------------------------------
#@# CA Normalize 2023年 04月 11日 星期二 11:14:44 CST

 mri_ca_normalize -c ctrl_pts.mgz -mask brainmask.mgz nu.mgz /media/nhp/data2/template/RB_all_2016-05-10.vc700.gca transforms/talairach.lta norm.mgz 

#--------------------------------------
#@# CA Reg 2023年 04月 11日 星期二 11:15:26 CST

 mri_ca_register -rusage /media/nhp/data2/KIZ_data/NMT/free_data/subjects/2021_NMT.T1/touch/rusage.mri_ca_register.dat -nobigventricles -T transforms/talairach.lta -align-after -mask brainmask.mgz norm.mgz /media/nhp/data2/template/RB_all_2016-05-10.vc700.gca transforms/talairach.m3z 

#--------------------------------------
#@# CA Reg Inv 2023年 04月 11日 星期二 11:54:27 CST

 mri_ca_register -invert-and-save transforms/talairach.m3z -rusage /media/nhp/data2/KIZ_data/NMT/free_data/subjects/2021_NMT.T1/touch/rusage.mri_ca_register.inv.dat 



#---------------------------------
# New invocation of recon-all 2023年 04月 11日 星期二 12:01:13 CST 
#--------------------------------------------
#@# Intensity Normalization2 2023年 04月 11日 星期二 12:01:13 CST

 mri_normalize -mprage -aseg aseg.presurf.mgz -mask brainmask.mgz norm.mgz brain.mgz 



#---------------------------------
# New invocation of recon-all 2023年 04月 11日 星期二 12:02:30 CST 
#--------------------------------------------
#@# Mask BFS 2023年 04月 11日 星期二 12:02:30 CST

 mri_mask -T 5 brain.mgz brainmask.mgz brain.finalsurfs.mgz 

#--------------------------------------------
#@# WM Segmentation 2023年 04月 11日 星期二 12:02:32 CST

 mri_segment -mprage brain.mgz wm.seg.mgz 


 mri_edit_wm_with_aseg -keep-in wm.seg.mgz brain.mgz aseg.presurf.mgz wm.asegedit.mgz 


 mri_pretess wm.asegedit.mgz wm norm.mgz wm.mgz 



#---------------------------------
# New invocation of recon-all 2023年 04月 11日 星期二 12:14:00 CST 
#--------------------------------------------
#@# Smooth2 lh 2023年 04月 11日 星期二 12:14:00 CST

 mris_smooth -n 3 -nw -seed 1234 ../surf/lh.white.preaparc ../surf/lh.smoothwm 

#--------------------------------------------
#@# Smooth2 rh 2023年 04月 11日 星期二 12:14:01 CST

 mris_smooth -n 3 -nw -seed 1234 ../surf/rh.white.preaparc ../surf/rh.smoothwm 

#--------------------------------------------
#@# Inflation2 lh 2023年 04月 11日 星期二 12:14:02 CST

 mris_inflate -rusage /media/nhp/data2/KIZ_data/NMT/free_data/subjects/2021_NMT.T1/touch/rusage.mris_inflate.lh.dat ../surf/lh.smoothwm ../surf/lh.inflated 

#--------------------------------------------
#@# Inflation2 rh 2023年 04月 11日 星期二 12:14:09 CST

 mris_inflate -rusage /media/nhp/data2/KIZ_data/NMT/free_data/subjects/2021_NMT.T1/touch/rusage.mris_inflate.rh.dat ../surf/rh.smoothwm ../surf/rh.inflated 

#--------------------------------------------
#@# Sphere lh 2023年 04月 11日 星期二 12:14:15 CST

 mris_sphere -rusage /media/nhp/data2/KIZ_data/NMT/free_data/subjects/2021_NMT.T1/touch/rusage.mris_sphere.lh.dat -seed 1234 ../surf/lh.inflated ../surf/lh.sphere 

#--------------------------------------------
#@# Sphere rh 2023年 04月 11日 星期二 12:22:09 CST

 mris_sphere -rusage /media/nhp/data2/KIZ_data/NMT/free_data/subjects/2021_NMT.T1/touch/rusage.mris_sphere.rh.dat -seed 1234 ../surf/rh.inflated ../surf/rh.sphere 



#---------------------------------
# New invocation of recon-all 2023年 04月 11日 星期二 12:46:09 CST 
#--------------------------------------------
#@# Jacobian white lh 2023年 04月 11日 星期二 12:46:09 CST

 mris_jacobian ../surf/lh.white.preaparc ../surf/lh.sphere.reg ../surf/lh.jacobian_white 

#--------------------------------------------
#@# Jacobian white rh 2023年 04月 11日 星期二 12:46:10 CST

 mris_jacobian ../surf/rh.white.preaparc ../surf/rh.sphere.reg ../surf/rh.jacobian_white 

