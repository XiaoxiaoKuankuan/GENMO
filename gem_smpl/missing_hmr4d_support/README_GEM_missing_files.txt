GEM missing hmr4d_support files

This folder contains supplemental EMDB / RICH / 3DPW support files for GEM.

Copy or merge the `inputs/` directory in this folder into the root of the
GEM repository. After copying, the files should be located at:

inputs/RICH/hmr4d_support/rich_test_vimo_preproc.pt
inputs/EMDB/hmr4d_support/emdb_vimo.pt
inputs/EMDB/hmr4d_support/emdb_slam_traj.pt
inputs/3DPW/hmr4d_support/test_3dpw_vimo_labels.pt
inputs/3DPW/hmr4d_support/3dpw_test_slam_traj.pt

These paths are the ones used by:

gem/datasets/rich/rich_motion_test.py
gem/datasets/emdb/emdb_motion_test.py
gem/datasets/threedpw/threedpw_motion_test.py
gem/datasets/threedpw/threedpw_occ_motion_test.py
