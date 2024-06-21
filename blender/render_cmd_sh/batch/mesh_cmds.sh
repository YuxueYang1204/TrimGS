
# mesh_type: mesh_dtu
python mesh.py --mesh_type mesh_dtu --mesh_list scan97 scan122 scan105 scan83 scan114 --cuda_id 0  --debug_mode=-1
python mesh.py --mesh_type mesh_dtu --mesh_list scan40 scan24 scan65 scan106 scan55 --cuda_id 1  --debug_mode=-1
python mesh.py --mesh_type mesh_dtu --mesh_list scan110 scan63 scan118 scan69 scan37 --cuda_id 2  --debug_mode=-1
python mesh.py --mesh_type mesh_dtu --mesh_list  --cuda_id 3  --debug_mode=-1
python mesh.py --mesh_type mesh_dtu --mesh_list  --cuda_id 4  --debug_mode=-1
python mesh.py --mesh_type mesh_dtu --mesh_list  --cuda_id 5  --debug_mode=-1
python mesh.py --mesh_type mesh_dtu --mesh_list  --cuda_id 6  --debug_mode=-1
python mesh.py --mesh_type mesh_dtu --mesh_list  --cuda_id 7  --debug_mode=-1

# mesh_type: mesh_mip
python mesh.py --mesh_type mesh_mip --mesh_list room flowers bonsai bicycle garden --cuda_id 0  --debug_mode=-1
python mesh.py --mesh_type mesh_mip --mesh_list stump treehill kitchen counter --cuda_id 1  --debug_mode=-1
python mesh.py --mesh_type mesh_mip --mesh_list  --cuda_id 2  --debug_mode=-1
python mesh.py --mesh_type mesh_mip --mesh_list  --cuda_id 3  --debug_mode=-1
python mesh.py --mesh_type mesh_mip --mesh_list  --cuda_id 4  --debug_mode=-1
python mesh.py --mesh_type mesh_mip --mesh_list  --cuda_id 5  --debug_mode=-1
python mesh.py --mesh_type mesh_mip --mesh_list  --cuda_id 6  --debug_mode=-1
python mesh.py --mesh_type mesh_mip --mesh_list  --cuda_id 7  --debug_mode=-1