1. "shrec_16" dataset added in folder "datasets"

2. Use python3.7 (3.7.9)

3. Install requirements (python3.7 -m pip install requirements.txt)

4. Run command to train the model on shrec_16 dataset
	python3.7 train.py --dataroot datasets/shrec_16 --name shrec16  --ncf 64 128 256 256 --pool_res 600 450 300 180 --norm group --resblocks 1 --flip_edges 0.2 --slide_verts 0.2 --num_aug 20 --niter_decay 100 

5. Run command to test the model on shrec_16 dataset
	python3.7 test.py --dataroot datasets/shrec_16 --name shrec16 --ncf 64 128 256 256 --pool_res 600 450 300 180 --norm group --resblocks 1 --export_folder meshes 

6. Run command to view the 3D object meshes
	python3.7 util/mesh_viewer.py --files checkpoints/shrec16/meshes/T74_0.obj checkpoints/shrec16/meshes/T74_3.obj checkpoints/shrec16/meshes/T74_4.obj


Note: I changed these files mesh_classifier.py, networks.py, and mesh_conv.py