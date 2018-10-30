README

Recommended requirements:
300GB of (fast) storage
Tesla-generation GPUs (for training)
72 core CPU (for generating features faster)

Data:
Please create a folder called 'data'.
Dump the testing data (pdbs) into 'data/testing_data'
Dump the training data (pdbs) into 'data/training_data'

Training & Validation (regression):
Please change the 'method' parameter in line 66 of 'pdb_parser.py' to 'CUDA' (it is faster for this section of the code)
First, generate training ligand data using the file 'gen_ligand2_descriptors.py'.
Next, generate protein-ligand pairs for regression training using 'gen_protein_ligand2_descriptors_regression.py'.

Please change the 'method' parameter in line 66 of 'pdb_parser.py' to 'C'.
We use a parallized implementation of our code to reduce the time required. 
Unfortunately, it does not seem to work with CUDA.
Next, generate protein-ligand pairs for acc@10 testing using 'gen_protein_ligand2_descriptors_acc10.py'.

Finally, train the model using train_final.ipynb

Testing (acc@10):
Please change the 'method' parameter in line 66 of 'pdb_parser.py' to 'CUDA' (it is faster for this section of the code)
First, generate training ligand data using the file 'gen_test_ligand2_descriptors.py'.

Please change the 'method' parameter in line 66 of 'pdb_parser.py' to 'C'.
We use a parallized implementation of our code to reduce the time required. 
Unfortunately, it does not seem to work with CUDA.
Next, generate protein-ligand pairs for acc@10 testing using 'gen_protein_ligand2_descriptors_acc10.py'.

Generating Accuracy Scores:
Use the script, 'eval_acc10.py' to generate predictions.
Use the script, 'calc_top_3_10_acc.py' to get acc@3 & acc@10 scores.

