README

Requirements:
300GB of (fast) storage
Tesla-generation GPUs (for training)
72 core CPU (for generating features faster)

Non-exhaustive list of libraries used:
jupyter lab, tqdm, keras, htmd, numpy, seaborn, pandas, cudatoolkit

Data:
Please create a folder called 'data'.
Dump the testing data (pdbs) into 'data/testing_data'
Dump the training data (pdbs) into 'data/training_data'
Please run the script, simply_training_data.py in root folder

Training & Validation:
First, generate training ligand data using the file 'gen_ligand2_descriptors.py'.

Next, generate protein-ligand pairs for regression training using 'gen_protein_ligand2_descriptors_regression.py'.
Note that we did not include a progress bar for this script. 
Do read the comments in the script for a suggestion on how to manually monitor progress.

Next, generate protein-ligand pairs for acc@10 testing using 'gen_protein_ligand2_descriptors_acc10.py'.
Please read the notes in the file before running it, some strings are hardcoded and have to be uncommented.
We use parallized code + CUDA for this part of the code.
If your number of cores is too high, there might be OOM errors in CUDA.
One core takes about 70mb of VRAM. Do pop into the code and reduce the core usage count if required, or contact us for help.

Finally, train the model using train_final.ipynb
We selected the 6th or so model to use as our final model.

Testing:
First, marshal the test pdbs using the file, 'marshal_test_pdbs.py'.

Next, generate training ligand data using the file 'gen_test_ligand2_descriptors.py'.
We suggest piping stderr to some other destination. 
The marshalled pdb files lack some information. hmtd will print warnings that block our progress bar.

Next, generate protein-ligand pairs for acc@10 testing using 'gen_protein_ligand2_descriptors_acc10.py'.
We suggest piping stderr to some other destination. 
Please read the notes in the file before running it, some strings are hardcoded and have to be swapped.
We use parallized code + CUDA for this part of the code.
If your number of cores is too high, there might be OOM errors in CUDA.
One core takes about 70mb of VRAM. Do pop into the code and reduce the core usage count if required, or contact us for help.


Generating Accuracy Scores:
Use the script, 'eval_acc10.py' to generate predictions.
Use the script, 'calc_top_3_10_acc.py' to get acc@3 & acc@10 scores.

