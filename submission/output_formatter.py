import numpy as np
import csv

HEADER = ['pro_id', 'lig1_id', 'lig2_id', 'lig3_id', 'lig4_id', 'lig5_id', 'lig6_id', 'lig7_id', 'lig8_id', 'lig9_id', 'lig10_id']

def write_predictions_to_file(results_dic, n=10, header=HEADER, out_filename='predictions.txt'):
    predictions = massage_predictions_for_output(results_dic)
    with open(out_filename, 'w') as f:
        prediction_writer = csv.writer(f, delimiter=' ', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        prediction_writer.writerow(header)
        for protein in predictions:
            prediction_writer.writerow(protein, results_dic[protein][0:n])
    return

def massage_predictions_for_output(results_dict):
    """
    key: ligand_id
    value: 1D numpy array, where index is protein_id and value[protein_id] is the goodness of fit

    returns dict
    """
    predictions = {}
    # output dict: key is protein_id, value is sorted list of (ligand_id, goodness_of_fit) tuples
    
    for ligand_index in results_dict:
        ligand_index = int(ligand_index)
        for protein_index, fit in enumerate(results_dict[ligand_index]):
            protein_index, fit = int(protein_index), float(fit)
            if protein_index not in predictions:
                predictions[protein_index] = []
            ligand_key = (ligand_index, fit)
            predictions[protein_index].append(ligand_key)

    for protein in predictions:
        predictions[protein].sort(key = lambda x : x[1], reverse=True)
    return predictions

test_dic = {
    1: np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
    2: np.array([0.11, 0.21, 0.31, 0.45, 0.55, 0.65]),
    3: np.array([0.12, 0.22, 0.32, 0.44, 0.54, 0.64]),
    4: np.array([0.13, 0.23, 0.33, 0.43, 0.53, 0.63]),
    5: np.array([0.14, 0.24, 0.34, 0.42, 0.52, 0.62]),
    6: np.array([0.15, 0.25, 0.35, 0.41, 0.51, 0.61]),
}
