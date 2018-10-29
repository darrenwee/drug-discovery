
def write_predictions_to_file(results_dic, out_filename='predictions.txt'):
    with open(out_filename, 'w') as f:
        
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
        for protein_index, fit in enumerate(results_dict[ligand_index]):
            if not predictions[protein_index]:
                predictions[protein_index] = []
            ligand_key = (ligand_index, fit)
            predictions[protein_index].append(ligand_key)

    for protein in predictions:
        predictions[protein].sort()
	return predictions
