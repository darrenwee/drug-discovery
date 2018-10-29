
import numpy as np
import csv

HEADER = ['pro_id', 'lig1_id', 'lig2_id', 'lig3_id', 'lig4_id', 'lig5_id', 'lig6_id', 'lig7_id', 'lig8_id', 'lig9_id',
          'lig10_id', 'lig11_id', 'lig12_id', 'lig13_id', 'lig14_id', 'lig15_id']


def write_predictions_to_file(results_dic, n=10, header=HEADER, out_filename='predictions.txt'):
    assert 0 < n <= 15

    predictions = _reduce_tuple(_massage_fits_for_output(results_dic))
    with open(out_filename, 'w') as f:
        prediction_writer = csv.writer(f, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        prediction_writer.writerow(header[0:n+1])
        for protein in predictions:
            line = [protein] + list(predictions[protein][0:n])
            prediction_writer.writerow(line)
    return


def _massage_fits_for_output(results_dict):
    """
    Transpose the ligand-protein fit dictionary.
    key: ligand_id
    value: 1D numpy array, where index is protein_id and value[protein_id] is the goodness of fit
    returns dict
    """
    fits = {}
    for ligand_index in results_dict:
        ligand_index = int(ligand_index)
        for protein_index, fit in enumerate(results_dict[ligand_index]):
            protein, fit = int(protein_index) + 1, float(fit)  # protein_index starts from 1 in dataset
            if protein not in fits:
                fits[protein] = []
            ligand_key = (ligand_index, fit)
            fits[protein].append(ligand_key)

    return fits


def _reduce_tuple(fits):
    """
    Reduce (ligand, fit) tuple to list of ligand
    :param fits:
    :return:
    """
    predictions = {}
    for protein in fits:
        fits[protein].sort(key=lambda x: float(x[1]), reverse=True)
        predictions[protein] = []
        for ligand, fit in fits[protein]:
            predictions[protein].append(ligand)
    return predictions
