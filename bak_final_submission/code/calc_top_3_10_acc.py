import numpy as np
import sys

predictions_filename = 'val_predictions_hydro_only.txt'

predictions_arr = np.loadtxt(predictions_filename, dtype=np.int, delimiter='\t', skiprows=1)

num_predictions = predictions_arr.shape[0]

# count correct predictions for acc@3 & acc@10
top10_num_correct_pred = 0
top3_num_correct_pred = 0

for i in range(num_predictions):
        pro_id = predictions_arr[i,0]
        lig_list = list(predictions_arr[i,1:])

        truth_lig_id = pro_id
        print(truth_lig_id)
        if truth_lig_id in lig_list:
                top10_num_correct_pred += 1
        if truth_lig_id in lig_list[:3]:
                top3_num_correct_pred += 1

        acc10 = top10_num_correct_pred / num_predictions
        acc3 = top3_num_correct_pred / num_predictions

print('acc10 accuracy:{:.3f}'.format(acc10))
print('acc3 accuracy:{:.3f}'.format(acc3))


