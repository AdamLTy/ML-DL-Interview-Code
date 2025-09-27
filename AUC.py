# AUC basic code
def auc_basic(y_true, y_scores):
    # Sort by prediction probability in descending order
    sorted_indices = sorted(
        range(len(y_scores)), key=lambda i: y_scores[i], reverse=True
    )

    # Calculate cumulative TP and FP
    tp_cumsum = 0
    fp_cumsum = 0
    tpr_list = [0]  # TPR list
    fpr_list = [0]  # FPR list

    pos_count = sum(y_true)  # Total number of positive samples
    neg_count = len(y_true) - pos_count  # Total number of negative samples

    if pos_count == len(y_true) or neg_count == len(y_true):
        return 0.5

    for i in sorted_indices:
        if y_true[i] == 1:
            tp_cumsum += 1
        else:
            fp_cumsum += 1

        tpr = tp_cumsum / pos_count
        fpr = fp_cumsum / neg_count
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    # Calculate AUC using trapezoidal rule
    auc = 0
    for i in range(1, len(fpr_list)):
        auc += (fpr_list[i] - fpr_list[i - 1]) * (tpr_list[i] + tpr_list[i - 1]) / 2

    return auc


# AUC adavanced code
