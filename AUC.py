# AUC basic code
def auc_basic(y_true, y_scores):
    # 按预测概率降序排序
    sorted_indices = sorted(
        range(len(y_scores)), key=lambda i: y_scores[i], reverse=True
    )

    # 计算累积的TP和FP
    tp_cumsum = 0
    fp_cumsum = 0
    tpr_list = [0]  # TPR列表
    fpr_list = [0]  # FPR列表

    pos_count = sum(y_true)  # 正样本总数
    neg_count = len(y_true) - pos_count  # 负样本总数

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

    # 用梯形法则计算AUC
    auc = 0
    for i in range(1, len(fpr_list)):
        auc += (fpr_list[i] - fpr_list[i - 1]) * (tpr_list[i] + tpr_list[i - 1]) / 2

    return auc


# AUC adavanced code
