from sklearn.metrics import precision_score, recall_score, f1_score, classification_report


def compute_metrics_for_label(true_labels, predicted_labels, label='All', average='weighted'):

    if label not in ['All', 'FUN', 'STR', 'MIX', 'OTH']:
        print("Error: You must specify a label from the following options: 'All', 'FUN', 'STR', 'MIX' or 'OTH'")
        return None

    # If label is 'All', compute overall metrics using averaging
    if label == 'All':
        precision = precision_score(true_labels, predicted_labels, average=average, zero_division=0)
        recall = recall_score(true_labels, predicted_labels, average=average, zero_division=0)
        f1 = f1_score(true_labels, predicted_labels, average=average, zero_division=0)
    
    # Otherwise, compute per-label metrics
    else:
        report = classification_report(true_labels, predicted_labels, output_dict=True, zero_division=0)
        precision = report[label]['precision']
        recall = report[label]['recall']
        f1 = report[label]['f1-score']

    return round(precision, 3) , round(recall, 3), round(f1, 3)