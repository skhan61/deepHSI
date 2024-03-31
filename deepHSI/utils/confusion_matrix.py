import seaborn as sns
import itertools

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def compute_confusion_matrix(preds_np, targets_np):
    # Print out the shapes of the predictions and targets for debugging
    print(f"Shape of preds_np: {preds_np.shape}")
    print(f"Shape of targets_np: {targets_np.shape}")

    if preds_np.ndim > 1:
        # Assuming preds_np contains raw scores or probabilities,
        # convert them to class indices
        preds_labels = np.argmax(preds_np, axis=1)
        print(f"Using np.argmax, preds_labels (first 10): {preds_labels[:10]}")
    else:
        # Assuming preds_np contains direct class labels
        preds_labels = preds_np
        print(f"Using direct class labels, preds_labels (first 10): {
              preds_labels[:10]}")

    # Compute confusion matrix
    cm = confusion_matrix(targets_np, preds_labels)
    return cm


# def plot_confusion_matrix(cm):
#     # Assuming class labels are 0, 1, 2, ..., n_classes-1
#     class_names = np.arange(cm.shape[0])
#     figure = plt.figure(figsize=(8, 8))
#     plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
#     plt.title("Test Confusion Matrix")
#     plt.colorbar()
#     tick_marks = np.arange(len(class_names))
#     plt.xticks(tick_marks, class_names, rotation=45)
#     plt.yticks(tick_marks, class_names)

#     # Normalize the confusion matrix.
#     cm_normalized = np.around(
#         cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

#     # Use white text if squares are dark; otherwise black.
#     threshold = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         color = "white" if cm[i, j] > threshold else "black"
#         plt.text(j, i, cm_normalized[i, j],
#                  horizontalalignment="center", color=color)

#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.show()


def plot_confusion_matrix(cm):
    # Assuming class labels are 0, 1, 2, ..., n_classes-1
    class_names = np.arange(cm.shape[0])

    # Normalize the confusion matrix.
    cm_normalized = np.around(
        cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Create a heatmap
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap=plt.cm.Blues, cbar_kws={'label': 'Scale'},
                xticklabels=class_names, yticklabels=class_names)

    plt.title("Test Confusion Matrix")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.show()
