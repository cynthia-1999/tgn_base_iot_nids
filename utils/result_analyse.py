import numpy as np
import umap
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix, classification_report

def plot_confusion_matrix(cm,
                          target_names,
                          save_path,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(12, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    # plt.show()
    plt.savefig(save_path)

def make_data_visual(num_data_path, save_path, label_path):
    num_data = np.load(num_data_path, allow_pickle=True)
    labels = np.load(label_path, allow_pickle=True)
    umap_model = umap.UMAP(n_components=2)
    umap_model = umap.UMAP(n_components=2, n_jobs=-1)
    data_umap = umap_model.fit_transform(num_data)
    df_umap = pd.DataFrame(data_umap, columns=['comp1', 'comp2'])
    df_umap['label'] = labels
    plt.figure(figsize=(8,8))
    sns.scatterplot(x='comp1', y='comp2', data=df_umap, hue='label')
    plt.legend(loc='upper left', frameon=False)
    plt.savefig(save_path)

def draw_confusion_matrix(pred_path, actual_path, label_path, save_path):
    test_preds = np.load(pred_path, allow_pickle=True)
    test_actuals = np.load(actual_path, allow_pickle=True)
    labels = np.load(label_path, allow_pickle=True)
    plot_confusion_matrix(cm = confusion_matrix(test_actuals, test_preds),
                                  save_path=save_path,
                                  target_names = labels,
                                  title        = "Confusion Matrix")
    
def get_report(pred_path, actual_path, label_path, save_path):
    test_preds = np.load(pred_path, allow_pickle=True)
    test_actuals = np.load(actual_path, allow_pickle=True)
    report = classification_report(test_actuals, test_preds, digits=4, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv(save_path)