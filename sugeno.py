import pandas as pd
import numpy as np
from sklearn.metrics import *
import matplotlib.pyplot as plt

#  Helper functions to perform Sugeno model

def plot_confusion(cf_matrix,title,classes):
    import seaborn as sns
    ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues',fmt="d")
    ax.set_title(title+'\n');
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('Actual labels ');
    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(classes)
    ax.yaxis.set_ticklabels(classes)
    ## Display the visualization of the Confusion Matrix.
    plt.show()

def generate_cardinality(N, p = 2):
    return [(x/ N)**p for x in np.arange(N, 0, -1)]

def sugeno_fuzzy_integral(X, measure=None, axis = 0, keepdims=True):
    if measure is None:
        measure = generate_cardinality(X.shape[axis])

    return sugeno_fuzzy_integral_generalized(X, measure, axis, np.minimum, np.amax, keepdims)


def sugeno_fuzzy_integral_generalized(X, measure, axis = 0, f1 = np.minimum, f2 = np.amax, keepdims=True):
    X_sorted = np.sort(X, axis = axis)
    return f2(f1(np.take(X_sorted, np.arange(0, X_sorted.shape[axis]), axis), measure), axis=axis, keepdims=keepdims)



def predicting(ensemble_prob):
    prediction = np.zeros((ensemble_prob.shape[0],))
    for i in range(ensemble_prob.shape[0]):
        temp = ensemble_prob[i]
        t = np.where(temp == np.max(temp))[0][0]
        prediction[i] = t
    return prediction

def metrics(labels,predictions,classes,name='sugeno'):
    print('=' * 10, '{}'.format(name), '=' * 10)
    print("Classification Report:")
    # print(classification_report(labels, predictions, target_names = classes,digits = 4))
    matrix = confusion_matrix(labels, predictions)
    plot_confusion(matrix,name,classes)

    print("Confusion matrix:")
    print(matrix)
    print("Classwise Accuracy :{}".format(matrix.diagonal()/matrix.sum(axis = 1)))
    print("Balanced Accuracy Score: ",balanced_accuracy_score(labels,predictions))
    print('=' * 40)

#Sugeno Integral
def ensemble_sugeno(names,pred,labels):

    prob1=pred[names[0]]
    prob2=pred[names[1]]
    prob3=pred[names[2]]
    prob4=pred[names[3]]
    num_classes = prob1.shape[1]
    num_samples = prob1.shape[0]

    Y = np.zeros(prob1.shape,dtype=float)
    for samples in range(num_samples):
        for classes in range(num_classes):
            X = np.array([prob1[samples][classes], prob2[samples][classes], prob3[samples][classes], prob4[samples][classes] ])
            measure = np.array([0.35, 0.35, 0.02, 0.28])
            X_agg = sugeno_fuzzy_integral_generalized(X,measure)
            Y[samples][classes] = X_agg

    sugeno_pred = predicting(Y)

    correct = np.where(sugeno_pred == labels)[0].shape[0]
    total = labels.shape[0]

    print("Accuracy = ",(correct/total)*100)
    classes = ['AD','CN']
    metrics(labels,sugeno_pred,classes)