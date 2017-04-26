import keras.backend as K

def fbetascore(y_true, y_pred, beta=2):
    '''Compute F score, the weighted harmonic mean of precision and recall.
    
    This is useful for multi-label classification where input samples can be
    tagged with a set of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.
    
    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    
    '''
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1))) + 0.002
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1))) + 0.001
    c3 = K.sum(K.round(K.clip(y_true, 0, 1))) + 0.001
    
    # How many selected items are relevant?
    precision = c1 / c2
    
    # How many relevant items are selected?
    recall = c1 / c3
    
    # Weight precision and recall together as a single scalar.
    beta2 = beta ** 2
    f_score = (1 + beta2) * (precision * recall) / (beta2 * precision + recall)
    return f_score


def f1score(y_true, y_pred):
    return fbetascore(y_true, y_pred, beta=1)

def f2score(y_true, y_pred):
    return fbetascore(y_true, y_pred, beta=2)
