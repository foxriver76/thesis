import sys,os
sys.path.append(os.path.abspath(__file__ + "../../../"))
from joblib import Parallel, delayed
from skmultiflow.prototype.robust_soft_learning_vector_quantization import RobustSoftLearningVectorQuantization as RSLVQ
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
from skmultiflow.lazy import KNN
from skmultiflow.meta.oza_bagging_adwin import OzaBaggingAdwin
from skmultiflow.trees.hoeffding_adaptive_tree import HAT
from skmultiflow.meta.adaptive_random_forests import AdaptiveRandomForest
from prototype_lvq.utils.study import Study
from skmultiflow.data import FileStream
from detector.naive_bayes_baseline import cdnb as NaiveBayes

# disable the stream generator warnings
import warnings
warnings.filterwarnings('ignore')

def init_classifiers():
    kswin = NaiveBayes(alpha=0.001, drift_detector="KSWIN")
    adwin = NaiveBayes(alpha=0.001, drift_detector="ADWIN")
    mebwind = NaiveBayes(alpha=0.001, drift_detector="MEBWIND", epsilon=0.1, w_size=100)
    k_mebwind = NaiveBayes(alpha=0.001, drift_detector="KMEBWIND", epsilon=0.1, w_size=100)

    clfs = [
            kswin, adwin, mebwind, k_mebwind
            ]
    # bug in skmultiflow on measuring memory, thus we add prefix
    names = [
            'KSWIN', 
             'ADWIN', 
             'MEBWIND', 
             'K-MEBWIND'
             ]

    return clfs, names

def evaluate(stream, metrics, study_size):
    clfs, names = init_classifiers()
    print(stream)
    evaluator = EvaluatePrequential(show_plot=False, batch_size=1, max_samples=study_size, metrics=metrics,
                                    output_file='_' + stream.name + "_memory_other.csv")

    evaluator.evaluate(stream=stream, model=clfs, model_names=names)

s = Study()
parallel = 2
study_size = 50000 #100000
metrics = ['accuracy', 'model_size']

#stream  = FileStream('high_dim_stream.csv')
#stream.name = 'SEA_high'
stream  = FileStream('very_high_dim_stream.csv')
stream.name = 'SEA_very_high'

evaluate(stream, metrics, study_size)