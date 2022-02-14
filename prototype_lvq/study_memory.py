import sys,os
sys.path.append(os.path.abspath(__file__ + "/../../../"))
from joblib import Parallel, delayed
from skmultiflow.prototype.robust_soft_learning_vector_quantization import RobustSoftLearningVectorQuantization as RSLVQ
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
from skmultiflow.lazy import KNN
from skmultiflow.meta.oza_bagging_adwin import OzaBaggingAdwin
from model.rrslvq import ReactiveRobustSoftLearningVectorQuantization as RRSLVQ
from skmultiflow.trees.hoeffding_adaptive_tree import HAT
from skmultiflow.meta.adaptive_random_forests import AdaptiveRandomForest
from utils.study import Study

def init_classifiers():
    n_prototypes_per_class = 4
    sigma = 4
    rslvq = RSLVQ(prototypes_per_class=4, sigma=4)
    rrslvq = RRSLVQ(prototypes_per_class=n_prototypes_per_class, sigma=sigma, confidence=0.0001, window_size=300)

    oza = OzaBaggingAdwin(base_estimator=KNN())
    adf = AdaptiveRandomForest()
    hat = HAT()

    clfs = [hat,rslvq, rrslvq, adf, oza]
    names = ["hat","rslvq", "rrslvq", "adf", "oza"]

    return clfs,names

def evaluate(stream,metrics,study_size):
    clfs,names = init_classifiers()
    evaluator = EvaluatePrequential(show_plot=False, batch_size=10, max_samples=study_size, metrics=metrics,
                                    output_file=stream.name+"_memory_other.csv")

    evaluator.evaluate(stream=stream, model=clfs, model_names=names)

s = Study()
parallel =2
study_size = 50000 #100000
metrics = ['accuracy','model_size']


streams = s.init_standard_streams()  + s.init_reoccuring_standard_streams() + s.init_real_world()

Parallel(n_jobs=parallel,max_nbytes=None)(delayed(evaluate)(stream,metrics,study_size) for stream in streams)
