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
from meb_classifier_sam import SWMEBClf
from libSAM.elm_kernel import elm_kernel_vec

# disable the stream generator warnings
import warnings
warnings.filterwarnings('ignore')

def init_classifiers():
    rslvq = RSLVQ(prototypes_per_class=4, sigma=4, gradient_descent='adadelta')
    swmeb = SWMEBClf(eps=0.1, w_size=100, kernelized=True, only_misclassified=True, kernel_fun=elm_kernel_vec)
    
    clfs = [rslvq, swmeb]
    names = ["ARSLVQ", "MEB"]

    return clfs, names

def evaluate(stream, metrics, study_size):
    clfs, names = init_classifiers()
    evaluator = EvaluatePrequential(show_plot=False, batch_size=10, max_samples=study_size, metrics=metrics,
                                    output_file='_' + stream.name + "_memory_other.csv")

    evaluator.evaluate(stream=stream, model=clfs, model_names=names)

s = Study()
parallel = 2
study_size = 50000 #100000
metrics = ['accuracy', 'model_size']

streams = s.init_standard_streams()  + s.init_reoccuring_standard_streams() #+ s.init_real_world()
Parallel(n_jobs=parallel,max_nbytes=None)(delayed(evaluate)(stream, metrics, study_size) for stream in streams)