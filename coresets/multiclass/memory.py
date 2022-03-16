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
from coresets.classifier.meb_classifier_sam import SWMEBClf
from coresets.classifier.libSAM.elm_kernel import elm_kernel_vec
from mccvm_fast import MCCVM 
from sklearn.metrics.pairwise import linear_kernel, laplacian_kernel, cosine_similarity, rbf_kernel
from skmultiflow.data import LEDGeneratorDrift

# disable the stream generator warnings
import warnings
warnings.filterwarnings('ignore')

def init_classifiers():
    rslvq = RSLVQ(prototypes_per_class=4, sigma=4, gradient_descent='adadelta')
    swmeb_5 = SWMEBClf(eps=0.1, w_size=5, kernelized=True, only_misclassified=False, kernel_fun=elm_kernel_vec)
#    swmeb_50 = SWMEBClf(eps=0.1, w_size=50, kernelized=True, only_misclassified=False, kernel_fun=elm_kernel_vec)
#    swmeb_100 = SWMEBClf(eps=0.1, w_size=100, kernelized=True, only_misclassified=False, kernel_fun=elm_kernel_vec)
    swmeb_300 = SWMEBClf(eps=0.1, w_size=300, kernelized=True, only_misclassified=False, kernel_fun=elm_kernel_vec)
    mccvm_5 = MCCVM(eps=0.1, w_size=5, kernel_fun=rbf_kernel)
#    mccvm_50 = MCCVM(eps=0.1, w_size=50, kernel_fun=rbf_kernel)
#    mccvm_100 = MCCVM(eps=0.1, w_size=100, kernel_fun=rbf_kernel)
    mccvm_300 = MCCVM(eps=0.1, w_size=300, kernel_fun=rbf_kernel)

    clfs = [
            rslvq, swmeb_5, swmeb_300, 
            mccvm_5, mccvm_300
            ]
    # bug in skmultiflow on measuring memory, thus we add prefix
    names = [
            "ARSLVQ", "MEB$_5$", 
             'MEB$_{300}$', 
             "MCCVM$_5$", 
             'MCCVM$_{300}$'
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

stream  = LEDGeneratorDrift(noise_percentage=0.1, has_noise=True)

# we need a multiclass stream here
evaluate(stream, metrics, study_size)