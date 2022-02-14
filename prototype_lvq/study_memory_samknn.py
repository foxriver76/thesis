import sys,os
sys.path.append(os.path.abspath(__file__ + "/../../../"))
from joblib import Parallel, delayed
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
from skmultiflow.lazy.sam_knn import SAMKNN
from utils.study import Study

def init_classifiers():
    samknn = SAMKNN()
    clfs = [samknn]
    names = ["SamKnn"]
    return clfs,names

def evaluate(stream,metrics,study_size):
    clfs,names = init_classifiers()
    evaluator = EvaluatePrequential(show_plot=False, batch_size=10, max_samples=study_size, metrics=metrics,
                                    output_file=stream.name+"_memory_other.csv")

    evaluator.evaluate(stream=stream, model=clfs, model_names=names)

s = Study()
parallel =2
study_size = 100000 
metrics = ['accuracy','model_size']

streams = s.init_standard_streams()  + s.init_reoccuring_standard_streams() + s.init_real_world()

Parallel(n_jobs=parallel,max_nbytes=None)(delayed(evaluate)(stream,metrics,study_size) for stream in streams)