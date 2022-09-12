from skmultiflow.data import RandomRBFGenerator
from model.arslvq import RSLVQ
from skmultiflow.evaluation import EvaluatePrequential

stream = RandomRBFGenerator(n_centroids=5, n_classes=2, sample_random_state=42)

clf = RSLVQ(prototypes_per_class=5, sigma=1)

evaluator = EvaluatePrequential(show_plot=True)

evaluator.evaluate(stream, clf)