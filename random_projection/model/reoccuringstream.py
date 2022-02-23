import numpy as np
from skmultiflow.data.base_stream import Stream
from skmultiflow.utils import check_random_state
from skmultiflow.data import AGRAWALGenerator

np.seterr(over='warn')

class ReoccuringDriftStream(Stream):
    """ ReoccuringDriftStream

    A stream generator creating frequent reoccurring concept drift by joining of two streams.  
    Drift is created by a weighted based voting of switching between streams.
    For given parameter width the weighted voting tends towards the new concept drift stream
    and eareses the current concept. The switch between concept happens after a given pause.

    The sigmoid function is an elegant and practical solution to define the probability that ech
    new instance of the stream belongs to the new concept after the drift. The sigmoid function
    introduces a gradual, smooth transition whose duration is controlled with two parameters:

    - :math:`p`, the position where the change occurs
    - :math:`w`, the width of the transition

    The sigmoid function at sample `t` is for the second stream is:

    :math:`f(t) = 1/(1+e^{-4*(t-p)/w})`
    
    The weighting function for first stream is:
    - :math: `g(t) = 1 - f(t)´

    Parameters
    ----------
    stream: Stream (default= AGRAWALGenerator(random_state=112))
        First stream, second drift

    drift_stream: Stream (default= AGRAWALGenerator(random_state=112, classification_function=2))
        Second stream, first drift


    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    alpha: float (optional, default: 0.0)
        Angle of change to estimate the width of concept drift change. If set will override the width parameter.
        Valid values are in the range (0.0, 90.0].

    position: int (default: 0)
        Central position of _first_ concept drift change.

    width: int (Default: 1000)
        Width of concept drift change.

    pause: int (Default: 1000)
        Defines pause between two concept drift, after first concept drift has happend.

    Notes
    -----
    An optional way to estimate the width of the transition :math:`w` is based on the angle :math:`\\alpha`:
    :math:`w = 1/ tan(\\alpha)`. Since width corresponds to the number of samples for the transition, the width
    is round-down to the nearest smaller integer. Notice that larger values of :math:`\\alpha` result in smaller
    widths. For :math:`\\alpha>45.0`, the width is smaller than 1 so values are round-up to 1 to avoid
    division by zero errors.

    References
    --------
    Class based on ConceptDriftStream class of scikit-multiflow package:
    https://scikit-multiflow.github.io/scikit-multiflow/skmultiflow.data.html#skmultiflow.data.ConceptDriftStream

    Examples
    --------
    >>> #Imports
    >>> from skmultiflow.data.mixed_generator import MIXEDGenerator
    >>> from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
    >>> from bix.data.reoccuringdriftstream import ReoccuringDriftStream
    >>> from skmultiflow.meta.oza_bagging_adwin import OzaBaggingAdwin
    >>> from skmultiflow.lazy.knn import KNN
    >>> #Create two streams
    >>> s1 = MIXEDGenerator(classification_function = 1, random_state= 112, balance_classes = False)
    >>> s2 = MIXEDGenerator(classification_function = 0, random_state= 112, balance_classes = False)
    >>> # Create reoccuring drift object
    >>> stream = ReoccuringDriftStream(stream=s1,
    >>>                         drift_stream=s2,
    >>>                         random_state=None,
    >>>                         alpha=90.0, # angle of change grade 0 - 90
    >>>                         position=2000,
    >>>                         width=500)
    >>> stream.prepare_for_use()
    >>> 
    >>> oza = OzaBaggingAdwin(base_estimator=KNN())
    >>> 
    >>> # Evaluation
    >>> evaluator = EvaluatePrequential(max_samples=10000, metrics=['accuracy', 'kappa_t', 'kappa_m', 'kappa'])
    >>> evaluator.evaluate(stream=stream, model=oza)

    """

    def __init__(self, stream=AGRAWALGenerator(random_state=112),
                 drift_stream=AGRAWALGenerator(random_state=112,classification_function=2),pause = 1000,
                 random_state=None,
                 alpha=0.0,
                 position=0,
                 width=1):

        self.n_samples = stream.n_samples
        self.n_targets = stream.n_targets
        self.n_features = stream.n_features
        self.n_num_features = stream.n_num_features
        self.n_cat_features = stream.n_cat_features
        self.n_classes = stream.n_classes
        self.cat_features_idx = stream.cat_features_idx
        self.feature_names = stream.feature_names
        self.target_names = ['target'] if self.n_targets == 1 else ['target_' + i for i in range(self.n_targets)]
        self.target_values = stream.target_values
        self.name = stream.name + "_"+drift_stream.name+"_"+str(pause)+"_"+str(width)
        self.probability_function = "sigmoid_prob"
        self.pause = pause
        self.counter = -1
        self._original_random_state = random_state
        self.random_state = None
        self.alpha = alpha
        if self.alpha != 0.0:
            if 0 < self.alpha <= 90.0:
                w = int(1 / np.tan(self.alpha * np.pi / 180))
                self.width = w if w > 0 else 1
            else:
                raise ValueError('Invalid alpha value: {}'.format(alpha))
        else:
            self.width = width
        
        if self.width < 0:
            raise ValueError("Width must be greater than 0")
        if self.pause < 0:
            raise ValueError("Pause must be greater than 0")

        self.position = position
        self._input_stream = stream
        self._drift_stream = drift_stream
        self.n_targets = stream.n_targets
        self._prepare_for_use()

    def _prepare_for_use(self):
        self.random_state = check_random_state(self._original_random_state)
        self.sample_idx = 0
        self._input_stream.prepare_for_use()
        self._drift_stream.prepare_for_use()

    def n_remaining_samples(self):
        n_samples = self._input_stream.n_remaining_samples() + self._drift_stream.n_remaining_samples()
        if n_samples < 0:
            n_samples = -1
        return n_samples

    def has_more_samples(self):
        return self._input_stream.has_more_samples() and self._drift_stream.has_more_samples()

    def is_restartable(self):
        return self._input_stream.is_restartable()

    def next_sample(self, batch_size=1):

        """ Returns the next `batch_size` samples.

        Parameters
        ----------
        batch_size: int
            The number of samples to return.

        Returns
        -------
        tuple or tuple list
            Return a tuple with the features matrix
            for the batch_size samples that were requested.

        """
        self.current_sample_x = np.zeros((batch_size, self.n_features))
        self.current_sample_y = np.zeros((batch_size, self.n_targets))

        for j in range(batch_size):
            self.sample_idx += 1
            x = -4.0 * float(self.sample_idx - self.position) / float(self.width)
            
            probability_drift = self._methods[self.probability_function](self,x)
        
            if self.position + self.width == self.sample_idx:
                self.position = self.sample_idx + self.width + self.pause
                self.probability_function = "inv_sigmoid_prob" if self.probability_function == "sigmoid_prob" else "sigmoid_prob"

                
            if self.random_state.rand() > probability_drift:
                X, y = self._input_stream.next_sample()
            else:
                X, y = self._drift_stream.next_sample()
                
            self.current_sample_x[j, :] = X
            self.current_sample_y[j, :] = y

        return self.current_sample_x, self.current_sample_y.flatten()

    def inv_sigmoid_prob(self,x):
        try:
            return 1 - 1.0 / (1.0 + np.exp(x))
        except:
            # Overflow in exp function return value gets 1 - asymptotically 0
            return 1 - 1e-8

    def sigmoid_prob(self,x):
        try:
            return  1.0 / (1.0 + np.exp(x))
        except Exception:
            # Overflow in exp function return value gets asymptotically 0
            return 1e-8
            
    def restart(self):
        self.random_state = check_random_state(self._original_random_state)
        self.sample_idx = 0
        self._input_stream.restart()
        self._drift_stream.restart()

    def get_info(self):
        """Collects information about the generator.

        Returns
        -------
        string
            Configuration for the generator object.
        """
        description = type(self).__name__ + ': '
        description += 'First Stream: {} - '.format(type(self._input_stream).__name__)
        description += 'Drift Stream: {} - '.format(type(self._drift_stream).__name__)
        description += 'alpha: {} - '.format(self.alpha)
        description += 'position: {} - '.format(self.position)
        description += 'width: {} - '.format(self.width)
        return description
    
    _methods = {
    'inv_sigmoid_prob': inv_sigmoid_prob,
    'sigmoid_prob': sigmoid_prob}
