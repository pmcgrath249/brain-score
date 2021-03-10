# TODO think about ceiling
# TODO double-check score function

import numpy as np
from scipy.spatial.distance import squareform, pdist

import brainscore
from brainscore.benchmarks import BenchmarkBase
from brainscore.benchmarks._neural_common import average_repetition
from brainscore.metrics.distribution_similarity import ks_similarity # tiagos brainscore copy
from brainscore.benchmarks.perturbation_prototypes import Rajalingham2019

BIBTEX = '''@article {Lee2020.07.09.185116,
	        author = {Lee, Hyodong and Margalit, Eshed and Jozwik, Kamila M. and Cohen, Michael A. and Kanwisher, Nancy and Yamins, Daniel L. K. and DiCarlo, James J.},
	        title = {Topographic deep artificial neural networks reproduce the hallmarks of the primate inferior temporal cortex face processing network},
	        elocation-id = {2020.07.09.185116},
	        year = {2020},
	        doi = {10.1101/2020.07.09.185116},
	        publisher = {Cold Spring Harbor Laboratory},
	        abstract = {A salient characteristic of monkey inferior temporal (IT) cortex is the IT face processing network. Its hallmarks include: {\textquotedblleft}face neurons{\textquotedblright} that respond more to faces than non-face objects, strong spatial clustering of those neurons in foci at each IT anatomical level ({\textquotedblleft}face patches{\textquotedblright}), and the preferential interconnection of those foci. While some deep artificial neural networks (ANNs) are good predictors of IT neuronal responses, including face neurons, they do not explain those face network hallmarks. Here we ask if they might be explained with a simple, metabolically motivated addition to current ANN ventral stream models. Specifically, we designed and successfully trained topographic deep ANNs (TDANNs) to solve real-world visual recognition tasks (as in prior work), but, in addition, we also optimized each network to minimize a proxy for neuronal wiring length within its IT layers. We report that after this dual optimization, the model IT layers of TDANNs reproduce the hallmarks of the IT face network: the presence of face neurons, clusters of face neurons that quantitatively match those found in IT face patches, connectivity between those patches, and the emergence of face viewpoint invariance along the network hierarchy. We find that these phenomena emerge for a range of naturalistic experience, but not for highly unnatural training. Taken together, these results show that the IT face processing network could be a consequence of a basic hierarchical anatomy along the ventral stream, selection pressure on the visual system to accomplish general object categorization, and selection pressure to minimize axonal wiring length.Competing Interest StatementThe authors have declared no competing interest.},
            URL = {https://www.biorxiv.org/content/early/2020/07/10/2020.07.09.185116},
            eprint = {https://www.biorxiv.org/content/early/2020/07/10/2020.07.09.185116.full.pdf},
            journal = {bioRxiv}}'''

class DicarloLee2020ITTissueResponseCorrelation(BenchmarkBase):

    def __init__(self, metric=ks_similarity, bootstrap_samples=100_000, num_sample_arrays_candidate=10):
        super(self).__init__(identifier='dicarlo.Lee2020.IT-tissue',
                             ceiling_func=None,
                             version=0.1,
                             parent='IT',
                             bibtex=BIBTEX)

        self._stimulus_set = Rajalingham2019()._target_assembly.stimulus_set
        assembly = make_static(brainscore.get_assembly('dicarlo.MajajHong2015').sel(region='IT'))
        self._target_assembly = assembly[:,np.in1d(assembly.presentation.image_id,
                                                   list(set(self._stimulus_set['image_id'])))]

        self._arr_size_mm = (np.ptp(self._target_assembly.neuroid.x.data),
                             np.ptp(self._target_assembly.neuroid.y.data))
        self.bootstrap_samples = bootstrap_samples
        self.num_sample_arrs = num_sample_arrays_candidate

        self._target_statistic_fn = self.create_tissue_response_correlation_target
        self._metric = metric

    def __call__(self, candidate):

        candidate.start_recording(recording_target='IT', time_bins=[(70,170)])
        activations = candidate.look_at(self._stimulus_set)
        activations = make_static(activations)

        # sample response correlation as a function of distance from candidate neuroids
        rcd_candidate = []
        for window in sample_array_locations(activations.neuroid, self.num_sample_arrs, self._arr_size_mm):
            rcd_candidate.append(sample_response_corr_vs_dist(activations[window],
                                                              int(self.bootstrap_samples/self.num_sample_arrs)))

        return self.score(np.hstack(rcd_candidate))

    def create_tissue_response_correlation_target(self):

        # sample response correlation as a function of distance from target assembly
        rcd_target = []
        for animal in set(self._target_assembly.neuroid.animal.data):
            for arr in set(self._target_assembly.neuroid.arr.data):

                sub_assembly = self._target_assembly.sel(animal=animal, arr=arr)
                bootstrap_samples_sub_assembly = int(self.bootstrap_samples*(sub_assembly.neuroid.size/
                                                                             self._target_assembly.neuroid.size))

                rcd_target.append(sample_response_corr_vs_dist(sub_assembly, bootstrap_samples_sub_assembly))

        return np.hstack(rcd_target)

    def score(self, candidate_statistic):

        self._target_statistic = self._target_statistic_fn()

        def bin_indices():
            t = np.where(np.logical_and(self._target_statistic[0] >= lower_bound_mm,
                                        self._target_statistic[0] < lower_bound_mm+0.1))[0]
            c = np.where(np.logical_and(candidate_statistic[0] >= lower_bound_mm,
                                        candidate_statistic[0] < lower_bound_mm + 0.1))[0]
            return t, c, t.size>30 & c.size>30 # TODO 30 neurons per bin enough for stable estimate?

        bin_scores = []
        arr_diag_mm = round(np.linalg.norm(self._arr_size_mm), 1)
        for lower_bound_mm in np.linspace(0, arr_diag_mm, int(arr_diag_mm*10+1)):
            in_bin_t, in_bin_c, enough_data = bin_indices()
            if enough_data:
                # TODO only evaluate on bins with min number of neurons for stability?
                bin_scores.append(self._metric(self._target_statistic[1, in_bin_t],
                                               candidate_statistic[1, in_bin_c]))

        # TODO think aobut how to avg: log scale?, #neurons in interval
        return np.mean(bin_scores)

def sample_response_corr_vs_dist(assembly, num_samples):

    avg_resp = average_repetition(assembly)

    rng = np.random.default_rng()
    neuroid_pairs = rng.integers(0, assembly.shape[0], (2, num_samples))

    pairwise_distances_all = pairwise_distances(assembly)
    pairwise_distance_samples = pairwise_distances_all[(*neuroid_pairs,)]

    response_samples = avg_resp.data[neuroid_pairs]
    response_correlation_samples = np.nan_to_num(corrcoef_rowwise(*response_samples))

    return np.vstack((pairwise_distance_samples, response_correlation_samples))

def corrcoef_rowwise(A, B):
    # https://stackoverflow.com/questions/41700840/correlation-of-2-time-dependent-multidimensional-signals-signal-vectors

    A_mA = A - A.mean(1)[:,None]
    B_mB = B - B.mean(1)[:,None]
    ssA = np.einsum('ij,ij->i', A_mA, A_mA) # var A
    ssB = np.einsum('ij,ij->i', B_mB, B_mB) # var B
    return np.einsum('ij,ij->i', A_mA, B_mB) / np.sqrt(ssA * ssB) # cov/sqrt(varA*varB)

def pairwise_distances(assembly):
    # distances for all pairs

    if hasattr(assembly, 'tissue_x'):
        locations = np.stack([assembly.neuroid.tissue_x.data, assembly.neuroid.tissue_y.data]).T
    else:
        locations = np.stack([assembly.neuroid.x.data, assembly.neuroid.y.data]).T

    return squareform(pdist(locations, metric='euclidean'))

def sample_array_locations(neuroid, num_sample_arrs, arr_size_mm):

    bound_max_x, bound_max_y = np.max([neuroid.tissue_x.data, neuroid.tissue_y.data], axis=1) - arr_size_mm

    rng = np.random.default_rng()
    lower_corner = np.column_stack((rng.choice(neuroid.tissue_x.data[neuroid.tissue_x.data<=bound_max_x],
                                               size=num_sample_arrs),
                                    rng.choice(neuroid.tissue_y.data[neuroid.tissue_y.data<=bound_max_y],
                                               size=num_sample_arrs)))
    upper_corner = lower_corner+arr_size_mm

    # create index masks of neuroids within sample windows
    window_masks = []
    for i in range(num_sample_arrs):
        window_masks.append(np.logical_and.reduce([neuroid.tissue_x.data <= upper_corner[i, 0],
                                                   neuroid.tissue_x.data >= lower_corner[i, 0],
                                                   neuroid.tissue_y.data <= upper_corner[i, 1],
                                                   neuroid.tissue_y.data >= lower_corner[i, 1]]))

    return window_masks

def make_static(assembly):

    if 'time_bin' in assembly.dims:
        assembly = assembly.squeeze('time_bin')
    if hasattr(assembly, "time_step"):
        assembly = assembly.squeeze("time_step")

    return assembly

'''
import pytest

def rand_corr_data():
    a = brainscore.get_assembly('dicarlo.MajajHong2015').sel(region='IT')
    a.data = np.random.rand(a.data.size).reshape(a.data.shape)
    if 'time_bin' in a.dims:
        a = a.squeeze('time_bin')
    if hasattr(a, "time_step"):
        a = a.squeeze("time_step")
    return a

def max_corr_data():
    a = brainscore.get_assembly('dicarlo.MajajHong2015').sel(region='IT')
    a.data = np.arange(a.data.size).reshape(a.data.shape)
    if 'time_bin' in d.dims:
        d = d.squeeze('time_bin')
    if hasattr(d, "time_step"):
        d = d.squeeze("time_step")
    return a

def extreme_corr_data():
    a = max_corr_data()
    a.data[0:-1:2] = a.data[0:-1:2]*-1
    return a

@pytest.mark.parametrize([('rand', rand_corr_data), ('extreme', extreme_corr_data), ('max', max_corr_data)])
def corr_test(test, assembly):
    b = Benchmark_Bootstrap(assembly)
    (t, c) = b.__call__(assembly)

    if test=='extreme':
        # extreme r=0, std=1
        assert np.abs(np.mean(c[1])) < 0.01
        assert np.std(c[1])-1 < 0.01

    if test='rand':
        # rand r=0 w/ std small
        assert np.abs(np.mean(c[1])) < 0.01
        assert np.std(c[1]) < .2

    if test='max':
        # same r=1 w/ std=0
        assert np.abs(np.mean(c[1]))-1 < 0.01
        assert np.std(c[1]) < 0.01 

def pairwise_dist_mat_symmetry():
    assert pwd==pwd.T

def test_target_assembly():
    b = DicarloLee2020TissueResponseCorrelation()
    assert b._target_assembly.dims[0] == 'neuroid'
    assert b._target_assembly.dims[1] = 'presentation'
    assert len(b._target_assembly.shape) == 2
    
    
    
    
    
    
    
Not whole cov mat

def response_corr_vs_dist(assembly, bsdfaf, xasd):

    # avg. responses over presentations of the same stimuli
    avg_resp = []
    for same_stim_presentations in assembly.presentation.groupby('stimulus').groups.values(): # TODO make sure it is stimulus vs. id
        avg_resp.append(np.mean(assembly[:, same_stim_presentations], axis=1))
    avg_resp = np.column_stack(avg_resp)

    # pairwise response correlation
    pairwise_response_correlation_mat = np.nan_to_num(np.corrcoef(avg_resp))

    # pairwise distances
    pairwise_distance_mat = self.pairwise_distances(assembly)

    # create bootstrap samples
    sample_coord_pairs = self.sample_coord_pairs(0, assembly.shape[0], self.bootstrap_samples)

    # sample & put into one matrix
    pairwise_distance_sampled = pairwise_distance_mat[sample_coord_pairs]
    pairwise_response_correlation_sampled = pairwise_response_correlation_mat[sample_coord_pairs]
    resp_corr_vs_dist = np.vstack((pairwise_distance_sampled, pairwise_response_correlation_sampled))

    return resp_corr_vs_dist
'''