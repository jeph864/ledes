
import os
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from sklearn.preprocessing import normalize,MultiLabelBinarizer
from logzero import logger
from multiprocessing import Process
from scipy import sparse

from pecos.xmc.base import *
import pecos

class BalancedKMeans(Indexer):
    """Indexer using Balanced K-means.
    """

    KMEANS = 0  # KMEANS
    SKMEANS = 5  # Spherical KMEANS

    @dc.dataclass
    class TrainParams(pecos.BaseParams):  # type: ignore
        """Training Parameters of Balanced K-means.
        nr_splits (int, optional): The out-degree of each internal node of the tree. Ignored if `imbalanced_ratio != 0` because imbalanced clustering supports only 2-means. Default is `16`.
        min_codes (int): The number of direct child nodes that the top level of the hierarchy should have.
        max_leaf_size (int, optional): The maximum size of each leaf node of the tree. Default is `100`.
        imbalanced_ratio (float, optional): Value between `0.0` and `0.5` (inclusive). Indicates how relaxed the balancedness constraint of 2-means can be. Specifically, if an iteration of 2-means is clustering `L` labels, the size of the output 2 clusters will be within approx `imbalanced_ratio * 2 * L` of each other. Default is `0.0`.
        imbalanced_depth (int, optional): Maximum depth of imbalanced clustering. After depth `imbalanced_depth` is reached, balanced clustering will be used. Default is `100`.
        spherical (bool, optional): True will l2-normalize the centroids of k-means after each iteration. Default is `True`.
        seed (int, optional): Random seed. Default is `0`.
        kmeans_max_iter (int, optional): Maximum number of iterations for each k-means problem. Default is `20`.
        threads (int, optional): Number of threads to use. `-1` denotes all CPUs. Default is `-1`.
        """

        nr_splits: int = 16
        min_codes: int = None  # type: ignore
        max_leaf_size: int = 100
        imbalanced_ratio: float = 0.0
        imbalanced_depth: int = 100
        spherical: bool = True
        seed: int = 0
        kmeans_max_iter: int = 20
        threads: int = -1

    @classmethod
    def gen(
        cls,
        feat_mat,
        train_params=None,
        dtype=np.float32,
        **kwargs,
    ):
        """Generate a cluster chain by using hierarchical k-means.
        Args:
            feat_mat (numpy.ndarray or scipy.sparse.csr.csr_matrix): Matrix of label features.
            train_params (HierarchicalKMeans.TrainParams, optional): training parameters for indexing.
            dtype (type, optional): Data type for matrices. Default is `numpy.float32`.
            **kwargs: Ignored.
        Returns:
            ClusterChain: The generated cluster chain.
        """
        if train_params is None:
            train_params = cls.TrainParams.from_dict(kwargs)
        else:
            train_params = cls.TrainParams.from_dict(train_params)

        if train_params.min_codes is None:
            train_params.min_codes = train_params.nr_splits

        LOGGER.debug(
            f"HierarchicalKMeans train_params: {json.dumps(train_params.to_dict(), indent=True)}"
        )

        # use optimized c++ clustering code if doing balanced clustering
        if train_params.imbalanced_ratio == 0:
            nr_instances = feat_mat.shape[0]
            if train_params.max_leaf_size >= nr_instances:
                # no-need to do clustering
                return ClusterChain.from_partial_chain(
                    smat.csc_matrix(np.ones((nr_instances, 1), dtype=np.float32))
                )

            depth = max(1, int(math.ceil(math.log2(nr_instances / train_params.max_leaf_size))))
            if (2**depth) > nr_instances:
                raise ValueError(
                    f"max_leaf_size > 1 is needed for feat_mat.shape[0] == {nr_instances} to avoid empty clusters"
                )

            algo = cls.SKMEANS if train_params.spherical else cls.KMEANS

            assert feat_mat.dtype == np.float32
            if isinstance(feat_mat, (smat.csr_matrix, ScipyCsrF32)):
                py_feat_mat = ScipyCsrF32.init_from(feat_mat)
            elif isinstance(feat_mat, (np.ndarray, ScipyDrmF32)):
                py_feat_mat = ScipyDrmF32.init_from(feat_mat)
            else:
                raise NotImplementedError(
                    "type(feat_mat) = {} is not supported.".format(type(feat_mat))
                )

            codes = np.zeros(py_feat_mat.rows, dtype=np.uint32)
            codes = clib.run_clustering(
                py_feat_mat,
                depth,
                algo,
                train_params.seed,
                codes=codes,
                kmeans_max_iter=train_params.kmeans_max_iter,
                threads=train_params.threads,
            )
            C = cls.convert_codes_to_csc_matrix(codes, depth)
            cluster_chain = ClusterChain.from_partial_chain(
                C, min_codes=train_params.min_codes, nr_splits=train_params.nr_splits
            )
        else:
            cluster_chain = hierarchical_kmeans(
                feat_mat,
                max_leaf_size=train_params.max_leaf_size,
                imbalanced_ratio=train_params.imbalanced_ratio,
                imbalanced_depth=train_params.imbalanced_depth,
                spherical=train_params.spherical,
                seed=train_params.seed,
                kmeans_max_iter=train_params.kmeans_max_iter,
                threads=train_params.threads,
            )
            cluster_chain = ClusterChain(cluster_chain)
        return cluster_chain

    @staticmethod
    def convert_codes_to_csc_matrix(codes, depth):
        """Convert a 1d array of cluster assignments into a binary clustering matrix format.
        Args:
            codes (numpy.ndarray): 1d array of integers. Each index of the array corresponds to a label index, each value of the array is the cluster index.
            depth (int): The depth of the hierarchical tree.
        Returns:
            scipy.sparse.csc.csc_matrix: A binary matrix of shape `(len(codes), 1 << depth)`. An entry `(r, c)` in the matrix has value `1` if and only if `codes[r] == c`.
        """

        nr_codes = 1 << depth
        nr_elements = len(codes)

        indptr = np.cumsum(np.bincount(codes + 1, minlength=(nr_codes + 1)), dtype=np.uint64)
        indices = np.argsort(codes * np.float64(nr_elements) + np.arange(nr_elements))
        C = smat_util.csc_matrix(
            (np.ones_like(indices, dtype=np.float32), indices, indptr),
            shape=(nr_elements, nr_codes),
        )
        return C



def build_tree_by_level(sparse_x, sparse_y, eps, max_leaf, levels, groups_path):
    os.makedirs(os.path.split(groups_path)[0], exist_ok=True)
    logger.info('Clustering')
    logger.info('Getting Labels Feature')
    labels_f = normalize(csr_matrix(sparse_y.T) @ csc_matrix(sparse_x))
    logger.info(F'Start Clustering {levels}')
    levels, q = [2**x for x in levels], None
    for i in range(len(levels)-1, -1, -1):
        if os.path.exists(f'{groups_path}/Level-{i}.npy'):
            labels_list = np.load(f'{groups_path}/Level-{i}.npy',allow_pickle=True)
            logger.info(F'Loaded  Level-{i} from memory')
            q = [(labels_i, labels_f[labels_i]) for labels_i in labels_list]
            break
    if q is None:
        q = [(np.arange(labels_f.shape[0]), labels_f)]
    while q:
        labels_list = np.asarray([x[0] for x in q])
        assert sum(len(labels) for labels in labels_list) == labels_f.shape[0]
        if len(labels_list) in levels:
            level = levels.index(len(labels_list))
            logger.info(F'Finish Clustering Level-{level}')
            np.save(f'{groups_path}/Level-{level}.npy', np.asarray(labels_list))
            logger.info(F'Finished saving Level-{level}')
        else:
            logger.info(F'Finish Clustering {len(labels_list)}')
        next_q = []
        for node_i, node_f in q:
            if len(node_i) > max_leaf:
                next_q += list(split_node(node_i, node_f, eps))
        q = next_q
    logger.info('Finish Clustering')


def split_node(labels_i: np.ndarray, labels_f: csr_matrix, eps: float):
    n = len(labels_i)
    c1, c2 = np.random.choice(np.arange(n), 2, replace=False)
    centers, old_dis, new_dis = labels_f[[c1, c2]].toarray(), -10000.0, -1.0
    l_labels_i, r_labels_i = None, None
    while new_dis - old_dis >= eps:
        dis = labels_f @ centers.T  # N, 2
        partition = np.argsort(dis[:, 1] - dis[:, 0])
        l_labels_i, r_labels_i = partition[:n//2], partition[n//2:]
        old_dis, new_dis = new_dis, (dis[l_labels_i, 0].sum() + dis[r_labels_i, 1].sum()) / n
        centers = normalize(np.asarray([np.squeeze(np.asarray(labels_f[l_labels_i].sum(axis=0))),
                                        np.squeeze(np.asarray(labels_f[r_labels_i].sum(axis=0)))]))
    return (labels_i[l_labels_i], labels_f[l_labels_i]), (labels_i[r_labels_i], labels_f[r_labels_i])

def get_mlb(mlb_path, labels=None) -> MultiLabelBinarizer:
    if os.path.exists(mlb_path):
        return joblib.load(mlb_path)
    mlb = MultiLabelBinarizer(sparse_output=True)
    mlb.fit(labels)
    joblib.dump(mlb, mlb_path)
    return mlb
def run(**kwargs):
    sparse_x = kwargs['sparse_x']
    sparse_y = kwargs['sparse_y']
    eps = kwargs['eps']
    max_leaf = kwargs['max_leaf']
    levels = kwargs['levels']
    groups_path = kwargs['groups_path']
    build_tree_by_level(sparse_x, sparse_y, eps, max_leaf, levels,groups_path)
    #cluster_process = Process(target=build_tree_by_level,
    #                             args=(sparse_x, sparse_y, eps, max_leaf, levels,groups_path))
    #cluster_process.start()
    #cluster_process.join()
    #cluster_process.close()

