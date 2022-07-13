import numpy as np
import pickle
import logging
from pecos.utils import logging_util

from pecos.utils import smat_util
from pecos.core import clib

from pecos.xmc.xlinear import XLinearModel
from configuration import parse_evaluation_arguments
from scipy import sparse
from pecos.utils import logging_util

LOGGER = logging.getLogger(__name__)

def do_prediction(args):
    """Predict and Evaluate for xlinear model
    """

    # Load data
    Xt = XLinearModel.load_feature_matrix(args.inst_path)

    if Xt.dtype == np.float64:
        Xt = Xt.astype('float32')


    if args.selected_output is not None:
        # Selected Output
        selected_outputs_csr = XLinearModel.load_feature_matrix(args.selected_output)
        xlinear_model = XLinearModel.load(
            args.model_folder, is_predict_only=True, weight_matrix_type="CSC"
        )
    else:
        # TopK
        selected_outputs_csr = None
        xlinear_model = XLinearModel.load(args.model_folder, is_predict_only=True)
    if args.overlap_model_folder:
        if args.selected_output is not None:
            # Selected Output
            overlap_selected_outputs_csr = XLinearModel.load_feature_matrix(args.selected_output)
            overlap_xlinear_model = XLinearModel.load(
                args.overlap_model_folder, is_predict_only=True, weight_matrix_type="CSC"
            )
        else:
            # TopK
            overlap_selected_outputs_csr = None
            overlap_xlinear_model = XLinearModel.load(args.overlap_model_folder, is_predict_only=True)
    # Model Predicting
    Yt_pred = xlinear_model.predict(
        Xt,
        selected_outputs_csr=selected_outputs_csr,
        topk=args.topk,
        beam_size=args.beam_size,
        post_processor=args.post_processor,
        threads=args.threads,
        max_pred_chunk=args.max_pred_chunk,
    )

    Yt_pred_overlap = overlap_xlinear_model.predict(
            Xt,
            selected_outputs_csr=overlap_selected_outputs_csr,
            topk=args.topk,
            beam_size=args.beam_size,
            post_processor=args.post_processor,
            threads=args.threads,
            max_pred_chunk=args.max_pred_chunk,
        )
    if args.overlap_model_folder:
        #save the results
        pred_path = f"{args.overlap_model_folder}/predictions"
        smat_util.save_matrix(pred_path, Yt_pred_overlap)

    # Save prediction
    if args.save_pred_path:
        smat_util.save_matrix(args.save_pred_path, Yt_pred)

    # Evaluate
    if args.label_path:
        Yt = XLinearModel.load_label_matrix(args.label_path)
        metric = smat_util.Metrics.generate(Yt, Yt_pred, args.topk)
        LOGGER.info("<><><><><><> evaluation results(XR-Model) <><><><><><>")
        LOGGER.info(metric)
        if Yt_pred_overlap is not None:
            metric = smat_util.Metrics.generate(Yt, Yt_pred_overlap, args.topk)
            LOGGER.info("<><><><><><> evaluation results(New Model) <><><><><><>")
            LOGGER.info(metric)

def get_precision(Yt, Y_pred, topk):
    pass


def get_matcher(model, X, beam_size):
    chain_models = model.model.model_chain
    len_chain = len(chain_models)
    matching_matrix = None
    for layer in range(len_chain -1):
        tmp_matching_matrix = chain_models[layer].predict(
            X,
            csr_codes = matching_matrix,
            only_topk = beam_size,
            post_processsor = "l3-hinge"
        )
        matching_matrix = tmp_matching_matrix
    return matching_matrix
def mask_out_unused_labels(all_labels_count, unused_labels):
    U = None
    if len(unused_labels) == 0:
        U = np.ones(all_labels_count)
    else:
        U = np.ones(all_labels_count)[unused_labels]
    new_shape = (all_labels_count, all_labels_count)
    return sparse.diags(U, shape= new_shape, format='csr', dtype= np.float32)
        

def deduplicate_labels_max(mapper, yt, y_pred):
    """
    Deduplication at inference time: the prediction results are merged with the ground truth to to generate aggreagted scores.
    (referenced libpecos implementation)--> maximum
    Returns: 
        deduplicated scores
    """
    y_pred = y_pred.tolil()
    yt = yt.tolil()
    labels_count = y_pred.shape[1] - len(mapper)
    i=0
    for pseudo, real in mapper.items():
        y_pred[:, real] = y_pred[:, real].maximum(y_pred[:, pseudo])
        if i%100 == 0:
            LOGGER.info(f"Pseudo Labels {i} completed")
        i = i+1
    return yt[:, :labels_count].tocsr(), y_pred[:, :labels_count].tocsr()
def deduplicate_labels_mean(mapper, yt, y_pred):
    """
    
    """
    labels_count = y_pred.shape[1] - len(mapper)
    duplicates_count = y_pred.shape[1]
    mapping_matrix = get_label_mapping(mapper,labels_count, duplicates_count )
    pred_norm = clib.sparse_matmul(y_pred, mapping_matrix)
    #remove bias
    bias = clib.sparse_matmul(y_pred.sign(), mapping_matrix)
    bias.data = 1.0/np.clip(bias.data, a_min=0.1, a_max=None)
    y_pred = pred_norm.multiply(bias)
    LOGGER.info("Finished removing bias")
    return yt[:, :labels_count].tocsr(), y_pred[:, :labels_count].tocsr() 
    


def get_label_mapping(mapper, labels_count, duplicate_labels_count):
    """
    Uses the mapper to get the mapping of duplicate labels to their original label id
    """
    mapper_rows = []
    mapper_cols = []
    mapper_data = []
    for i in range(labels_count, duplicate_labels_count ):
        mapper_rows.append(i -labels_count)
        mapper_cols.append(mapper[i])
        mapper_data.append(1)
    mapper_coo_matrix = sparse.coo_matrix(
        (mapper_data, (mapper_rows, mapper_cols)), 
        shape = (duplicate_labels_count - labels_count, labels_count), 
        dtype=np.float32
        )
    #fill the first labels_count x labels_count with an Identity matrix
    I = sparse.diags(np.ones(labels_count), format="csc", dtype=np.float32)
    return sparse.vstack((I, mapper_coo_matrix), format="csc")

def batch_wise_prediction(model, X, batch_size, beam_size , only_topk, post_processor):
    prediction_batches, matcher_batches = [], []
    n_batches = (X.shape[0]-1) // batch_size +1
    LOGGER.info(f"Predicting for a total of {n_batches} batches")
    for batch_idx in range(n_batches):
        batch_x = get_batch(X, batch_idx, batch_size)
        single_batch_matcher = get_matcher(model, batch_x, beam_size)
        matcher_batches.append(single_batch_matcher)
        single_batch_pred = model.predict(batch_x, 
            beam_size = beam_size,
            only_topk = only_topk,
            post_processor = post_processor
        )
        prediction_batches.append(single_batch_pred)
    #LOGGER.info(f"1st Matcher size; {len(matcher_batches[0])} / Predicted : {len(prediction_batches[0])}")
    return matcher_batches, prediction_batches

def get_batch(X, i, batch_size):
    batch_start = i*batch_size
    batch_end = min((1+i) * batch_size, X.shape[0])
    return X[batch_start: batch_end, :]  

 
def evaluate_new_model(args):
     # Load Data
    LOGGER.info("started loading the instances")
    Xt = XLinearModel.load_feature_matrix(args.inst_path)
    Yt = XLinearModel.load_label_matrix(args.label_path)
    LOGGER.info("Finished loading instances")

    if Xt.dtype == np.float64:
        Xt = Xt.astype('float32')
    
    mapper, unused_labels = {}, {}
    with open(args.mapper, "rb") as f:
        mapper = pickle.load(f)
    with open(args.unused_labels, "rb") as r:
        unused_labels = pickle.load(r)
    LOGGER.info(f"#Unused labels: {len(unused_labels)}")
    
    #load model
    xlinear = XLinearModel.load(args.model_folder)
    LOGGER.info("Finished loading the model")
    #Get Cluster Matrix
    C = xlinear
    batch_size = 8192 * 16
    beam_size = args.beam_size
    only_topk = 160
    post_processor = "l3-hinge"#For ranker
    #batchwise matching and prediction
    matcher_batches, pred_batches = batch_wise_prediction(xlinear, Xt, batch_size, beam_size, only_topk, post_processor)
    Y_pred = sparse.vstack(pred_batches)
    #Binarize the Matching matrix(see paper), C is already binarized
    matching_matrix = sparse.vstack(matcher_batches)
    binarized_matcher = smat_util.binarized(matching_matrix)
    C = xlinear.model.model_chain[-1].pC.buf
    #Equation 8 from the paper
    BinaryMC = clib.sparse_matmul(binarized_matcher, C.transpose())

    transformed_unused_labels = mask_out_unused_labels(Y_pred.shape[1], unused_labels)

    Y_pred = clib.sparse_matmul(Y_pred, transformed_unused_labels)
    LOGGER.info("Merging labels(pseudo and real)")

    Yt, Y_pred = deduplicate_labels_mean(mapper, Yt, Y_pred)
    LOGGER.info("Calculating metrics")
    metric = smat_util.Metrics.generate(Yt, Y_pred, topk=args.topk)
    LOGGER.info(metric)

    avg_inner_prod = BinaryMC.sum(axis=1).mean(axis=0)[0, 0]
    LOGGER.info(f"Average #inner prod: {avg_inner_prod}")



    


get_p_1 = lambda Yt, Y_pred : get_precision(Yt, Y_pred, 1)
get_p_3 = lambda Yt, Y_pred : get_precision(Yt, Y_pred, 3)
get_p_5 = lambda Yt, Y_pred : get_precision(Yt, Y_pred, 5)

        




if __name__ == "__main__":
    logging_util.setup_logging_config(level=2)
    LOGGER.info("parse args")
    args = parse_evaluation_arguments()
    LOGGER.info(args)
    #do_prediction(args)
    evaluate_new_model(args)

    