import torch
import numpy as np
from sklearn.metrics import average_precision_score


def compute_dist_mat(representations, test_dataset):
    print("Computing Distance Matrix")
    query_set = test_dataset.query_data
    gallery_set = test_dataset.gallery_data

    query_representations = []
    for datum in query_set:
        file_name = datum.img_path.split('/')[-1]
        query_representations.append(representations[file_name].unsqueeze(0))
    query_representations = torch.cat(query_representations, 0)

    gallery_representations = []
    for datum in gallery_set:
        file_name = datum.img_path.split('/')[-1]
        gallery_representations.append(representations[file_name].unsqueeze(0))
    gallery_representations = torch.cat(gallery_representations, 0)

    m, n = query_representations.size(0), gallery_representations.size(0)

    dist_mat = torch.pow(query_representations, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gallery_representations, 2).sum(dim=1, keepdim=True).expand(n, m).t()

    dist_mat.addmm_(query_representations, gallery_representations.t(), beta=1, alpha=-2)

    return dist_mat


def evaluate(dist_mat, test_dataset, cmc_top_k=(1, 5, 10)):
    query_set = test_dataset.query_data
    gallery_set = test_dataset.gallery_data

    query_ids = []
    query_cams = []
    for datum in query_set:
        query_ids.append(int(datum.class_name))
        query_cams.append(datum.domain_label)

    gallery_ids = []
    gallery_cams = []
    for datum in gallery_set:
        gallery_ids.append(int(datum.class_name))
        gallery_cams.append(datum.domain_label)

    mAP = compute_mean_ap(dist_mat, query_ids, gallery_ids, query_cams, gallery_cams)
    print("Mean AP: {:4.2%}".format(mAP))
    cmc_scores = compute_cmc(dist_mat, query_ids, gallery_ids, query_cams, gallery_cams, first_match_break=True)
    for k in cmc_top_k:
        print('top-{:<4}{:12.2%}'.format(k, cmc_scores[k-1]))


def compute_mean_ap(dist_mat, query_ids, gallery_ids, query_cams, gallery_cams):
    print("Computing mAP")
    dist_mat = dist_mat.numpy()
    num_query, num_gallery = dist_mat.shape
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    query_cams = np.asarray(query_cams)
    gallery_cams = np.asarray(gallery_cams)

    indices = np.argsort(dist_mat, axis=1)
    # Replace Indexes to ID values.
    sorted_gallery_ids = gallery_ids[indices]
    matches = sorted_gallery_ids == query_ids[:, np.newaxis]

    aps = []
    for i in range(num_query):
        valid_query = (gallery_ids[indices[i]] != query_ids[i]) | (gallery_cams[indices[i]] != query_cams[i])
        valid_label_true = matches[i, valid_query]
        if not np.any(valid_label_true):
            continue
        valid_label_pred = -dist_mat[i][indices[i]][valid_query]
        aps.append(average_precision_score(valid_label_true, valid_label_pred))

    if len(aps) == 0:
        raise RuntimeError("No Valid Query")

    return np.mean(aps)


def compute_cmc(dist_mat, query_ids, gallery_ids, query_cams, gallery_cams, top_k=100, first_match_break=False):
    print("Computing CMC Scores")
    dist_mat = dist_mat.numpy()
    num_query, num_gallery = dist_mat.shape
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    query_cams = np.asarray(query_cams)
    gallery_cams = np.asarray(gallery_cams)

    indices = np.argsort(dist_mat, axis=1)
    # Replace Indexes to ID values.
    sorted_gallery_ids = gallery_ids[indices]
    matches = sorted_gallery_ids == query_ids[:, np.newaxis]

    ret = np.zeros(top_k)
    num_valid_queries = 0

    for i in range(num_query):
        valid_query = (gallery_ids[indices[i]] != query_ids[i]) | (gallery_cams[indices[i]] != query_cams[i])
        valid_label_true = matches[i, valid_query]
        if not np.any(valid_label_true):
            continue
        valid_label_true_index = np.nonzero(valid_label_true)[0]
        delta = 1. / len(valid_label_true_index)

        for order_true, order_pred in enumerate(valid_label_true_index):
            if order_pred - order_true >= top_k:
                break

            if first_match_break:
                ret[order_pred - order_true] += 1
                break
            else:
                ret[order_pred - order_true] += delta

        num_valid_queries += 1

    if num_valid_queries == 0:
        raise RuntimeError("No Valid Query")

    return ret.cumsum() / num_valid_queries
