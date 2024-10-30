import torch
import numpy as np
from torch.linalg import eigvalsh
from torch.distributions import Categorical
import os


def normalized_gaussian_kernel(x, y, sigma, batchsize):
    """
    calculate the kernel matrix, the shape of x and y should be equal except for the batch dimension

    x:
        input, dim: [batch, dims]
    y:
        input, dim: [batch, dims]
    sigma:
        bandwidth parameter
    batchsize:
        Batchify the formation of kernel matrix, trade time for memory
        batchsize should be smaller than length of data

    return:
        scalar : mean of kernel values
    """
    batch_num = (y.shape[0] // batchsize) + 1
    assert x.shape[1:] == y.shape[1:]

    total_res = torch.zeros((x.shape[0], 0), device=x.device)
    for batchidx in range(batch_num):
        y_slice = y[batchidx * batchsize : min((batchidx + 1) * batchsize, y.shape[0])]
        res = torch.norm(x.unsqueeze(1) - y_slice, dim=2, p=2).pow(2)
        res = torch.exp((-1 / (2 * sigma * sigma)) * res)
        total_res = torch.hstack([total_res, res])

        del res, y_slice

    total_res = total_res / np.sqrt(x.shape[0] * y.shape[0])

    return total_res


def print_novelty_metrics(eigenvalues, args):
    """
    Output and save the KEN score and other statistics to log file
    """

    postive_eigenvalues = eigenvalues * (eigenvalues > 0)
    prob_postive_eigenvalues = postive_eigenvalues / postive_eigenvalues.sum()
    sum_of_postive_eigenvalues = postive_eigenvalues.sum()
    entropy = Categorical(probs=postive_eigenvalues).entropy()

    args.logger.info(
        "Top eigenvalues: "
        + str(postive_eigenvalues.topk(min(100, len(postive_eigenvalues))).values.data)
    )
    args.logger.info(
        "Sum of positive eigenvalues (Total novel frequency): {:.4f}".format(
            postive_eigenvalues.sum()
        )
    )
    args.logger.info(
        "Sum of square of positive eigenvalues: {:.6f}".format(
            postive_eigenvalues.pow(2).sum()
        )
    )
    args.logger.info("Shannon Entropy: {:.4f}".format(entropy))
    args.logger.info("KEN score: {:.4f}".format(entropy * sum_of_postive_eigenvalues))

    return


def build_matrix(x, y, args):
    """
    Build kernel matrix shown in the paper,
    which has the same positive eigenvalues of conditional kernel covariance matrix
    """

    kxx = normalized_gaussian_kernel(x, x, args.sigma, args.batchsize)
    kyy = normalized_gaussian_kernel(y, y, args.sigma, args.batchsize)
    kxy = normalized_gaussian_kernel(x, y, args.sigma, args.batchsize)
    kyx = kxy.T

    matrix_first_row = torch.hstack([kxx, kxy * np.sqrt(args.eta)])
    matrix_second_row = torch.hstack([-kyx * np.sqrt(args.eta), -kyy * args.eta])
    matrix = torch.vstack([matrix_first_row, matrix_second_row])

    return matrix


def build_matrix_cholesky(x, y, args):
    """
    Build kernel matrix for cholesky method shown in the paper,
    which has the same positive eigenvalues of conditional kernel covariance matrix
    """

    kxx = normalized_gaussian_kernel(x, x, args.sigma, args.batchsize)
    kyy = normalized_gaussian_kernel(y, y, args.sigma, args.batchsize)
    kxy = normalized_gaussian_kernel(x, y, args.sigma, args.batchsize)
    kyx = kxy.T

    matrix_first_row = torch.hstack([kxx, kxy * np.sqrt(args.eta)])
    matrix_second_row = torch.hstack([kyx * np.sqrt(args.eta), kyy * args.eta])
    matrix = torch.vstack([matrix_first_row, matrix_second_row])

    return matrix


def visualize_mode_by_eigenvectors_in_both_sets(
    x,
    y,
    test_dataset,
    test_idxs,
    ref_dataset,
    ref_idxs,
    args,
    absolute=True,
    print_KEN=True,
):
    """
    Retrieve the most similar samples to the top novel mode in both test set and ref set
    This function could be used when user is interested on mode-similar samples in reference set as well
    If the user wish to retrieve samples from the test dataset only, please use "visualize_mode_by_eigenvectors"

    IMPORTANT:
        argument "absolute" need to be set "True" if you wish to retrieve mode-similar samples in reference set
        Otherwise, it will retrieve the least similar samples in reference set if "absolute=False"
    """

    fused_kernel_matrix = build_matrix(x, y, args)
    test_idxs = test_idxs.to(dtype=torch.long)
    ref_idxs = ref_idxs.to(dtype=torch.long)

    eigenvalues, eigenvectors = torch.linalg.eig(fused_kernel_matrix)
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real

    if print_KEN:
        print_novelty_metrics(eigenvalues, args)

    _, max_id = eigenvalues.topk(args.num_text_mode)

    now_time = args.current_time

    for i in range(args.num_text_mode):

        top_eigenvector = eigenvectors[:, max_id[i]]
        if top_eigenvector[: len(x)].sum() < 0:
            top_eigenvector = -top_eigenvector

        if absolute:
            top_eigenvector = top_eigenvector.abs()

        top_text_ids = top_eigenvector.sort(descending=True)[1]
        save_folder_name = os.path.join(
            args.path_save_text,
            "backbone_{}/{}_{}/".format(args.backbone, args.text_name, now_time),
            "top{}".format(i + 1),
        )
        os.makedirs(save_folder_name)
        summary_test = []
        summary_ref = []

        cnt_saved_txt_test = 0
        cnt_saved_txt_ref = 0

        for _, top_text_id in enumerate(top_text_ids):

            if (
                top_text_id >= args.num_samples
                and cnt_saved_txt_ref < args.num_txt_per_mode
            ):
                idx = ref_idxs[top_text_id - len(x)]
                top_txt = ref_dataset[idx][0]
                summary_ref.append(top_txt + "\n")
                cnt_saved_txt_ref += 1

            elif (
                top_text_id < args.num_samples
                and cnt_saved_txt_test < args.num_txt_per_mode
            ):
                idx = test_idxs[top_text_id]
                top_txt = test_dataset[idx][0]
                summary_test.append(top_txt + "\n")
                cnt_saved_txt_test += 1

            if (
                cnt_saved_txt_test >= args.num_txt_per_mode
                and cnt_saved_txt_ref >= args.num_txt_per_mode
            ):
                break

        with open(os.path.join(save_folder_name, "summary_test.txt"), "w") as file:
            file.writelines(summary_test)
        with open(os.path.join(save_folder_name, "summary_ref.txt"), "w") as file:
            file.writelines(summary_ref)


def visualize_mode_by_eigenvectors(x, y, dataset, idxs, args, print_KEN=True):

    fused_kernel_matrix = build_matrix(x, y, args)
    idxs = idxs.to(dtype=torch.long)

    eigenvalues, eigenvectors = torch.linalg.eig(fused_kernel_matrix)
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real

    if print_KEN:
        print_novelty_metrics(eigenvalues, args)

    _, max_id = eigenvalues.topk(args.num_text_mode)

    now_time = args.current_time

    for i in range(args.num_text_mode):

        top_eigenvector = eigenvectors[:, max_id[i]]
        if top_eigenvector[: len(x)].sum() < 0:
            top_eigenvector = -top_eigenvector

        top_text_ids = top_eigenvector.sort(descending=True)[1]
        save_folder_name = os.path.join(
            args.path_save_text,
            "backbone_{}/{}_{}/".format(args.backbone, args.text_name, now_time),
            "top{}".format(i + 1),
        )
        os.makedirs(save_folder_name)
        summary = []
        cnt_saved_txt = 0

        for _, top_text_id in enumerate(top_text_ids):

            if top_text_id >= args.num_samples:
                continue
            else:
                idx = idxs[top_text_id]
                top_txt = dataset[idx][0]
                cnt_saved_txt += 1

            summary.append(top_txt + "\n")

            if cnt_saved_txt >= args.num_txt_per_mode:
                break

        with open(os.path.join(save_folder_name, "summary.txt"), "w") as file:
            file.writelines(summary)


def KEN_by_cholesky_decomposition(x, y, args):
    """
    Accelerating KEN score computation by cholesky decomposition
    It is observed that cholesky method will be a few times faster than original eigen-decomposition
    However, cholesky decomposition requires positive-definiteness (PD), but kernel similarity matrix is PSD
    The results in the paper used the slower eigen-decomposition

    IMPORTANT:
    If you encounter zero eigenvalues, you may:
        1. tune the parameter sigma,
        2. or add a very small eps*Identity to ensure PD, however, this will lead to slightly different results to eigen-decomposition
        3. or switch to slower eigen-decomposition

    """
    args.logger.info("Use cholesky acceleration.")
    eps = 1e-7

    with torch.no_grad():
        kernel_matrix = build_matrix_cholesky(x, y, args) + eps * torch.eye(
            x.shape[0] + y.shape[0], device=x.device
        )

        U = torch.linalg.cholesky(kernel_matrix, upper=True)
        diagonal = torch.cat(
            [
                torch.ones(x.shape[0], device=x.device),
                -1 * torch.ones(y.shape[0], device=y.device),
            ]
        )
        d_matrix = torch.diag(diagonal)
        assert len(diagonal) == x.shape[0] + y.shape[0]

        matrix_to_be_decomposed = U @ d_matrix @ U.T
        eigenvalues = eigvalsh(matrix_to_be_decomposed)

        print_novelty_metrics(eigenvalues, args)

    return


def KEN_by_eigendecomposition(x, y, args):
    fused_matrix = build_matrix(x, y, args)
    args.logger.info("Use matrix eigen-decomposition.")
    eigenvalues = torch.linalg.eigvals(fused_matrix)
    eigenvalues = eigenvalues.real
    print_novelty_metrics(eigenvalues, args)

    return
