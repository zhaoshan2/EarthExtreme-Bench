import numpy as np
import torch
from typing import Tuple


def unlog_tp(x, eps=1e-5):
    #    return np.exp(x + np.log(eps)) - eps
    return eps * (np.exp(x) - 1)


def unlog_tp_torch(x, eps=1e-5):
    #    return torch.exp(x + torch.log(eps)) - eps
    return eps * (torch.exp(x) - 1)


def mean(x, axis=None):
    # spatial mean
    y = np.sum(x, axis) / np.size(x, axis)
    return y


def lat_np(j, num_lat):
    return 90 - j * 180 / (num_lat - 1)


def weighted_acc(pred, target, weighted=True):
    # takes in shape [1, num_lat, num_long]
    if len(pred.shape) == 2:
        pred = np.expand_dims(pred, 0)
    if len(target.shape) == 2:
        target = np.expand_dims(target, 0)

    num_lat = np.shape(pred)[1]
    num_long = np.shape(target)[2]
    #    pred -= mean(pred)
    #    target -= mean(target)
    s = np.sum(np.cos(np.pi / 180 * lat_np(np.arange(0, num_lat), num_lat)))
    weight = (
        np.expand_dims(latitude_weighting_factor(np.arange(0, num_lat), num_lat, s), -1)
        if weighted
        else 1
    )
    r = (weight * pred * target).sum() / np.sqrt(
        (weight * pred * pred).sum() * (weight * target * target).sum()
    )
    return r


def weighted_acc_masked(pred, target, weighted=True, maskarray=1):
    # takes in shape [1, num_lat, num_long]
    if len(pred.shape) == 2:
        pred = np.expand_dims(pred, 0)
    if len(target.shape) == 2:
        target = np.expand_dims(target, 0)

    num_lat = np.shape(pred)[1]
    num_long = np.shape(target)[2]
    pred -= mean(pred)
    target -= mean(target)
    s = np.sum(np.cos(np.pi / 180 * lat(np.arange(0, num_lat), num_lat)))
    weight = (
        np.expand_dims(latitude_weighting_factor(np.arange(0, num_lat), num_lat, s), -1)
        if weighted
        else 1
    )
    r = (maskarray * weight * pred * target).sum() / np.sqrt(
        (maskarray * weight * pred * pred).sum()
        * (maskarray * weight * target * target).sum()
    )
    return r


def weighted_rmse(pred, target):
    if len(pred.shape) == 2:
        pred = np.expand_dims(pred, 0)
    if len(target.shape) == 2:
        target = np.expand_dims(target, 0)
    # takes in arrays of size [1, h, w]  and returns latitude-weighted rmse
    num_lat = np.shape(pred)[1]
    num_long = np.shape(target)[2]
    s = np.sum(np.cos(np.pi / 180 * lat_np(np.arange(0, num_lat), num_lat)))
    weight = np.expand_dims(
        latitude_weighting_factor(np.arange(0, num_lat), num_lat, s), -1
    )
    return np.sqrt(
        1
        / num_lat
        * 1
        / num_long
        * np.sum(np.dot(weight.T, (pred[0] - target[0]) ** 2))
    )


def latitude_weighting_factor(j, num_lat, s):
    return num_lat * np.cos(np.pi / 180.0 * lat_np(j, num_lat)) / s


def top_quantiles_error(pred, target):
    if len(pred.shape) == 2:
        pred = np.expand_dims(pred, 0)
    if len(target.shape) == 2:
        target = np.expand_dims(target, 0)
    qs = 100
    qlim = 5
    qcut = 0.1
    qtile = 1.0 - np.logspace(-qlim, -qcut, num=qs)
    P_tar = np.quantile(target, q=qtile, axis=(1, 2))
    P_pred = np.quantile(pred, q=qtile, axis=(1, 2))
    return np.mean(P_pred - P_tar, axis=0)


# torch version for rmse comp
@torch.jit.script
def lat(j: torch.Tensor, num_lat: int) -> torch.Tensor:
    return 90.0 - j * 180.0 / float(num_lat - 1)


@torch.jit.script
def latitude_weighting_factor_torch(
    j: torch.Tensor, num_lat: int, s: torch.Tensor
) -> torch.Tensor:
    return num_lat * torch.cos(3.1416 / 180.0 * lat(j, num_lat)) / s


@torch.jit.script
def weighted_rmse_torch_channels(
    pred: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    # takes in arrays of size [n, c, h, w]  and returns latitude-weighted rmse for each channel
    num_lat = pred.shape[-2]
    # num_long = target.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)

    s = torch.sum(torch.cos(3.1416 / 180.0 * lat(lat_t, num_lat)))

    if pred.dim() == 3:
        weight = torch.reshape(
            latitude_weighting_factor_torch(lat_t, num_lat, s), (1, -1, 1)
        )
    else:
        weight = torch.reshape(
            latitude_weighting_factor_torch(lat_t, num_lat, s), (1, 1, -1, 1)
        )
    result = torch.sqrt(torch.mean(weight * (pred - target) ** 2.0, dim=(-1, -2)))
    return result


@torch.jit.script
def weighted_rmse_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    result = weighted_rmse_torch_channels(pred, target)
    return torch.mean(result, dim=0)


@torch.jit.script
def weighted_acc_masked_torch_channels(
    pred: torch.Tensor, target: torch.Tensor, maskarray: torch.Tensor
) -> torch.Tensor:
    # takes in arrays of size [n, c, h, w]  and returns latitude-weighted acc
    num_lat = pred.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)
    s = torch.sum(torch.cos(3.1416 / 180.0 * lat(lat_t, num_lat)))
    weight = torch.reshape(
        latitude_weighting_factor_torch(lat_t, num_lat, s), (1, 1, -1, 1)
    )
    result = torch.sum(maskarray * weight * pred * target, dim=(-1, -2)) / torch.sqrt(
        torch.sum(maskarray * weight * pred * pred, dim=(-1, -2))
        * torch.sum(maskarray * weight * target * target, dim=(-1, -2))
    )
    return result


@torch.jit.script
def weighted_acc_torch_channels(
    pred: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    # takes in arrays of size [n, c, h, w]  and returns latitude-weighted acc
    num_lat = pred.shape[-2]
    # num_long = target.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)
    s = torch.sum(torch.cos(3.1416 / 180.0 * lat(lat_t, num_lat)))
    if pred.dim() == 3:
        weight = torch.reshape(
            latitude_weighting_factor_torch(lat_t, num_lat, s), (1, -1, 1)
        )
    else:
        weight = torch.reshape(
            latitude_weighting_factor_torch(lat_t, num_lat, s), (1, 1, -1, 1)
        )
    result = torch.sum(weight * pred * target, dim=(-1, -2)) / torch.sqrt(
        torch.sum(weight * pred * pred, dim=(-1, -2))
        * torch.sum(weight * target * target, dim=(-1, -2))
    )
    return result


@torch.jit.script
def weighted_acc_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    result = weighted_acc_torch_channels(pred, target)
    return torch.mean(result, dim=0)


@torch.jit.script
def unweighted_acc_torch_channels(
    pred: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    result = torch.sum(pred * target, dim=(-1, -2)) / torch.sqrt(
        torch.sum(pred * pred, dim=(-1, -2)) * torch.sum(target * target, dim=(-1, -2))
    )
    return result


@torch.jit.script
def unweighted_acc_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    result = unweighted_acc_torch_channels(pred, target)
    return torch.mean(result, dim=0)


@torch.jit.script
def unweighted_rmse_torch_channels(
    pred: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    # takes in arrays of size [n, c, h, w]  and returns rmse for each channel
    result = torch.sqrt(torch.mean((pred - target) ** 2.0, dim=(-1, -2)))
    return result


@torch.jit.script
def unweighted_rmse_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    result = unweighted_rmse_torch_channels(pred, target)
    return torch.mean(result, dim=0)


@torch.jit.script
def top_quantiles_error_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    qs = 100
    qlim = 3
    qcut = 0.1
    n, c, h, w = pred.size()
    qtile = 1.0 - torch.logspace(-qlim, -qcut, steps=qs, device=pred.device)
    P_tar = torch.quantile(target.view(n, c, h * w), q=qtile, dim=-1)
    P_pred = torch.quantile(pred.view(n, c, h * w), q=qtile, dim=-1)
    return torch.mean(P_pred - P_tar, dim=0)


def pixel_to_rainfall(img, a=None, b=None):
    """Convert the pixel values to real rainfall intensity

    Parameters
    ----------
    img : np.ndarray
    a : float32, optional
    b : float32, optional

    Returns
    -------
    rainfall_intensity : np.ndarray
    """
    if a is None:
        a = 58.53
    if b is None:
        b = 1.56
    dBZ = img * 70.0 - 10.0
    dBR = (dBZ - 10.0 * np.log10(a)) / b
    rainfall_intensity = np.power(10, dBR / 10.0)
    return rainfall_intensity


def rainfall_to_pixel(rainfall_intensity, a=None, b=None):
    """Convert the rainfall intensity to pixel values

    Parameters
    ----------
    rainfall_intensity : np.ndarray
    a : float32, optional
    b : float32, optional

    Returns
    -------
    pixel_vals : np.ndarray
    """
    if a is None:
        a = 58.53
    if b is None:
        b = 1.56
    dBR = np.log10(rainfall_intensity) * 10.0
    dBZ = dBR * b + 10.0 * np.log10(a)
    pixel_vals = (dBZ + 10.0) / 70.0
    return pixel_vals


def uint2single(x):
    """
    uint8 to float32 [0,255] to [0,1]
    """
    return np.float32(x / 255.0)


def single2uint(x):
    """
    float32 to uint8 [0,1] to [0,255]
    """
    return np.uint8((x.clip(0, 1) * 255.0).round())


def tensor2uint(x):
    x = x.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    return np.uint8((x * 255.0).round())


try:
    import cPickle as pickle
except:
    import pickle

from numba import boolean, float32, float64, int32, int64, jit, njit


@njit(
    float32[:, :](float32[:, :, :, :, :], float32[:, :, :, :, :], int32[:, :, :, :, :])
)
def get_GDL_numba(prediction, truth, mask):
    """Accelerated version of get_GDL using numba(http://numba.pydata.org/)

    Parameters
    ----------
    prediction
    truth
    mask

    Returns
    -------
    gdl
    """
    seqlen, batch_size, _, height, width = prediction.shape
    gdl = np.zeros(shape=(seqlen, batch_size), dtype=np.float32)
    for i in range(seqlen):
        for j in range(batch_size):
            for m in range(height):
                for n in range(width):
                    if m + 1 < height:
                        if mask[i][j][0][m + 1][n] and mask[i][j][0][m][n]:
                            pred_diff_h = abs(
                                prediction[i][j][0][m + 1][n]
                                - prediction[i][j][0][m][n]
                            )
                            gt_diff_h = abs(
                                truth[i][j][0][m + 1][n] - truth[i][j][0][m][n]
                            )
                            gdl[i][j] += abs(pred_diff_h - gt_diff_h)
                    if n + 1 < width:
                        if mask[i][j][0][m][n + 1] and mask[i][j][0][m][n]:
                            pred_diff_w = abs(
                                prediction[i][j][0][m][n + 1]
                                - prediction[i][j][0][m][n]
                            )
                            gt_diff_w = abs(
                                truth[i][j][0][m][n + 1] - truth[i][j][0][m][n]
                            )
                            gdl[i][j] += abs(pred_diff_w - gt_diff_w)
    return gdl


def get_hit_miss_counts_numba(prediction, truth, mask, thresholds=None):
    """This function calculates the overall hits and misses for the prediction, which could be used
    to get the skill scores and threat scores:


    This function assumes the input, i.e, prediction and truth are 3-dim tensors, (timestep, row, col)

    Parameters
    ----------
    prediction : np.ndarray
        Shape: (seq_len, batch_size, 1, height, width)
    truth : np.ndarray
        Shape: (seq_len, batch_size, 1, height, width)
    mask : np.ndarray or None
        Shape: (seq_len, batch_size, 1, height, width)
        0 --> not use
        1 --> use
    thresholds : list or tuple

    Returns
    -------
    hits : np.ndarray
        (seq_len, batch_size, len(thresholds))
        TP
    misses : np.ndarray
        (seq_len, batch_size, len(thresholds))
        FN
    false_alarms : np.ndarray
        (seq_len, batch_size, len(thresholds))
        FP
    correct_negatives : np.ndarray
        (seq_len, batch_size, len(thresholds))
        TN
    """
    assert 5 == prediction.ndim
    assert 5 == truth.ndim
    assert prediction.shape == truth.shape
    assert prediction.shape[2] == 1
    thresholds = [thresholds[i] for i in range(len(thresholds))]
    thresholds = sorted(thresholds)
    ret = _get_hit_miss_counts_numba(
        prediction=prediction,
        truth=truth,
        mask=mask,
        thresholds=np.array(thresholds).astype(np.float32),
    )
    return ret[:, :, :, 0], ret[:, :, :, 1], ret[:, :, :, 2], ret[:, :, :, 3]


@njit(
    int32[:, :, :, :](
        float32[:, :, :, :, :], float32[:, :, :, :, :], int32[:, :, :, :, :], float32[:]
    )
)
def _get_hit_miss_counts_numba(prediction, truth, mask, thresholds):
    seqlen, batch_size, _, height, width = prediction.shape
    threshold_num = len(thresholds)
    ret = np.zeros(shape=(seqlen, batch_size, threshold_num, 4), dtype=np.int32)

    for i in range(seqlen):
        for j in range(batch_size):
            for m in range(height):
                for n in range(width):
                    if mask[i][j][0][m][n]:
                        for k in range(threshold_num):
                            bpred = prediction[i][j][0][m][n] >= thresholds[k]
                            btruth = truth[i][j][0][m][n] >= thresholds[k]
                            ind = (1 - btruth) * 2 + (1 - bpred)
                            ret[i][j][k][ind] += 1
    return ret


class RadarEvaluation(object):
    def __init__(self, seq_len, no_ssim=True, thresholds=None):
        print("thresholds: ", thresholds)
        if thresholds is None:
            self._thresholds = [0.5, 2.0, 5.0, 10.0, 30.0]
        else:
            self._thresholds = [i for i in thresholds]
        self._seq_len = seq_len
        self._no_ssim = no_ssim
        # self._exclude_mask = get_exclude_mask()
        self.begin()

    def begin(self):
        self._total_hits = np.zeros(
            (self._seq_len, len(self._thresholds)), dtype=np.int32
        )
        self._total_misses = np.zeros(
            (self._seq_len, len(self._thresholds)), dtype=np.int32
        )
        self._total_false_alarms = np.zeros(
            (self._seq_len, len(self._thresholds)), dtype=np.int32
        )
        self._total_correct_negatives = np.zeros(
            (self._seq_len, len(self._thresholds)), dtype=np.int32
        )
        self._mse = np.zeros((self._seq_len,), dtype=np.float32)
        self._mae = np.zeros((self._seq_len,), dtype=np.float32)
        self._gdl = np.zeros((self._seq_len,), dtype=np.float32)
        self._ssim = np.zeros((self._seq_len,), dtype=np.float32)
        self._datetime_dict = {}
        self._total_batch_num = 0

    def clear_all(self):
        self._total_hits[:] = 0
        self._total_misses[:] = 0
        self._total_false_alarms[:] = 0
        self._total_correct_negatives[:] = 0
        self._mse[:] = 0
        self._mae[:] = 0
        self._gdl[:] = 0
        self._ssim[:] = 0
        self._total_batch_num = 0

    def update(self, gt, pred, mask=None, start_datetimes=None):
        """

        Parameters
        ----------
        gt : np.ndarray
        pred : np.ndarray
        mask : np.ndarray
            0 indicates not use and 1 indicates that the location will be taken into account
        start_datetimes : list
            The starting datetimes of all the testing instances

        Returns
        -------

        """
        if start_datetimes is not None:
            batch_size = len(start_datetimes)
            assert gt.shape[1] == batch_size
        else:
            batch_size = gt.shape[1]
        assert gt.shape[0] == self._seq_len
        assert gt.shape == pred.shape
        if mask is None:
            mask = np.ones(gt.shape, dtype=np.int32)
        else:
            mask = mask.astype(np.int32)
        assert gt.shape == mask.shape
        self._total_batch_num += batch_size
        # (l, b, c, h, w)
        mse = (mask * np.square(pred - gt)).sum(axis=(2, 3, 4))
        mae = (mask * np.abs(pred - gt)).sum(axis=(2, 3, 4))
        gdl = get_GDL_numba(prediction=pred, truth=gt, mask=mask)
        self._mse += mse.sum(axis=1)
        self._mae += mae.sum(axis=1)
        self._gdl += gdl.sum(axis=1)
        if not self._no_ssim:
            raise NotImplementedError
            # self._ssim += get_SSIM(prediction=pred, truth=gt)
        hits, misses, false_alarms, correct_negatives = get_hit_miss_counts_numba(
            prediction=pred, truth=gt, mask=mask, thresholds=self._thresholds
        )
        self._total_hits += hits.sum(axis=1)
        self._total_misses += misses.sum(axis=1)
        self._total_false_alarms += false_alarms.sum(axis=1)
        self._total_correct_negatives += correct_negatives.sum(axis=1)
        return mse, mae

    def calculate_stat(self):
        """The following measurements will be used to measure the score of the forecaster

        See Also
        [Weather and Forecasting 2010] Equitability Revisited: Why the "Equitable Threat Score" Is Not Equitable
        http://www.wxonline.info/topics/verif2.html

        We will denote
        (a b    (hits       false alarms
         c d) =  misses   correct negatives)

        We will report the
        POD = a / (a + c)
        FAR = b / (a + b)
        CSI = a / (a + b + c)
        Heidke Skill Score (HSS) = 2(ad - bc) / ((a+c) (c+d) + (a+b)(b+d))
        Gilbert Skill Score (GSS) = HSS / (2 - HSS), also known as the Equitable Threat Score
            HSS = 2 * GSS / (GSS + 1)
        MSE = mask * (pred - gt) **2
        MAE = mask * abs(pred - gt)
        GDL = valid_mask_h * abs(gd_h(pred) - gd_h(gt)) + valid_mask_w * abs(gd_w(pred) - gd_w(gt))
        Returns
        -------

        """
        a = self._total_hits.astype(np.float64)
        b = self._total_false_alarms.astype(np.float64)
        c = self._total_misses.astype(np.float64)
        d = self._total_correct_negatives.astype(np.float64)
        pod = a / (a + c)
        far = b / (a + b)
        csi = a / (a + b + c)
        n = a + b + c + d
        aref = (a + b) / n * (a + c)
        gss = (a - aref) / (a + b + c - aref)
        hss = 2 * gss / (gss + 1)
        mse = self._mse / self._total_batch_num
        mae = self._mae / self._total_batch_num
        gdl = self._gdl / self._total_batch_num
        if not self._no_ssim:
            raise NotImplementedError
            # ssim = self._ssim / self._total_batch_num
        return pod, far, csi, hss, gss, mse, mae, gdl


@torch.jit.script
def top_quantiles_error_torch(
    pred: torch.Tensor, target: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    qs = 10
    qlim = 2
    qcut = 1
    n, c, h, w = pred.size()
    qtile = 1.0 - 2 * torch.logspace(
        -qlim, -qcut, steps=qs, device=pred.device, dtype=target.dtype
    )
    P_tar = torch.quantile(target.view(n, c, h * w), q=qtile, dim=-1)
    qtile = 1.0 - 2 * torch.logspace(
        -qlim, -qcut, steps=qs, device=pred.device, dtype=pred.dtype
    )
    P_pred = torch.quantile(pred.view(n, c, h * w), q=qtile, dim=-1)

    return (qtile, torch.mean((P_pred - P_tar) / P_tar, dim=1))


def lower_quantiles_error_torch(
    pred: torch.Tensor, target: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    qs = 10
    qlim = 2
    qcut = 1
    n, c, h, w = pred.size()

    # Generate quantiles for the lower tail
    qtile = 2 * torch.logspace(
        -qlim, -qcut, steps=qs, device=pred.device, dtype=target.dtype
    )

    # Compute quantiles for the target and predictions
    P_tar = torch.quantile(target.view(n, c, h * w), q=qtile, dim=-1)
    qtile = 2 * torch.logspace(
        -qlim, -qcut, steps=qs, device=pred.device, dtype=pred.dtype
    )
    P_pred = torch.quantile(pred.view(n, c, h * w), q=qtile, dim=-1)
    # Compute the relative error (qs, n, 1)
    return (qtile, torch.mean((P_pred - P_tar) / P_tar, dim=1))


def TQE(pred_real, gt_real):
    scores = top_quantiles_error_torch(pred_real, gt_real)
    q = [round(num * 100, 2) for num in scores[0].tolist()]
    s = [round(num[0], 4) for num in scores[1].tolist()]
    return q, s


def LQE(pred_real, gt_real):
    scores = lower_quantiles_error_torch(pred_real, gt_real)
    q = [round(num * 100, 2) for num in scores[0].tolist()]
    s = [round(num[0], 4) for num in scores[1].tolist()]
    return q, s
