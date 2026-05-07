import numpy as np
from scipy.stats import Normal
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from typing import Tuple, Generator, List


def read_norm(cover_path: str, stego_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Open npy files, compute the norm, ensure equal number of samples for cover and stego and return two arrays of norms,
    one for each class.
    :param cover_path: str, the path to a npy file
    :param stego_path:  str, the path to a npy file
    :return: X the norms
    """
    cover = np.linalg.norm(np.load(cover_path), axis=-1)
    stego = np.linalg.norm(np.load(stego_path), axis=-1)

    n = np.min([cover.shape[0], stego.shape[0]])

    return cover[:n], stego[:n]


def kfold_generator(cover: np.ndarray,
                    stego: np.ndarray,
                    n_splits: int = 5,
                    seed: int = 123) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None, None]:
    """
    Generate train and test folds, one fold is use for the training, the remaining folds are used for the testing.
    :param cover: array of cover norms.
    :param stego: array of stego norms.
    :param n_splits: number of splits
    :param seed: random seed
    :return: cover_train, cover_test, stego_train, stego_test
    """
    rng = np.random.default_rng(seed)

    assert cover.shape[0] == stego.shape[0]

    n = cover.shape[0]

    cover_idx = rng.permutation(n)
    stego_idx = rng.permutation(n)

    folds = np.array_split(np.arange(n), n_splits)
    for train_indices in folds:
        test_indices = np.setdiff1d(np.arange(n), train_indices)
        cover_train = cover[cover_idx[train_indices]]
        cover_test = cover[cover_idx[test_indices]]
        stego_train = stego[stego_idx[train_indices]]
        stego_test = stego[stego_idx[test_indices]]
        yield cover_train, cover_test, stego_train, stego_test


def batch_generator(data: np.ndarray, batch_size: int) -> Generator[np.ndarray, None, None]:
    """
    Generate batch from an array of data. If the lenght of data is not divisible by the size of a batch, the remaining
    elements are ignored.
    :param data: an array.
    :param batch_size: size of a batch.
    :return: a batch of data in the sequential order.
    """
    for i in range(0, len(data) - batch_size + 1, batch_size):
        yield data[i:i + batch_size]


def compute_log_lr(data: np.ndarray, cover_model: Normal, stego_model: Normal) -> float:
    """
    Compute the log-likelihood ratio given some samples in data and two Normal distributions, one for the cover model
    and one for the stego model.
    :param data: an array of observations.
    :param cover_model: a scipy distribution with the .logpdf() method.
    :param stego_model: a scipy distribution with the .logpdf() method.
    :return: float, the log likelihood ratio of all samples.
    """
    return float(np.sum(cover_model.logpdf(data) - stego_model.logpdf(data)))


def approximate_distributions(cover: np.ndarray, stego: np.ndarray) -> Tuple[Normal, Normal]:
    """
    Approximate mu the mean of the norm for both cover and stego, and sigma_0 (respec. sigma_1) the standard
    deviation of the cover (respec.stego) class.
    :param cover: samples of norm for cover images.
    :param stego: samples of norm for stego images.
    :return: two Normal distribution with parameters mu and sigma_0 for the cover model and mu and sigma_1 for the
    stego model.
    """
    mu = np.mean(np.concatenate([cover, stego]))
    eps = 1e-12
    sigma_0 = np.std(cover) + eps
    sigma_1 = np.std(stego) + eps
    return Normal(mu=mu, sigma=sigma_0), Normal(mu=mu, sigma=sigma_1)


def predict_batchwise(data: np.ndarray,
                      batch_size: int,
                      cover_model: Normal,
                      stego_model: Normal,
                      tau: float = 0) -> np.ndarray:
    """
    Given an array of observations, a cover model and a stego model, this function returns predictions for each
    sequential batch of data with the decision threshold tau.
    :param data: an array of observations.
    :param batch_size: int, the size of a batch.
    :param cover_model: a scipy distribution with the .logpdf() method.
    :param stego_model: a scipy distribution with the .logpdf() method.
    :param tau: float, the decision threshold.
    :return: an array of size len(data) // batch_size with 0 for the cover and 1 for the stegos.
    """
    prediction = []
    for batch in batch_generator(data, batch_size):
        log_lrt_score = compute_log_lr(batch, cover_model, stego_model)
        prediction.append(log_lrt_score < tau)
    return np.array(prediction).astype(int)


def compute_pe(predictions: np.ndarray, label: np.ndarray) -> float:
    """
    Compute the probability of error for an array of predictions and its labels.
    :param predictions: an array of predictions.
    :param label: an array of labels.
    :return: the PE score as float.
    """
    fpr, tpr, thresholds = roc_curve(label, predictions)
    return float(np.min((fpr + 1 - tpr) / 2))


def run_batch_size_prediction(cover_train: np.ndarray,
                              cover_test: np.ndarray,
                              stego_train: np.ndarray,
                              stego_test: np.ndarray,
                              batch_size: np.ndarray,
                              tau: float) -> List:
    """
    Given a fold of train and test for each class, approximate the models and return the PE for a given decision
    threshold tau.
    :param cover_train: an array of cover norm for the training.
    :param cover_test: an array of cover norm for the testing.
    :param stego_train: an array of stego norm for the training.
    :param stego_test: an array of stego norm for the testing.
    :param batch_size: an array of multiples size for a batch.
    :param tau: float, the decision threshold.
    :return: an array of PE value for each different batch size.
    """
    cover_model, stego_model = approximate_distributions(cover_train, stego_train)

    pe = []
    for b_size in batch_size:
        cover_pred = predict_batchwise(cover_test, b_size, cover_model, stego_model, tau)
        cover_label = np.zeros(cover_pred.shape)
        stego_pred = predict_batchwise(stego_test, b_size, cover_model, stego_model, tau)
        stego_label = np.ones(stego_pred.shape)

        label = np.concatenate([cover_label, stego_label])
        prediction = np.concatenate([cover_pred, stego_pred])

        pe.append(compute_pe(prediction, label))

    return pe


def run_experiment(cover_path: str,
                   stego_path: str,
                   n_splits: int,
                   tau: float,
                   batch_size: np.ndarray,
                   seed: int = 123) -> np.ndarray:
    """
    Main loop for the experiment. Given the paths to npy files of cover and stego latent representations, compute the PE
    for different batch size and repeat the experiment for each fold. The prediction depends on the decision threshold
    tau.
    :param cover_path: str, the path to the npy cover file containing the latent representations.
    :param stego_path: str, the path to the npy stego file containing the latent representations.
    :param n_splits: int, number of folds.
    :param tau: float, the decision threshold.
    :param batch_size: an array of int, different batch sizes.
    :param seed: int, a random seed.
    :return: an array of PE values, one for each batch size averaged across each fold.
    """
    cover, stego = read_norm(cover_path, stego_path)
    kf_generator = kfold_generator(cover, stego, n_splits=n_splits, seed=seed)

    pe = np.zeros((n_splits, batch_size.size))
    for i, (cover_train, cover_test, stego_train, stego_test) in enumerate(kf_generator):
        pe[i] = run_batch_size_prediction(cover_train, cover_test, stego_train, stego_test, batch_size, tau)

    return np.mean(pe, axis=0)


if __name__ == "__main__":
    n_splits = 5
    tau = 0
    batch_size = np.unique(np.geomspace(1, 1000, 25).astype(int))

    paths = [("data/cover/prompt/guidance_5_steps_20_SS.npy", "data/stego/prompt/guidance_5_steps_20_SS.npy",
              "SS with prompt", "C0", "-"),
             ("data/cover/no_prompt/guidance_5_steps_20_SS.npy", "data/stego/no_prompt/guidance_5_steps_20_SS.npy",
              "SS without prompt", "C0", "--"),
             ("data/cover/prompt/guidance_5_steps_20_scaled_SS.npy",
              "data/stego/prompt/guidance_5_steps_20_scaled_SS.npy", "Scaled SS with prompt", "C1", "-"),
             ("data/cover/no_prompt/guidance_5_steps_20_scaled_SS.npy",
              "data/stego/no_prompt/guidance_5_steps_20_scaled_SS.npy", "Scaled SS without prompt", "C1", "--")
             ]

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=13)

    for cover_path, stego_path, label, color, pattern in paths:
        print(f"Running prediction for {label}...")
        pe = run_experiment(cover_path, stego_path, n_splits, tau, batch_size)
        print(np.max(pe))
        plt.plot(batch_size, pe, pattern, label=label, color=color)

    plt.xscale('log')
    plt.grid()
    plt.xlabel("Batch size")
    plt.ylabel(r"$P_E$")
    plt.title(r"$P_E$ depending on the batch size")
    plt.legend()
    plt.savefig("data/img/PE_plot.pdf", bbox_inches='tight')
    plt.show()
