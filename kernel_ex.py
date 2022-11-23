import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.gaussian_process.kernels import ExpSineSquared
from true_funcs import polyn, peri, comb
import matplotlib.pyplot as plt
import params


def sample_funcs(num=10):
    np.random.seed(3)
    X_sample = (np.append(np.random.uniform(0, 5, num), 5)).reshape((-1, 1))
    y_poly_sample = polyn(X_sample)
    y_peri_sample = peri(X_sample)
    y_comb_sample = comb(X_sample)
    return X_sample, y_poly_sample, y_peri_sample, y_comb_sample


def plot_true_funcs():
    X = np.linspace(0, 5, 1000)
    y_poly = polyn(X)
    y_peri = peri(X)
    y_comb = comb(X)

    plt.rcParams.update({"font.family": "Helvetica"})

    plot_dat(X, y_poly, "poly_true", y_min=0, y_max=6)
    plot_dat(X, y_peri, "peri_true", y_min=0.4, y_max=1.6)
    plot_dat(X, y_comb, "comb_true", y_min=-0.5, y_max=8)


def approx_funcs():
    X = np.linspace(0, 5, 1000).reshape((-1, 1))
    X_sample, y_poly, y_peri, y_comb = sample_funcs()

    # fit data with polynomial kernel of degree 3
    kpoly_poly = KernelRidge(alpha=0.1, kernel="polynomial", degree=3)
    kpoly_poly.fit(X_sample, y_poly)
    preds_kpoly_poly = kpoly_poly.predict(X)
    kpoly_peri = KernelRidge(alpha=0.1, kernel="polynomial", degree=3)
    kpoly_peri.fit(X_sample, y_peri)
    preds_kpoly_peri = kpoly_peri.predict(X)
    kpoly_comb = KernelRidge(alpha=0.1, kernel="polynomial", degree=3)
    kpoly_comb.fit(X_sample, y_comb)
    preds_kpoly_comb = kpoly_comb.predict(X)

    # fit data with periodic kernel
    kernel = ExpSineSquared(periodicity=np.pi / 2)
    kperi_poly = KernelRidge(alpha=0.1, kernel=kernel)
    kperi_poly.fit(X_sample, y_poly)
    preds_kperi_poly = kperi_poly.predict(X)
    kperi_peri = KernelRidge(alpha=0.1, kernel=kernel)
    kperi_peri.fit(X_sample, y_peri)
    preds_kperi_peri = kperi_peri.predict(X)
    kperi_comb = KernelRidge(alpha=0.1, kernel=kernel)
    kperi_comb.fit(X_sample, y_comb)
    preds_kperi_comb = kperi_comb.predict(X)

    # fit data with squared exponential kernel
    kse_poly = KernelRidge(alpha=0.1, kernel="rbf")
    kse_poly.fit(X_sample, y_poly)
    preds_kse_poly = kse_poly.predict(X)
    kse_peri = KernelRidge(alpha=0.1, kernel="rbf")
    kse_peri.fit(X_sample, y_peri)
    preds_kse_peri = kse_peri.predict(X)
    kse_comb = KernelRidge(alpha=0.1, kernel="rbf")
    kse_comb.fit(X_sample, y_comb)
    preds_kse_comb = kse_comb.predict(X)

    plt.rcParams.update({"font.family": "Helvetica"})

    # plot data with polynomial kernel
    plot_dat(X, preds_kpoly_poly, "kpoly_poly_pred", X_sample, y_poly, y_min=0, y_max=6)
    plot_dat(
        X, preds_kpoly_peri, "kpoly_peri_pred", X_sample, y_peri, y_min=0.4, y_max=1.6
    )
    plot_dat(
        X, preds_kpoly_comb, "kpoly_comb_pred", X_sample, y_comb, y_min=-0.5, y_max=8
    )

    # plot data with periodic kernel
    plot_dat(X, preds_kperi_poly, "kperi_poly_pred", X_sample, y_poly, y_min=0, y_max=6)
    plot_dat(
        X, preds_kperi_peri, "kperi_peri_pred", X_sample, y_peri, y_min=0.4, y_max=1.6
    )
    plot_dat(
        X, preds_kperi_comb, "kperi_comb_pred", X_sample, y_comb, y_min=-0.5, y_max=8
    )

    # plot data with squared exponential kernel
    plot_dat(X, preds_kse_poly, "kse_poly_pred", X_sample, y_poly, y_min=0, y_max=6)
    plot_dat(X, preds_kse_peri, "kse_peri_pred", X_sample, y_peri, y_min=0.4, y_max=1.6)
    plot_dat(X, preds_kse_comb, "kse_comb_pred", X_sample, y_comb, y_min=-0.5, y_max=8)


def plot_dat(
    X, y, name, X_samples=None, y_samples=None, _type=".png", y_min=0, y_max=1
):
    font_size = 10
    fig = plt.figure()
    fig.set_size_inches(10, 6)
    # Uncomment if labels should be added to the plot
    # plt.xlabel("$x$", fontsize=font_size)
    # plt.ylabel("$f(x)$", fontsize=font_size)
    # Comment is ticks should be added to the plot
    plt.xticks([], [])
    plt.yticks([], [])
    plt.plot(X, y, color=params.colors["red"], lw=2)
    if X_samples is not None:
        plt.plot(X_samples, y_samples, "o", color=params.colors["grey"], ms=8)
    plt.xlim(0, 5)
    plt.ylim(y_min, y_max)
    fig.tight_layout()
    plt.show()
    fig.savefig("plots//" + name + _type, bbox_inches="tight")


if __name__ == "__main__":
    plot_true_funcs()
    approx_funcs()
