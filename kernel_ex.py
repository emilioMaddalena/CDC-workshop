import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.gaussian_process.kernels import ExpSineSquared
from true_funcs import polyn, peri, comb, rkhs
import matplotlib.pyplot as plt
import params
from models import KRR, OptiBound


def sample_funcs(num=10):
    np.random.seed(3)
    X_sample = (np.append(np.random.uniform(0, 5, num), 5)).reshape((-1, 1))
    y_poly_sample = polyn(X_sample)
    y_peri_sample = peri(X_sample)
    y_comb_sample = comb(X_sample)
    return X_sample, y_poly_sample, y_peri_sample, y_comb_sample


def sample_rkhs_func(num=10, noise=0):
    np.random.seed(5)
    X_sample = (np.random.uniform(0, 7, num)).reshape((-1, 1))
    y_sample = rkhs(X_sample, lengthscale=3)
    for sample in y_sample:
        sample += np.random.uniform(-noise, noise)
    print(y_sample[:7])
    input("..")
    return X_sample, y_sample


def plot_rkhs_func(samples=True, num=20, num_plot=7, noise=0):
    X = np.linspace(0, 7, 500).reshape((-1, 1))
    y = rkhs(X, lengthscale=3)
    colors = []  # "red"
    if samples:
        X_samples, y_samples = sample_rkhs_func(num=num, noise=noise)
        plot_dat(
            X,
            [y],
            colors,
            "rkhs_true_samples",
            X_samples[:num_plot],  #
            y_samples[:num_plot],  #
            y_min=-3.5,
            y_max=5.5,
            x_min=0,
            x_max=7,
        )
    else:
        plot_dat(
            X,
            [y],
            colors=colors,
            name="rkhs_true_nosamples",
            y_min=-1.5,
            y_max=4.5,
            x_min=0,
            x_max=8,
        )


def train_plot_krr(samples=True, wtrue=True):
    num_sampl = 10
    X_samples, y_samples = sample_rkhs_func()
    model = KRR(1e-8, 1, 0, 6.4)
    model.fit(X_samples[:num_sampl], y_samples[:num_sampl])
    print(f"Norm KRR {model.get_norm()}")
    X = np.linspace(0, 8, 1000).reshape((-1, 1))
    y = model.predict(X)
    lb = []
    ub = []
    for x in X:
        lb.append(model.get_interp_lower(x.reshape((1, -1))))
        ub.append(model.get_interp_upper(x.reshape((1, -1))))
    lb = np.array(lb).flatten()
    print(lb)
    ub = np.array(ub).flatten()
    ys = [y]
    colors = ["red"]
    add_name = ""
    if wtrue:
        ytrue = rkhs(X)
        ys.append(ytrue)
        colors.append("green")
        add_name = "both"
    if samples:
        plot_dat(
            X,
            ys,
            colors,
            "krr_rkhs_samples" + add_name + str(num_sampl),
            X_samples[:num_sampl],
            y_samples[:num_sampl],
            y_min=-1.5,
            y_max=4.5,
            x_min=0,
            x_max=8,
        )
    else:
        plot_dat(
            X,
            ys,
            colors,
            "krr_rkhs_nosamples" + add_name + str(num_sampl),
            y_min=-1.5,
            y_max=4.5,
            x_min=0,
            x_max=8,
        )
        plot_bounds(
            X.flatten(),
            [lb],
            [ub],
            ["red"],
            "bounds_krr" + str(num_sampl),
            y_min=-1.5,
            y_max=4.5,
            x_min=0,
            x_max=8,
        )


def train_plot_single_noise(samples=True, wtrue=True):
    num_sampl = 7
    x_min = 0
    x_max = 7
    y_min = -3.5
    y_max = 5.5
    dbar = 0.4
    lengthscale = 3
    gamma = 227.2  # 39.4
    X_samples, y_samples = sample_rkhs_func(num=20, noise=dbar)
    opti_model = OptiBound(lengthscale, dbar, gamma)  # 6.4
    opti_model.fit(X_samples[:num_sampl], y_samples[:num_sampl])
    x_new = np.array([[5.5]])
    model2 = KRR(0.00001, lengthscale, dbar, gamma)
    lb = opti_model.get_lower_bound(x_new)
    model2.fit(X_samples[:num_sampl], opti_model.interp_val.reshape((-1, 1)))
    ub = opti_model.get_upper_bound(x_new)
    model1 = KRR(0.00001, lengthscale, dbar, gamma)  # 6.4
    print(opti_model.interp_val)
    model1.fit(X_samples[:num_sampl], opti_model.interp_val.reshape((-1, 1)))
    print(f"Norm KRR {model1.get_norm()}")
    X = np.linspace(0, x_max, 500).reshape((-1, 1))
    y = model1.predict(X)
    y2 = model2.predict(X)
    ytrue = rkhs(X, lengthscale=lengthscale)
    fig = plt.figure()
    fig.set_size_inches(10, 6)
    # Uncomment if labels should be added to the plot
    # plt.xlabel("$x$", fontsize=font_size)
    # plt.ylabel("$f(x)$", fontsize=font_size)
    # Comment is ticks should be added to the plot
    plt.xticks([], [])
    plt.yticks([], [])
    plt.plot(X, ytrue, "k--", lw=2)
    plt.plot(X, y, color=params.colors["red"], lw=2)
    plt.plot(X, y2, color=params.colors["red"], lw=2)
    if X_samples is not None:
        plt.plot(
            X_samples[:num_sampl],
            y_samples[:num_sampl],
            "o",
            color=params.colors["grey"],
            ms=8,
        )
    plt.errorbar(x_new, (lb + ub) / 2, (ub - lb) / 2, ecolor=params.colors["red"])
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    fig.tight_layout()
    plt.show()
    fig.savefig("plots//" + "strangeteststuff", bbox_inches="tight")


def train_plot_noise(samples=True, wtrue=True):
    num_sampl = 7
    x_max = 7
    dbar = 0.4
    lengthscale = 3
    gamma = 227.2  # 39.4
    X_samples, y_samples = sample_rkhs_func(num=20, noise=dbar)
    model = KRR(0.001, lengthscale, dbar, gamma)  # 6.4
    opti_model = OptiBound(lengthscale, dbar, gamma)  # 6.4
    model.fit(X_samples[:num_sampl], y_samples[:num_sampl])
    opti_model.fit(X_samples[:num_sampl], y_samples[:num_sampl])
    print(f"Norm KRR {model.get_norm()}")
    X = np.linspace(0, x_max, 500).reshape((-1, 1))
    y = model.predict(X)
    lb_subopt = []
    ub_subopt = []
    lb_opt = []
    ub_opt = []
    for x in X:
        lb_subopt.append(model.get_lower_bound(x.reshape((1, -1))))
        ub_subopt.append(model.get_upper_bound(x.reshape((1, -1))))
        lb_opt.append(opti_model.get_lower_bound(x.reshape((1, -1))))
        ub_opt.append(opti_model.get_upper_bound(x.reshape((1, -1))))
    lb_subopt = np.array(lb_subopt).flatten()
    ub_subopt = np.array(ub_subopt).flatten()
    lb_opt = np.array(lb_opt).flatten()
    ub_opt = np.array(ub_opt).flatten()
    ys = [y]
    colors = ["red"]
    add_name = ""
    if wtrue:
        ytrue = rkhs(X, lengthscale=lengthscale)
        ys.append(ytrue)
        # colors.append("green")
        add_name = "both"
    gt = rkhs(X, lengthscale=lengthscale)
    if samples:
        plot_dat(
            X,
            ys,
            colors,
            "krr_rkhs_samples_noise" + str(dbar) + add_name + str(num_sampl),
            X_samples[:num_sampl],
            y_samples[:num_sampl],
            y_min=-3.5,
            y_max=5.5,
            x_min=0,
            x_max=x_max,
        )
        plot_bounds(
            X.flatten(),
            [lb_subopt, lb_opt],
            [ub_subopt, ub_opt],
            ["red", "green"],
            "bounds_krr_noise" + str(dbar) + str(num_sampl),
            ys=ys,
            gt=gt,
            X_samples=X_samples[:num_sampl],
            y_samples=y_samples[:num_sampl],
            y_min=-3.5,
            y_max=5.5,
            x_min=0,
            x_max=x_max,
        )
        plot_bounds(
            X.flatten(),
            [lb_subopt],
            [ub_subopt],
            ["red"],
            "bounds_krr_noise_subopt" + str(dbar) + str(num_sampl),
            ys,
            gt,
            X_samples[:num_sampl],
            y_samples[:num_sampl],
            y_min=-3.5,
            y_max=5.5,
            x_min=0,
            x_max=x_max,
        )
        plot_bounds(
            X.flatten(),
            [lb_opt],
            [ub_opt],
            ["red"],
            gt=gt,
            name="bounds_krr_noise_opt" + str(dbar) + str(num_sampl),
            X_samples=X_samples[:num_sampl],
            y_samples=y_samples[:num_sampl],
            y_min=-3.5,
            y_max=5.5,
            x_min=0,
            x_max=x_max,
        )
    else:
        plot_dat(
            X,
            ys,
            colors,
            "krr_rkhs_nosamples_noise" + str(dbar) + add_name + str(num_sampl),
            y_min=-1.5,
            y_max=4.5,
            x_min=0,
            x_max=x_max,
        )
        plot_bounds(
            X.flatten(),
            [lb_subopt, lb_opt],
            [ub_subopt, lb_opt],
            ["red", "green"],
            "bounds_krr_noise" + str(dbar) + str(num_sampl),
            y_min=-1.5,
            y_max=4.5,
            x_min=0,
            x_max=x_max,
        )


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
    X,
    ys,
    colors,
    name,
    X_samples=None,
    y_samples=None,
    _type=".png",
    y_min=0,
    y_max=1,
    x_min=0,
    x_max=5,
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
    for i, y in enumerate(ys):
        if i + 1 > len(colors):
            plt.plot(X, y, "k--", lw=2)
        else:
            plt.plot(X, y, color=params.colors[colors[i]], lw=2)
    if X_samples is not None:
        plt.plot(X_samples, y_samples, "o", color=params.colors["grey"], ms=8)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    fig.tight_layout()
    plt.show()
    fig.savefig("plots//" + name + _type, bbox_inches="tight")


def plot_bounds(
    X,
    lbs,
    ubs,
    colors,
    name,
    ys=None,
    gt=None,
    X_samples=None,
    y_samples=None,
    _type=".png",
    y_min=0,
    y_max=1,
    x_min=0,
    x_max=5,
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
    for i, lb in enumerate(lbs):
        plt.fill_between(X, lb, ubs[i], color=params.colors[colors[i]], alpha=0.2)
    if ys is not None:
        for i, y in enumerate(ys):
            plt.plot(X, y, color=params.colors[colors[i]], lw=2)
    if gt is not None:
        plt.plot(X, gt, "--k", lw=2)
    if X_samples is not None:
        plt.plot(X_samples, y_samples, "o", color=params.colors["grey"], ms=8)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    fig.tight_layout()
    plt.show()
    fig.savefig("plots//" + name + _type, bbox_inches="tight")


if __name__ == "__main__":
    plot_rkhs_func(samples=True, num=20, num_plot=7, noise=0.4)
    train_plot_noise(wtrue=False)
    train_plot_single_noise()
    # plot_rkhs_func(samples=False)
    # train_plot_krr(samples=False)
    # train_plot_krr()
    # plot_true_funcs()
    # approx_funcs()
