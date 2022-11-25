import math
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model._ridge import _solve_cholesky_kernel
import numpy as np
import cvxpy as cp
from scipy.spatial import distance
import abc


class BaseModel(KernelRidge):
    def __init__(self, alpha, lengthscale, noisebound, Gamma, kernel="rbf"):
        super().__init__(alpha=alpha, gamma=(1 / (2 * lengthscale**2)), kernel=kernel)
        self.dbar = noisebound
        self.Gamma = Gamma
        self.dual_coef_ = None

    def fit(self, X, y, sample_weight=None):
        super().fit(X, y, sample_weight)
        self.K = self._get_kernel(X)
        self.y = y
        self.N = len(y)

    def get_norm(self):
        assert self.dual_coef_ is not None
        return np.sqrt(np.dot(self.y.T, self.dual_coef_)).flatten()

    @abc.abstractmethod
    def get_lower_bound(self, X):
        pass

    @abc.abstractmethod
    def get_upper_bound(self, X):
        pass


class OptiBound(BaseModel):
    def __init__(self, lengthscale, noisebound, Gamma, kernel="rbf", precision=1e-8):
        super().__init__(
            alpha=precision,
            lengthscale=lengthscale,
            noisebound=noisebound,
            Gamma=Gamma,
            kernel=kernel,
        )
        self.precision = precision

    def get_bound(self, X, btype):
        dist = distance.cdist(X, self.X_fit_).min()
        if dist <= 1e-16:
            cost_index = np.where(distance.cdist(X, self.X_fit_) <= self.precision)[1]
            Kn = self.K + self.precision * np.eye(self.N)
            In = np.eye(self.N)
            cost_len = self.N
        else:
            Xn = np.append(self.X_fit_, X, axis=0)
            Kn = self._get_kernel(Xn) + self.precision * np.eye(self.N + 1)
            In = np.append(np.eye(self.N), np.zeros((self.N, 1)), axis=1)
            cost_len = self.N + 1
            cost_index = -1
        costvec = np.zeros(cost_len)
        costvec[cost_index] = 1
        c = cp.Variable(cost_len)
        if btype == "min":
            cost = cp.Minimize(costvec.T * c)
        elif btype == "max":
            cost = cp.Maximize(costvec.T * c)
        else:
            raise ValueError(f"Unknown bound type {btype}.")
        constraints = [
            cp.matrix_frac(c, Kn) <= self.Gamma**2,
            In * c - self.y.flatten() <= np.ones(self.N) * self.dbar,
            -In * c + self.y.flatten() <= np.ones(self.N) * self.dbar,
        ]
        prob = cp.Problem(cost, constraints)
        prob.solve()
        return prob.value

    def get_bound_v2(self, X, btype):
        Kx = self._get_kernel(self.X_fit_, X)
        costvec = _solve_cholesky_kernel(self.K, Kx, self.precision)
        In = np.eye(self.N)
        c = cp.Variable(self.N)
        if btype == "min":
            cost = cp.Minimize(costvec.T * c)
        elif btype == "max":
            cost = cp.Maximize(costvec.T * c)
        else:
            raise ValueError(f"Unknown bound type {btype}.")
        constraints = [
            cp.matrix_frac(c, self.K) <= self.Gamma**2,
            In * c - self.y.flatten() <= np.ones(self.N) * self.dbar,
            -In * c + self.y.flatten() <= np.ones(self.N) * self.dbar,
        ]
        prob = cp.Problem(cost, constraints)
        prob.solve()
        return prob.value

    def get_lower_bound(self, X):
        return self.get_bound(X, "min").flatten()

    def get_upper_bound(self, X):
        return self.get_bound(X, "max").flatten()


class KRR(BaseModel):
    def __init__(
        self, alpha, lengthscale, noisebound, Gamma, kernel="rbf", precision=1e-8
    ):
        super().__init__(
            alpha=alpha,
            lengthscale=lengthscale,
            noisebound=noisebound,
            Gamma=Gamma,
            kernel=kernel,
        )
        self.precision = precision
        self.interp_dual_coef_ = None
        self.delta_tilde = None

    def fit(self, X, y, sample_weight=None):
        super().fit(X, y, sample_weight)
        self.interp_dual_coef_ = _solve_cholesky_kernel(self.K, self.y, self.precision)
        self.delta_tilde = self.get_delta_tilde()

    def get_delta_tilde(self):
        nu = cp.Variable(self.N)
        cost = cp.Minimize(
            1 / 4 * cp.quad_form(nu, self.K)
            + nu.T @ self.y
            + self.dbar * cp.norm(nu, 1)
        )
        prob = cp.Problem(cost)
        prob.solve()
        return prob.value

    def power(self, X):
        Kx = self._get_kernel(self.X_fit_, X)
        sub = np.dot(Kx.T, _solve_cholesky_kernel(self.K, Kx, self.precision))
        return np.sqrt(self._get_kernel(X, X) - sub)

    def get_bound(self, X):
        Kx = self._get_kernel(self.X_fit_, X)
        noisefreeerror = self.power(X) * np.sqrt(self.Gamma**2 - self.delta_tilde)
        noiseerror = self.dbar * np.linalg.norm(
            _solve_cholesky_kernel(self.K, Kx, self.precision), ord=1
        )
        regularizationerror = np.abs(
            np.dot(Kx.T, self.dual_coef_ - self.interp_dual_coef_)
        )
        err = noisefreeerror + noiseerror + regularizationerror
        return err.flatten()

    def get_lower_bound(self, X):
        lb = self.predict(X).flatten() - self.get_bound(X)
        return lb

    def get_upper_bound(self, X):
        ub = self.predict(X).flatten() + self.get_bound(X)
        return ub

    def get_interp_norm(self):
        assert self.interp_dual_coef_ is not None
        return np.sqrt(np.dot(self.y.T, self.interp_dual_coef_)).flatten()


    def get_interp_bound(self, X):
        return self.power(X) * np.sqrt(self.Gamma**2 - self.get_norm() ** 2)
