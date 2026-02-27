import numpy as np
from load_data import load_movielens_fold


class ALSRecommender:

    def __init__(self, n_factors=50, lambda_reg=0.3, n_iters=25):
        self.k = n_factors
        self.lambda_reg = lambda_reg
        self.n_iters = n_iters
        self.U = None
        self.V = None
        self.mu = None

    def fit(self, R):

        m, n = R.shape
        mask = R > 0

        self.mu = np.sum(R) / np.sum(mask)

        R_centered = R.copy()
        R_centered[mask] -= self.mu

        self.U = np.random.normal(scale=0.1, size=(m, self.k))
        self.V = np.random.normal(scale=0.1, size=(n, self.k))

        for iteration in range(self.n_iters):

            print(f"Iteration {iteration + 1}/{self.n_iters}")

            for u in range(m):

                idx = mask[u]
                if not np.any(idx):
                    continue

                V_i = self.V[idx]
                R_u = R_centered[u, idx]

                A = V_i.T @ V_i + self.lambda_reg * np.eye(self.k)
                b = V_i.T @ R_u

                self.U[u] = np.linalg.solve(A, b)

            for i in range(n):

                idx = mask[:, i]
                if not np.any(idx):
                    continue

                U_u = self.U[idx]
                R_i = R_centered[idx, i]

                A = U_u.T @ U_u + self.lambda_reg * np.eye(self.k)
                b = U_u.T @ R_i

                self.V[i] = np.linalg.solve(A, b)

    def predict(self):
        return self.mu + self.U @ self.V.T

    def rmse(self, R_true):

        mask = R_true > 0
        R_pred = self.predict()

        return np.sqrt(
            np.sum((R_true[mask] - R_pred[mask]) ** 2)
            / np.sum(mask)
        )


if __name__ == "__main__":

    print("Loading MovieLens fold u1...")
    train, test = load_movielens_fold(
        "data/ml-100k/u1.base",
        "data/ml-100k/u1.test"
    )

    factor_grid = [20, 40, 60]
    lambda_grid = [0.1, 0.2, 0.3, 0.5]
    iter_grid = [15, 25]

    best_rmse = float("inf")
    best_config = None

    for k in factor_grid:
        for lam in lambda_grid:
            for iters in iter_grid:

                print("\n-----------------------------------")
                print(f"Testing: k={k}, lambda={lam}, iters={iters}")

                model = ALSRecommender(
                    n_factors=k,
                    lambda_reg=lam,
                    n_iters=iters
                )

                model.fit(train)

                rmse_val = model.rmse(test)

                print("Validation RMSE:", rmse_val)

                if rmse_val < best_rmse:
                    best_rmse = rmse_val
                    best_config = (k, lam, iters)

    print("\n==============================")
    print("Best Configuration Found:")
    print("k =", best_config[0])
    print("lambda =", best_config[1])
    print("iterations =", best_config[2])
    print("Best Test RMSE =", best_rmse)