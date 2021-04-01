import math
import time
import cvxpy as cp
from utils import Laplacian_dual, Laplacian_inv, Laplacian

class LearnGraphTopolgy:
    def __init__(self, S, alpha=0, beta=1e4, n_iter=10000, c1=0., c2=1e10, tol = 1e-6):
        self.tol = tol
        self.S = S
        self.p = S.shape[0]
        self.n = np.int(self.p * (self.p - 1)/2)
        self.alpha = alpha
        self.beta = beta
        self.n_iter = n_iter
        self.c1 = c1
        self.c2 = c2

    def w_init(self, w0_init, Sinv):
        """
        Initialization of w
        """
        if w0_init == 'naive':
            w0 = Laplacian_inv(Sinv, self.p, self.n)
            w0[w0<0] = 0
        else:
            raise ValueError('Method not implemented')
        return w0

    def update_w(self, w, Lw, U, lambda_, K):
        """
        Compute w update as equation 12
        """
        c = Laplacian_dual(U @ np.diag(lambda_) @ U.T - K / self.beta, self.p, self.n)
        grad_f = Laplacian_dual(Lw, self.p, self.n) - c
        M_grad_f = Laplacian_dual(Laplacian(grad_f, self.p), self.p, self.n)
        wT_M_grad_f = sum(w * M_grad_f)
        dwT_M_dw = sum(grad_f * M_grad_f)
        ## exact line search
        t = (wT_M_grad_f - sum(c * grad_f)) / dwT_M_dw
        ## no line search :
        # p = int(0.5*(1 + np.sqrt(1 + 8*w.shape[0])))
        # t = 0.5*p
        w_new = w - t * grad_f
        w_new[w_new < 0] = 0
        return w_new

    def adjacency(self, w):
        '''
        Compute the Adjacency matrix from w
        '''
        Aw = np.zeros((self.p,self.p))
        k=0
        for i in range(0, self.p):
            for j in range(i+1,self.p):
                Aw[i][j] = w[k]
                k = k + 1
        Aw = Aw + Aw.T
        return Aw

    def update_lambda(self, U, Lw, k):
        """
        Compute lambda update as proposed in the supplementary
        """
        q = Lw.shape[1] - k
        d = np.diag(np.dot(U.T, np.dot(Lw, U)))
        assert(d.shape[0] == q)
        lambda_ = (d + np.sqrt(d**2 + 4/self.beta))/2

        cond = (lambda_[q-1] - self.c2 <= 1e-9) and (lambda_[0] - self.c1 >= -1e-9) and np.all(lambda_[1:q] - lambda_[0:q-1] >= -1e-9)

        if cond:
            return lambda_
        else:
            lambda_[lambda_ < self.c1] = self.c1
            lambda_[lambda_ > self.c2] = self.c2
        cond = (lambda_[q-1] - self.c2 <= 1e-9) and (lambda_[0] - self.c1 >= -1e-9) and np.all(lambda_[1:q] - lambda_[0:q-1] >= -1e-9)

        if cond:
            return lambda_
        else:
            raise ValueError("Consider increasing value of beta")

    def update_U(self, Lw, k):
        """
        Compute U update as equation 14
        """
        _, eigvec = np.linalg.eigh(Lw)
        assert(eigvec.shape[1] == self.p)
        return eigvec[:, k:]

    def objective(self, Lw, lambda_, K, U):
        """
        Compute objective function - equation 8
        """
        term1 = np.sum(-np.log(lambda_))
        term2 = np.trace(np.dot(K, Lw))
        term3 = 0.5 * self.beta * np.linalg.norm(Lw - np.dot(U, np.dot(np.diag(lambda_), U.T)), ord='fro')**2
        return term1 + term2 + term3

    def learn_graph(self, k=1, w0_init='naive', eps = 1e-4):

        # find an appropriate inital guess
        Sinv = np.linalg.pinv(self.S)
        # if w0 is either "naive" or "qp", compute it, else return w0
        w = self.w_init(w0_init, Sinv)
        # compute quantities on the initial guess
        Lw = Laplacian(w, self.p)
        # l1-norm penalty factor
        H = self.alpha * (np.eye(self.p) - np.ones((self.p, self.p)))
        K = self.S + H
        U = self.update_U(Lw = Lw, k = k)
        lambda_ = self.update_lambda(U = U, Lw = Lw, k = k)

        objective_seq = []

        for _ in range(self.n_iter):
            w_new = self.update_w(w = w, Lw = Lw, U = U, lambda_ = lambda_, K = K)
            Lw = Laplacian(w_new, self.p)
            U = self.update_U(Lw = Lw, k = k)
            lambda_ = self.update_lambda(U = U, Lw = Lw, k = k)

            # check for convergence
            convergence = np.linalg.norm(w_new - w, ord=2) < self.tol
            objective_seq.append(self.objective(Lw, lambda_, K, U))

            if convergence:
                break

            # update estimates
            w = w_new
            K = self.S + H / (-Lw + eps)

        # compute the adjancency matrix
        Aw = self.adjacency(w)
        results = {'laplacian' : Lw, 'adjacency' : Aw, 'w' : w, 'lambda' : lambda_, 'U' : U,
        'convergence' : convergence, 'objective_seq' : objective_seq}
        return results
