import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

class GLEMR:
    '''
    Parameters::
        K: int, default=3, recommendation=[2, 3, 4, 5]
            Number of mixture components.
        lamz: float, default=1e-3, recommendation=[1e-3, 1e-2, ..., 1e0]
            Strength of the prior of label distributions.
        lamy: float, default=1e5, recommendation=[1e5, 2e5, 3e5, 4e5]
            Strength of logical labels.
        trace_step: int, default=20
            Result recording step, which facilitates early-stop technique.
        verbose: int, default=0 
            How many intermediate ELBO values will be printed during training.
        lr: float, default=1e-3
            Learning rate of Adam.
        max_iter: int, default=500
            Maximum iterations of Adam.
    --------------------------------------------------------------
    Attributes::
        label_distribution_: ndarray of shape (n_samples, n_labels)
            Recovered label distributions.
        trace_: dict of label_distribution_
            Recovered label distributions on different epoch.
    --------------------------------------------------------------
    Methods::
        fit(X, L): training the model with feature matrix X and logical label matrix L.
    --------------------------------------------------------------
    Examples::
        >>> Drec = GLEMR(trace_step=np.inf).fit(X, L).label_distribution_
        >>> evaluate(Drec, ground_truth)
        >>> Drec_trace = GLEMR(trace_step=100).fit(X, L).trace_
        >>> for k in Drec_trace.keys():
        >>>     evaluate(Drec_trace[k], ground_truth)
    '''

    def __init__(self, lamz=1e-3, K=3, lamy=1e5, lr=1e-3, max_iter=500, verbose=0, trace_step=20, random_state=1):
        self.K = K
        self.lamy = lamy
        self.lamz = lamz
        self.lr = lr
        self.trace_step = trace_step
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, L):
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        N, M = L.shape
        K = self.K
        X, L = torch.FloatTensor(X), torch.FloatTensor(L)
        X = (X - X.min(0, keepdims=True)[0]) / (X.max(0, keepdims=True)[0] - X.min(0, keepdims=True)[0])
        observation = torch.cat([X, L], dim=1)
        Lmask = L == 1
        C = 1 - L.sum(1) / M
        pxmu = nn.Sequential(nn.Linear(M, X.shape[1] * K))
        pxsigma = nn.Sequential(nn.Linear(M, X.shape[1] * K), nn.Softplus())
        self.qzmu = nn.Sequential(nn.Linear(X.shape[1] + M, M))
        self.qzsigma = nn.Sequential(nn.Linear(X.shape[1] + M, M), nn.Softplus())
        paramls = list(pxmu.parameters()) + list(pxsigma.parameters()) + \
                  list(self.qzmu.parameters()) + list(self.qzsigma.parameters())
        for p in paramls:
            nn.init.normal_(p, mean=0.0, std=0.1)
        self.trace_ = dict()
        optimizer = torch.optim.Adam(paramls, lr=self.lr)
        for epoch in range(self.max_iter + 1):
            optimizer.zero_grad()
            epsilon = torch.randn(N, M)
            zmu, zsigma = self.qzmu(observation), self.qzsigma(observation) + 1e-5   # shape = (N, M)
            zsample = zmu + zsigma.sqrt() * epsilon    # shape = (N, M)
            dsample = torch.softmax(zsample, dim=1) + 1e-5 # shape = (N, M)
            xmu, xsigma = pxmu(dsample).view(N, -1, K), pxsigma(dsample).view(N, -1, K) + 1e-5
            loglike_x = Normal(xmu, xsigma).log_prob(X.unsqueeze(-1)).sum(1)   # shape = (N, K)
            loglike_y = ((dsample * Lmask).sum(-1) + C).log()
            pi = torch.softmax(loglike_x, dim=1) + 1e-5
            Lrec = ( (pi * loglike_x).sum(-1) + self.lamy *  loglike_y) / (1 + self.lamy)
            zprior = .5 * (self.lamz * zsigma.sum(-1) + self.lamz * zmu.pow(2).sum(-1) - zsigma.log().sum(-1))
            cprior = (pi * pi.log()).sum(-1)
            ELBO = (Lrec - zprior - cprior).mean()
            loss = -ELBO
            loss.backward()
            optimizer.step()
            if (not np.isinf(self.trace_step)) and (epoch % self.trace_step == 0):
                Drec = self._transform(observation)
                self.trace_[epoch] = Drec
            if (self.verbose > 0) and (epoch % (self.max_iter // self.verbose) == 0):
                with torch.no_grad():
                    print("* epoch: %4d, elbo: %.3f" % (epoch, ELBO.item()))
        if np.isinf(self.trace_step):
            self.label_distribution_ = self._transform(observation)
        return self

    def _transform(self, observation):
        with torch.no_grad():
            mu, sigma = self.qzmu(observation).numpy(), self.qzsigma(observation).numpy() + 1e-5
            expect = lambda i: 1 / ( np.exp( (mu - mu[:,[i]]) / np.sqrt(1+3/np.pi**2*(sigma[:,[i]] + sigma)) ).sum(1) )
            Drec = np.concatenate([expect(i).reshape(-1, 1) for i in range(mu.shape[1])], axis=1)
            Drec /= Drec.sum(1,keepdims=True)
            return Drec
