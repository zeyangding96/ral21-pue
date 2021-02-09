import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class myRnnPue(nn.Module):
    def __init__(self, input_dim, hidden_dim, mu_dim, logvar_dim, core='lstm', dropout=0, device='cpu'):
        super().__init__()
        self.device = device
        self.core = core
        if core == 'gru': self.recurrent = nn.GRU(input_dim, hidden_dim, num_layers=3, batch_first=True, dropout=dropout)
        elif core == 'lstm': self.recurrent = nn.LSTM(input_dim, hidden_dim, num_layers=3, batch_first=True, dropout=dropout)
        self.mu_func = nn.Linear(hidden_dim, mu_dim)
        self.logvar_func = nn.Linear(hidden_dim, logvar_dim)

    def forward(self, x):
        x, _ = self.recurrent(x)
        mu = self.mu_func(x)
        logvar = self.logvar_func(x)
        return mu, logvar

    def nll_loss(self, mu, logvar, target):
        ## NLL of y given x under Gaussian assumption
        ## arguments have shape (bs, seq_len, mu/logvar/_dim)
        mu = mu.reshape(-1, mu.shape[-1])
        logvar = logvar.reshape(-1, logvar.shape[-1])
        target = target.reshape(-1, target.shape[-1])
        N = mu.shape[0]
        term1 = logvar
        term2 = F.mse_loss(mu, target, reduction='none') / torch.exp(logvar)
        loss = 0.5 * (term1 + term2).sum() / N / mu.shape[-1]
        return loss

    def init_train(self, optimizer, loader_train, loader_val=None):
        self.to(device=self.device)
        self.optimizer = optimizer
        self.loader_train = loader_train
        self.loader_val = loader_val     
        self.loss_train_history, self.loss_val_history = [], []
        self.lr_init = [p['lr'] for p in self.optimizer.param_groups]
        self.lr_final = 1e-6

    def Train(self, epochs=1, criterion='nll', check_train=True, check_val=True, verbose=True):
        if check_val: assert self.loader_val is not None, 'Validation data not initialized.'
        for e in range(epochs):
            self.train()
            for t, tup in enumerate(self.loader_train):
                x = tup[0].to(device=self.device, dtype=torch.float)
                y = tup[1].to(device=self.device, dtype=torch.float)
                self.optimizer.zero_grad()
                mu, logvar = self(x)
                if criterion == 'nll': loss = self.nll_loss(mu, logvar, y)
                elif criterion == 'mse': loss = F.mse_loss(mu, y)
                loss.backward()
                self.optimizer.step()
            self.decay_lr(e+1, epochs)
            if check_train:
                loss_train = self.validate(self.loader_train, loss=True)
                self.loss_train_history.append(loss_train)
            if check_val:
                loss_val = self.validate(self.loader_val, loss=True)
                self.loss_val_history.append(loss_val)
            if verbose:
                print('Epochs %d/%d' % (e+1, epochs))
                if check_train: print('Train MSE Loss = %.4f' % loss_train[1], end=', ')
                if check_val: print('Val MSE Loss = %.4f' % loss_val[1])
        if verbose: print('Training done!')

    def decay_lr(self, e, epochs):
        for p, lr_init in zip(self.optimizer.param_groups, self.lr_init):
            amplitude = (lr_init - self.lr_final) / 2
            p['lr'] = amplitude * math.cos(math.pi * e / epochs) + amplitude + self.lr_final

    def validate(self, loader, loss=True, mcdropout=False):
        if mcdropout: self.train()
        else: self.eval()
        Mu, LogVar, y_true = [], [], []
        with torch.no_grad():
            for tup in loader:
                x = tup[0].to(device=self.device, dtype=torch.float)
                y = tup[1].to(device='cpu', dtype=torch.float)
                mu, logvar = self(x)
                Mu += [ mu.cpu() ]
                LogVar += [logvar.cpu()]
                y_true += [ y ]
            Mu, LogVar, y_true = torch.cat(Mu), torch.cat(LogVar), torch.cat(y_true)
            if loss:
                return self.nll_loss(Mu, LogVar, y_true).item(), F.mse_loss(Mu, y_true).item()
            else:
                Var = torch.exp(LogVar)
                return Mu, Var, y_true
