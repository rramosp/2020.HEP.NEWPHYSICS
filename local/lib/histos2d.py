import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KernelDensity
from scipy import stats
import pandas as pd
from sklearn.neighbors import KernelDensity
from scipy.optimize import minimize
from progressbar import progressbar as pbar


def kdensity_smoothed_histogram(x):
    x = pd.Series(x).dropna().values
    t = np.linspace(np.min(x), np.max(x), 100)
    p = kdensity(x)(t)
    return t, p

def kdensity(x):
    import numbers
    if len(x.shape) != 1:
        raise ValueError("x must be a vector. found "+str(x.shape)+" dimensions")
    stdx = np.std(x)
    bw = 1.06*stdx*len(x)**-.2 if stdx != 0 else 1.
    kd = KernelDensity(bandwidth=bw)
    kd.fit(x.reshape(-1, 1))

    func = lambda z: np.exp(kd.score_samples(np.array(z).reshape(-1, 1)))
    return func


def plot_kdensity_smoothed_histogram(x, plot_equivalent_gaussian=False, plot_equivalent_poisson=False):
    plt.plot(*kdensity_smoothed_histogram(x), color="black", lw=2, label="data")
    
    if plot_equivalent_gaussian:
        m,s = np.mean(x), np.std(x)
        p = np.percentile(x, [1,99])
        xr = np.linspace(p[0], p[1], 100)
        plt.plot(xr, stats.norm(loc=m, scale=s).pdf(xr), color="blue", alpha=.5, lw=2, label="equiv gaussian")  
    if plot_equivalent_poisson:
        assert x.dtype==int, "for plotting poisson equivalent your data must be composed of integers"
        
        m,v = np.mean(x), np.std(x)
        ep = stats.poisson(loc=np.round(m-v**2,0), mu=v**2)
        ks = pd.Series(x).value_counts().sort_index()/len(x)
        plt.plot(ks.index, ep.pmf(ks.index.values), color="red", label="equiv poisson")
        
class BS_raw_experiment:
    def __init__(self, t, mu_s, sigma_s, mu):
        self.b = stats.expon(scale=1/t)
        self.s = stats.norm(loc=mu_s, scale=sigma_s)    
        
        self.t, self.mu_s, self.sigma_s, self.mu = t, mu_s, sigma_s, mu
        
    def sample(self, n):
        nb = int(n/(1+self.mu))
        ns = n - nb
        self.b_data = self.b.rvs(nb)
        self.s_data = self.s.rvs(ns)
        return np.hstack([self.b_data, self.s_data])
    
    def get_pdf(self, mu=None):
        mu = mu or self.mu            
        p = lambda x: (self.b.pdf(x) + mu*self.s.pdf(x)) / (mu+1)
        return p
    
    def log_likelihood(self, m, data):
        return np.mean(np.log( self.get_pdf(m)(data)))
    
    def get_loglikelihood_estimate_for_mu(self, data):
        r = minimize(lambda m: -self.log_likelihood(m, data), [np.random.random()], 
                     constraints = ({'type': 'ineq', 'fun': lambda x: x-1e-3}))
        return r.x[0]
    
    def get_bins_counts(self, data, bin_edges):
        binned_data = np.r_[[np.argwhere(i>bin_edges)[-1][0] for i in data]]
        return pd.Series(binned_data).value_counts()
    
    # ----------------------------
    # plot stuff                    
    # ----------------------------
    def plot_mle(self, data, maxmu=None, show_mle=True):
        maxmu = maxmu if maxmu is not None else self.mu*2
        mr = np.linspace(1e-6,maxmu,30)
        lm = [self.log_likelihood(m,data) for m in mr]
        plt.plot(mr, lm)
        plt.axvline(self.mu, color="black", label="real $\mu$")
        if show_mle:
            mle_mu = self.get_loglikelihood_estimate_for_mu(data)
            plt.axvline(mle_mu, color="red", label="mle $\mu$")
        plt.grid(); plt.legend();      
        plt.xlabel("$\mu$"); plt.ylabel("log likelihood")
        
    def plot_hist(self, data, bins=100, plot_pdf=True):
        plt.hist(data, density=True, bins=bins, color="blue", alpha=.3);
        maxx = np.percentile(data, 99)
        plt.xlim(0, maxx)        
        xr = np.linspace(0, maxx, 1000)
        plt.plot(xr, self.get_pdf()(xr), color='black', label="pdf")
        plt.xlim(0,maxx)  
        plt.grid(); plt.legend();
        plt.title("$\mu=%.3f$"%self.mu)
        plt.xlabel("$M{\gamma\gamma}$"); 
        
    def plot(self, data, kwhist={}, kwmle={}):
        plt.figure(figsize=(13,4))
        plt.subplot(121)
        self.plot_hist(data, **kwhist)
        plt.subplot(122)
        self.plot_mle(data, **kwmle)
        
    def plot_MLE_mu_distributions(self):
        sample_sizes = [50,100,500,1000,5000] 
        n = 400
        estimates = [[self.get_loglikelihood_estimate_for_mu(self.sample(sample_size)) for _ in range(n)] \
                     for sample_size in pbar(sample_sizes)]

        plt.figure(figsize=(12,4))
        plt.subplot(121)
        r = np.r_[[[np.mean(i), np.std(i)] for i in estimates]]
        plt.plot(r[:,0], lw=3, label="mean MLE $\mu$")
        plt.fill_between(range(len(r)), r[:,0]+r[:,1], r[:,0]-r[:,1], color="blue", alpha=.1, label="+-1 std MLE $\mu$")
        plt.axhline(self.mu, color="red", alpha=.4, label="true $\mu=%.2f$"%self.mu)
        plt.grid();
        plt.xticks(range(len(sample_sizes)), sample_sizes);
        plt.xlabel("data size")
        plt.ylabel("MLE $\mu$")
        plt.legend();

        plt.subplot(122)
        for size, i in zip(sample_sizes, estimates):
            plt.plot(*kdensity_smoothed_histogram(i), label="data size = %d"%size)
        plt.grid(); plt.legend();
        plt.title("smoothed histograms for estimated $\mu$")
        plt.xlabel("MLE $\mu$");      
        plt.xlim(0, self.mu*2)
                
    def plot_bins_distributions(self, dataset_size, n_datasets, bin_edges):
        k = [self.get_bins_counts(self.sample(dataset_size), bin_edges) for _ in pbar(range(n_datasets))]
        k = pd.DataFrame(k)
        k.index.name="dataset_nb"
        k.columns.name="bin_nb"

        plt.figure(figsize=(2.5*len(k.columns),2.5))
        for i,col in enumerate(k.columns):
            plt.subplot(1,len(k.columns), i+1)
            plot_kdensity_smoothed_histogram(k[col].values, plot_equivalent_poisson=True)
            plt.title("bin %d\n$M_{\gamma \gamma} \in [%.2f, %.2f]$"%(col, bin_edges[i], bin_edges[i+1]))
            plt.yticks([])
            plt.xlabel("nb events")
            if i==len(k.columns)//2:
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25))      
                
                
class BS_hist_experiment:
    def __init__(self, t, mu_s, sigma_s, mu, stot, btot, bin_edges):
        self.b = stats.expon(scale=1/t)
        self.s = stats.norm(loc=mu_s, scale=sigma_s)            
        self.t, self.mu_s, self.sigma_s, self.mu, self.stot, self.btot = t, mu_s, sigma_s, mu, stot, btot
        self.bin_edges = bin_edges
        self.n_bins = len(bin_edges)-1
        
        # precompute analytical values
        self.ni, self.si, self.bi = self.get_analytical_histogram()
        
    def get_analytical_histogram(self, mu=None, stot=None, btot=None):
        mu = self.mu if mu is None else mu
        stot = self.stot if stot is None else stot
        btot = self.btot if btot is None else btot
        si = self.stot*pd.Series([self.s.cdf(i) for i in self.bin_edges]).diff().dropna().values
        bi = self.btot*pd.Series([self.b.cdf(i) for i in self.bin_edges]).diff().dropna().values
        ni = self.mu * si + bi
        return ni,si,bi
    
    def get_poisson_pmf(self, mu, si, bi):        
        return stats.poisson(mu=mu*si+bi).pmf
    
    def get_poisson_pmf_for_bin(self, bin_nb):
        return stats.poisson(mu=self.ni[bin_nb]).pmf
        
    def loglikehood(self, mu, observed_histogram):
        return np.sum([np.log(1e-100+self.get_poisson_pmf(mu, self.si[bin_nb], self.bi[bin_nb])(observed_histogram[bin_nb])) for bin_nb in range(self.n_bins)])
    
    def get_loglikelihood_estimate_for_mu(self, observed_histogram):
        from scipy.optimize import minimize
        r = minimize(lambda mu: -self.loglikehood(mu=mu, observed_histogram=observed_histogram) , np.random.random(), 
                 constraints = ({'type': 'ineq', 'fun': lambda x: x-1e-3}))
        return r.x[0]
        
    def sample_histograms(self, n_samples):
        """
        samples the histograms distribution: n = s + b
        """
        nl, sl, bl = [], [], []
        be = self.bin_edges
        iterator = pbar(range(n_samples)) if n_samples>1 else range(n_samples)
        for _ in iterator:
            sample_s = self.s.rvs(self.stot)
            sample_b = self.b.rvs(self.btot)
            ks = np.r_[[np.sum((sample_s>be[i])&(sample_s<be[i+1])) for i in range(0,len(be)-1)]]
            kb = np.r_[[np.sum((sample_b>be[i])&(sample_b<be[i+1])) for i in range(0,len(be)-1)]]
            nl.append(self.mu*ks+kb)
            sl.append(ks)
            bl.append(kb)
        nl = pd.DataFrame(np.r_[nl]); nl.index.name ="sample_nb"; nl.columns.name = "bin_nb"
        sl = pd.DataFrame(np.r_[sl]); sl.index.name ="sample_nb"; sl.columns.name = "bin_nb"
        bl = pd.DataFrame(np.r_[bl]); bl.index.name ="sample_nb"; bl.columns.name = "bin_nb"
        return nl,sl,bl
    
    def plot_histograms_distribution(self):
        plt.bar(range(len(self.ni)), self.ni, alpha=.2, color="green", label="analytical histograms distribution")
        plt.bar(range(len(self.bi)), self.bi, alpha=.2, color="blue",  label="analytical background distribution")
        plt.bar(range(len(self.si)), self.si, alpha=.2, color="red",   label="analytical signal distribution")
        n = 1000
        k,_,_ = self.sample_histograms(n)
        k.mean(axis=0).plot(color="black", label="mean of %d sampled histograms"%n)
        plt.grid(); plt.legend();
        plt.ylabel("number of measurements")
        
    def plot_bin_distributions(self):

        n_histograms = 1000
        sn,ss,sb = self.sample_histograms(n_histograms)

        ncols = 9
        nrows = int(np.ceil(len(self.ni)/ncols))

        fig = plt.figure(figsize=(2.2*ncols,1.8*nrows))

        for bin_nb in range(len(self.ni)):
            ax = fig.add_subplot(nrows,ncols,bin_nb+1)
            kr = np.arange(0,self.btot+self.stot)
            probs = self.get_poisson_pmf_for_bin(bin_nb=bin_nb)(kr)
            kr = kr[probs>1e-5]
            probs = probs[probs>1e-5]
            plt.plot(kr, probs, color="black", label="analytical Poisson distribution")
            plt.hist(sn[bin_nb], bins=30, density=True, color="blue", alpha=.2, label="sampled %d histograms"%n_histograms);
            plt.yticks([])
            plt.title("bin %d [%.2f, %.2f]"%(bin_nb, self.bin_edges[bin_nb], self.bin_edges[bin_nb+1]))
            plt.tight_layout()
            handles, labels = ax.get_legend_handles_labels()
            if bin_nb//ncols == nrows-1:
                plt.xlabel("nb measurements")
            
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2)

    def plot_loglikelihood(self, observed_histogram):
        rmu = np.linspace(0,2,200)
        ll = [self.loglikehood(mu=mu, observed_histogram=observed_histogram) for mu in rmu]
        plt.plot(rmu, ll)
        plt.axvline(self.mu, color="black", label="actual $\mu$")
        plt.xlabel("$\mu$");
        plt.ylabel('$\mathcal{L}(\mu)$')
        llmu = rmu[np.argmax(ll)]
        plt.axvline(llmu, color="red", label="max loglikelihood $\mu$=%.2f"%llmu)
        plt.grid(); plt.legend();        
                                