import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KernelDensity
from scipy import stats
import pandas as pd
from sklearn.neighbors import KernelDensity
from scipy.optimize import minimize
from progressbar import progressbar as pbar
from rlxutils.optimization import coordinate_descent_minimize
import tensorflow as tf
from joblib import Parallel, delayed
import sys

class mParallel(Parallel):
    """
    substitutes joblib.Parallel with richer verbose progress information
    """
    def _print(self, msg, msg_args):
        if self.verbose > 10:
            fmsg = '[%s]: %s' % (self, msg % msg_args)
            sys.stdout.write('\r ' + fmsg)
            sys.stdout.flush()


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
                                
            
class BShist_distribution:
    
    def __init__(self, t, mu_s, sigma_s, mu, stot, btot, bin_edges, use_gaussian_appoximation=False):
        # render all params positive
        t, mu_s, sigma_s, mu, stot, btot = np.abs([t, mu_s, sigma_s, mu, stot, btot])
        stot,btot = int(stot), int(btot)
        self.t, self.mu_s, self.sigma_s, self.mu, self.stot, self.btot = t, mu_s, sigma_s, mu, stot, btot
        self.bin_edges = bin_edges
        self.n_bins = len(bin_edges)-1
        self.params = {'stot': self.stot, 'btot': self.btot, 't': self.t, 'mu_s': self.mu_s, 'sigma_s': self.sigma_s, "mu": self.mu}      
        
        # base continuous distributions
        self.b = stats.expon(scale=1/t)
        self.s = stats.norm(loc=mu_s, scale=sigma_s)            
        
        # compute distribution for each bin
        self.si = self.stot*pd.Series([self.s.cdf(i) for i in self.bin_edges]).diff().dropna().values
        self.bi = self.btot*pd.Series([self.b.cdf(i) for i in self.bin_edges]).diff().dropna().values
        if len(self.si)!=len(self.bi):
            print ("ERROR!!", self.t, self.mu_s, self.sigma_s, mu, btot, stot)
        self.ni = self.mu * self.si + self.bi  
            
        if not use_gaussian_appoximation:
            self.bins_distributions = [stats.poisson(mu=self.ni[i]).pmf for i in range(len(self.ni))]       
        else:
            self.bins_distributions = [stats.norm(loc=self.ni[i], scale=np.sqrt(self.ni[i])+1e-50).pdf for i in range(len(self.ni))]        
        
        
    def rvs(self, n_samples):
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
    
    def log_probability(self, observed_histogram):
        return np.sum([np.log(1e-100+self.bins_distributions[i](observed_histogram[i])) for i in range(len(observed_histogram))])
    
    def plog_probability(self, observed_histogram):
        """
        log probability for optimization (i.e. without the terms not depending on parameters to optimize)
        """        
        r = np.r_[[observed_histogram[i]*np.log(1e-20+self.ni[i])-self.ni[i] for i in range(len(observed_histogram))]]
        r[r==-np.inf] = np.min(r[r!=-np.inf])*100
        return np.sum(r)        
      
    def probability(self, observed_histogram):
        return np.product([self.bins_distributions[i](observed_histogram[i]) for i in range(len(observed_histogram))])     

    def plot_theoretical_distribution(self, label_prefix="", **kwargs):
        plt.plot(self.bin_edges[:-1], self.ni, label=label_prefix+"theoretical distribution", **kwargs)
    
    def plot_histogram(self, histogram):
        be = (self.bin_edges[1:]+self.bin_edges[:-1])/2
        bw = self.bin_edges[0]-self.bin_edges[1]
        plt.bar(self.bin_edges[:-1], histogram, width=bw*.9, alpha=.5, label="a sampled histogram")        
        self.plot_theoretical_distribution(color="black")
        plt.grid(); 
        plt.legend();
        plt.title("logprob of histogram %.3f"%self.log_probability(histogram))
    
    def clone(self, t=None, mu_s=None, sigma_s=None, mu=None, stot=None, btot=None):
        t       = t or self.t
        mu_s    = mu_s or self.mu_s
        sigma_s = sigma_s or self.sigma_s
        mu      = mu or self.mu
        stot    = stot or self.stot
        btot    = btot or self.btot
        return self.__class__(t, mu_s, sigma_s, mu, stot, btot, self.bin_edges)        
    
    def plot_logprob_param_sensitivity(self, observed_histogram, use_plog=False):
        param_names = list(self.params.keys())

        interval_size = .8

        plt.figure(figsize=(5*len(param_names), 3))
        for k, param in enumerate(param_names):
            plt.subplot(1,len(param_names), k+1)
            param_value = eval('self.%s'%param)
            param_range = np.linspace(param_value*(1-interval_size/2), param_value*(1+interval_size/2), 100)
            if use_plog:
                probs = [self.clone(**{param: i}).plog_probability(observed_histogram) for i in param_range]
            else:
                probs = [self.clone(**{param: i}).log_probability(observed_histogram) for i in param_range]
            plt.plot(param_range, probs)
            plt.grid();
            plt.axvline(param_value, color="red", label="true param")
            plt.legend();
            plt.title(param)
            if k==0:
                plt.ylabel("log probability")
        plt.show()    
        
        
    def get_random_params(self, observed_histogram, params_names):
        params_ranges = {"mu_s":    [self.bin_edges[0], self.bin_edges[-2]],
                         "sigma_s": [1e-3, 10],
                         "btot":    [0, np.sum(observed_histogram)],
                         "stot":    [0, np.sum(observed_histogram)],
                         "t":       [1e-3, 10],
                         "mu":      [0,10]}          
        
        def get_random_init(param_name):
            xmin, xmax = params_ranges[param_name][0], params_ranges[param_name][1]
            x = np.random.random()
            return x*(xmax-xmin)+xmin
        
        x0 = np.r_[[get_random_init(i) for i in params_names]]
        return x0
        
        
    def MLE(self, fixed_params, observed_histogram, verbose=False, **kw_minimizer_args):
        """
        computes MLE for all params except the ones listed in fixed_params
        """
        fixed_params = {k:self.params[k] for k in fixed_params}
        params_names = list(self.params.keys())

        
        params_ranges = {"mu_s":    [self.bin_edges[0], self.bin_edges[-2]],
                         "sigma_s": [1e-3, 10],
                         "btot":    [0, np.sum(observed_histogram)],
                         "stot":    [0, np.sum(observed_histogram)],
                         "t":       [1e-3, 10],
                         "mu":      [0,10]}          
        
        def get_random_init(param_name):
            xmin, xmax = params_ranges[param_name][0], params_ranges[param_name][1]
            x = np.random.random()
            return x*(xmax-xmin)+xmin
        
        def log_likelihood(varparams_values):
            assert len(varparams_values)+len(fixed_params)==len(params_names), "mismatched params"
            params = {**fixed_params, **{k:v for k,v in zip([i for i in params_names if not i in fixed_params.keys()], varparams_values)}}
            r = -BShist_distribution(**params, bin_edges=self.bin_edges).plog_probability(observed_histogram)
            return np.r_[r]

        # x0 = np.r_[[get_random_init(i) for i in params_names if not i in fixed_params.keys()]]
        x0 = self.get_random_params(observed_histogram, [i for i in params_names if not i in fixed_params.keys()])
        
        # r = minimize(log_likelihood, method="Nelder-Mead", x0=x0)

        r = coordinate_descent_minimize(log_likelihood, x0=x0, **kw_minimizer_args)
        
        optimized_params = {k:v for k,v in zip([i for i in params_names if not i in fixed_params.keys()], r.x)}
    
        if verbose:
            print ("fixed params    ", fixed_params)
            print ("allowable paramss ranges ", params_ranges)
            print ("init params     ", {k:v for k,v in zip([i for i in params_names if not i in fixed_params.keys()], x0)})
            print ("optimized params", optimized_params)
            print ("expected params ", {i:self.params[i] for i in optimized_params.keys()})
            print ()
            print (r)
        return optimized_params, self.clone(**{**fixed_params, **optimized_params})        
                    
        
sqrt2 = tfpi = tf.constant(np.sqrt(2), name="sqrt2", dtype=tf.float64)
tfpi = tf.constant(np.pi, name="pi", dtype=tf.float64)

gaus_pdf_tf = lambda x, mu, sigma: tf.math.exp(-0.5*((x-mu)/sigma)**2)/(sigma*tf.sqrt(2*tfpi))    
gaus_cdf_tf = lambda x, mu, sigma: 0.5*(1+tf.math.erf((x-mu)/(sigma*sqrt2)))

from scipy import special
gaus_pdf_np = lambda x, mu, sigma: np.exp(-0.5*((x-mu)/sigma)**2)/(sigma*np.sqrt(2*tfpi))    
gaus_cdf_np = lambda x, mu, sigma: 0.5*(1+special.erf((x-mu)/(sigma*sqrt2)))

exp_pdf_tf  = lambda x, t: t * tf.math.exp(-t*x)   
exp_cdf_tf  = lambda x, t: 1 - tf.math.exp(-t*x)       
    
exp_pdf_np  = lambda x, t: t * np.exp(-t*x)   
exp_cdf_np  = lambda x, t: 1 - np.exp(-t*x)       



class MLE_SignalBg:
    """
    Tensorflow implementation
    """
    
    def __init__(self, params = None, use_tf=False):
        self.params = self.get_random_params() if params is None else np.r_[params]

        t, mu_s, sigma_s, mu = self.params        
        self.b = stats.expon(scale=1/t)
        self.s = stats.norm(loc=mu_s, scale=sigma_s)     
        self.use_tf = use_tf

    def get_params(self, mu=None):
        """
        returns current params, possibly substituting mu
        """
        r = self.params
        r[-1] = mu if mu is not None else r[-1]
        return r  
    
    def get_sb_probs(self, mu):
        """
        scales p_s by mu and then normalizes with the corresponding complementary p_b
        """
        p_s = mu
        p_b = 1.
        z = p_s + p_b
        p_s = p_s/z
        p_b = p_b/z

        return p_s, p_b
    
    def unpack_params(self, params=None):
        params = self.params if params is None else params
        t, mu_s, sigma_s, mu = params 
        t = tf.abs(t)
        mu = tf.abs(mu)
        sigma_s = tf.abs(sigma_s)
        p_s, p_b = self.get_sb_probs(mu)
        return t, mu_s, sigma_s, mu, p_s, p_b
    
    def rvs(self, n):
        t, mu_s, sigma_s, mu, p_s, p_b = self.unpack_params(self.params)     
        
        xb = self.b.rvs(int(n*p_b))
        xs = self.s.rvs(int(n*p_s))
        x = np.random.permutation(np.concatenate((xb,xs)))
        return x.astype(np.float64)
    
    def build_ts_distribution(self, n_events, n_experiments, show_pbar=True):
        from progressbar import progressbar as pbar

        pbar = pbar if show_pbar else list
        s  = [self.rvs(n_events) for _ in range(n_experiments)]
        self.ts = np.r_[[-self.likelihood(i) for i in pbar(s)]]

    def plot_ts_distribution(self, alpha=0.05, ts_val=None):
        assert "ts" in dir(self), "must call first build_ts_distribution"
        plt.hist(self.ts, bins=30, density=True, alpha=.5);
        plt.title("$t_s$ distribution")
        plt.grid(); plt.xlabel("likelihood"); plt.ylabel("probability");
        if alpha is not None:
            lim = np.percentile(self.ts, 100*(1-alpha))
            plt.axvline(lim, color="black", ls="--", label="alpha=%.2f @ ts=%.3f confidence interval"%(alpha, lim))

        if ts_val is not None:
            plt.axvline(ts_val, color="red", label="$t_s$ for our data")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    def get_pvalue(self, x):
        assert "ts" in dir(self), "must call first build_ts_distribution"

        t = self.neg_likelihood(x)
        return np.mean(t<self.ts)    
    
    
    def plot_sample(self, x, bins=30):

        # exclude small percentile larger to make nicer plot
        x = x[x<np.percentile(x, 99.5)]
        
        t, mu_s, sigma_s, mu, p_s, p_b = self.unpack_params(self.params)     
        plt.hist(x, bins=bins, alpha=.5, density=True);

        xr = np.linspace(np.min(x), np.max(x), 100).astype(np.float64)
        plt.plot(xr, np.exp(self.logprob(xr)), color="red", label="analytical pdf")
        plt.legend(); plt.grid();
        plt.title("$t$=%.3f  ::  $\mu_s$=%.3f  ::  $\sigma_s$=%.3f  ::  $\mu$=%.3f"%(t, mu_s, sigma_s, mu))
        
    def get_random_params(self):
        params_ranges = {"mu_s":    [10, 100],
                         "sigma_s": [1e-3, 10],
                         #"p_s":     [-5, 5], # for sigmoid
                         "t":       [1e-3, 10],
                         "mu":      [0,.001]}          
        
        def get_random_init(param_name):
            xmin, xmax = params_ranges[param_name][0], params_ranges[param_name][1]
            x = np.random.random()
            return x*(xmax-xmin)+xmin
        
        x0 = np.r_[[get_random_init(i) for i in ["t", "mu_s", "sigma_s", "mu"]]]
        
        return x0  

    def lprobs(self, x, params=None):
        params = self.params if params is None else params
        t, mu_s, sigma_s, mu, p_s, p_b = self.unpack_params(params)     
        r1 = gaus_pdf_tf(x, mu_s, sigma_s) if self.use_tf else gaus_pdf_np(x, mu_s, sigma_s)
        r2 = exp_pdf_tf(x, t) if self.use_tf else exp_pdf(x, t)
        return r1, r2
    
    
    def logprob(self, x, params=None):
        params = self.params if params is None else params
        t, mu_s, sigma_s, mu, p_s, p_b = self.unpack_params(params)     
        if self.use_tf:
            r = tf.math.log(p_s * gaus_pdf_tf(x, mu_s, sigma_s) + p_b*exp_pdf_tf(x, t) + 1e-100)
        else:            
            r = np.log(p_s * gaus_pdf_np(x, mu_s, sigma_s) + p_b*exp_pdf_np(x, t) + 1e-100)

        return r
    
    def init_optimizer(self, fixed_params=[], **optimizer_params):
        rt, rmu_s, rsigma_s, rmu = self.get_random_params()                

        self.fixed_params = fixed_params
        
        self.init_params = [self.params[0] if "t" in fixed_params else rt,
                            self.params[1] if "mu_s" in fixed_params else rmu_s,
                            self.params[2] if "sigma_s" in fixed_params else rsigma_s,
                            self.params[3] if "mu" in fixed_params else rmu]                
        
        self.tt       = tf.Variable(initial_value=self.init_params[0], name="t", dtype=tf.float64)
        self.tmu_s    = tf.Variable(initial_value=self.init_params[1], name="mu_s", dtype=tf.float64)
        self.tsigma_s = tf.Variable(initial_value=self.init_params[2], name="sigma_s", dtype=tf.float64)
        self.tmu      = tf.Variable(initial_value=self.init_params[3], name="mu", dtype=tf.float64)

        self.tparams = [self.tt, self.tmu_s, self.tsigma_s, self.tmu]
        
        self.free_params = []
        if not "t" in fixed_params:
            self.free_params.append(self.tt)
        if not "mu_s" in fixed_params:
            self.free_params.append(self.tmu_s)
        if not "sigma_s" in fixed_params:
            self.free_params.append(self.tsigma_s)
        if not "mu" in fixed_params:
            self.free_params.append(self.tmu)        
        
        self.optimizer = tf.optimizers.Adam(**optimizer_params)

        @tf.function
        def train_step(x):
            
            """
            loss = lambda x, params: -self.logprob(x,params)
            with tf.GradientTape() as tape:
                loss_value = tf.reduce_mean(loss(x,self.tparams))
            """
            with tf.GradientTape() as tape:
                loss_value = -self.likelihood(x, self.tparams)                
                
            grads = tape.gradient(loss_value, self.free_params)
            self.optimizer.apply_gradients(zip(grads, self.free_params))

            return loss_value
        
        self.train_step = train_step
        
        self.history = []
        
    def log_likelihood(self, x, params=None):
        if self.use_tf:
            r = tf.reduce_mean(self.logprob(x,params))
        else:
            r = np.mean(self.logprob(x,params))
        return r
        
    def fit(self, x, n_steps=5000, use_pbar=False):
        assert self.use_tf, "must set use_tf=True"
        self.history_lastbatch = []
        it = range(n_steps)
        it = pbar(it) if use_pbar else it
        for epoch in it:
            loss_value = self.train_step(x)
            if np.isnan(loss_value):
                break
            self.history_lastbatch.append(loss_value.numpy())
            self.history.append(loss_value.numpy())

    def get_mu_hat(self, x):
        """
        returns the MLE for mu, assuming the rest of the params fixed
        """
        if self.use_tf:
            self.init_optimizer(learning_rate=.01, fixed_params=["t", "mu_s", "sigma_s"])
            self.fit(x, n_steps=50)
            r = self.tparams[-1].numpy()
        else:
            f = lambda mu: -self.log_likelihood(x, self.get_params(mu=mu))
            r = minimize(f, 1).x[0]
        return r

    def get_t_mu(self, x, mu):
        """
        t_mu = -2 * lkhood(mu)/lkhood(mu_hat)
        mu_hat is the MLE for mu
        """
        lmu = self.log_likelihood(x, params=self.get_params(mu=mu),)
        lmu_hat = self.log_likelihood(x, params=self.get_params(mu=self.get_mu_hat(x)))
        return -2 * (lmu - lmu_hat)

    def print_values(self):
        print ("   param     obtained  true_value")
        for a,b in zip(self.tparams, self.params):
            print ("%8s"%a.name.split(":")[0], "    %8.3f  %8.3f"%(a.numpy(), b))   
            
            
class MLE_BinarizedSignalBg(MLE_SignalBg):
    """
    Tensorflow implementation
    """
    
    def __init__(self, n_events, params = None, bin_edges = None):
        self.params = self.get_random_params() if params is None else np.r_[params]

        t, mu_s, sigma_s, mu = self.params        
        self.b = stats.expon(scale=1/t)
        self.s = stats.norm(loc=mu_s, scale=sigma_s)       
    
        if bin_edges is None:
            self.n_bins = 20
            self.bin_edges = np.r_[list(np.linspace(0,180,self.n_bins))+[np.inf]]
        else:
            self.bin_edges = bin_edges
            self.n_bins = len(bin_edges)-1        
    
        self.n_events = n_events
    
    def hrvs_c(self, n):
        """
        returns histogram samples from sampling continuous distribution
        """
        r = np.r_[[np.histogram(self.rvs(self.n_events), bins=self.bin_edges)[0].astype(np.float64) \
                   for _ in range(n)]]
        return r
    
    def hrvs_b(self, n):
        """
        sampling histograms directly from bins poissons
        """
        _,_,_, poisson_i = self.get_bins_poisson_distributions()
        r = np.r_[[p.rvs(n) for p in poisson_i]].T
        return r
            
    def plot_sample(self, hc, hb):
        """
        hc is a sample from hrvs_c
        hb is a sample from hrvs_b
        """
        plt.figure(figsize=(10,3))
        plt.subplot(121)
        plt.plot(hc.mean(axis=0), label="from continuous disitribution")
        plt.plot(hb.mean(axis=0), label="from bins distributions")
        plt.grid();
        plt.legend()
        plt.ylabel("bin mean")

        plt.subplot(122)
        plt.plot(hc.std(axis=0), label="from continuous disitribution")
        plt.plot(hb.std(axis=0), label="from bins distributions")
        plt.grid();
        plt.legend()
        plt.ylabel("bin std")    
    
    def get_expected_ni(self, params=None):
        params = self.params if params is None else params
        t, mu_s, sigma_s, mu, p_s, p_b = self.unpack_params(params) 
        
        be = self.bin_edges
        bi = p_b*(exp_cdf_tf(be[1:],t)-exp_cdf_tf(be[:-1],t))
        si = p_s*(gaus_cdf_tf(be[1:],mu_s, sigma_s)-gaus_cdf_tf(be[:-1],mu_s, sigma_s))
        ei = (bi+si)*self.n_events
        return bi,si,ei
    
    def logprob(self, ni, params=None):
        params = self.params if params is None else params
        t, mu_s, sigma_s, mu, p_s, p_b = self.unpack_params(params) 
        bi,si,ei = self.get_expected_ni(params=params)

        prob_i = tf.map_fn(lambda x: ((x[0]*tf.math.log(x[1]+1e-100)) - \
                                      tf.reduce_sum(tf.math.log(tf.range(1,x[0]+1)+1e-100)) - x[1]),
                           tf.stack((ni,ei), axis=1))        
        return prob_i
    
    def slogprob(self, ni, params=None):
        """
        just to check, logprob using scipy.stats
        """
        params = self.params if params is None else params
        t, mu_s, sigma_s, mu, p_s, p_b = self.unpack_params(params) 
        be = self.bin_edges

        exp = stats.expon(scale=1/t)
        gaus = stats.norm(loc=mu_s, scale=sigma_s)
        
        bi = p_b*(exp.cdf(be[1:])-exp.cdf(be[:-1]))
        si = p_s*(gaus.cdf(be[1:])-gaus.cdf(be[:-1]))
        ei = (bi+si)*self.n_events
        
        pi = [stats.poisson(_e) for _e in ei]
        prob_i = [np.log(p.pmf(n)) for p,n in zip(pi,ni)]
        return prob_i
    
    def get_bins_poisson_distributions(self, params=None):
        
        # analytics Poisson distribution for each bin
        params = self.params if params is None else params
        t, mu_s, sigma_s, mu, p_s, p_b = self.unpack_params(self.params) 
        be = self.bin_edges

        exp = stats.expon(scale=1/t)
        gaus = stats.norm(loc=mu_s, scale=sigma_s)
        
        bi = p_b*(exp.cdf(be[1:])-exp.cdf(be[:-1]))
        si = p_s*(gaus.cdf(be[1:])-gaus.cdf(be[:-1]))
        ei = (bi+si)*self.n_events
        
        poisson_i = [stats.poisson(_e) for _e in ei] 
        return bi,si,ei,poisson_i
        
      
        
    def likelihood(self, ni, params=None):
        return tf.reduce_sum(self.logprob(ni,params))
        
    def neg_likelihood(self, x, params=None):
        return -self.likelihood(x, params)

    def plot_bins_histograms(self, hc, hb):
                
        ncols=5
        nrows = np.ceil(self.n_bins / ncols).astype(int)

        plt.figure(figsize=(3*ncols, 3*nrows))

        nr = np.r_[[np.linspace(np.min(hb[:,k]), np.max(hb[:,k])+1,100).astype(int) for k in range(self.n_bins)]]
        p = np.r_[[np.exp(self.slogprob(n)) for n in nr.T]].T

        for k in range(self.n_bins):
            plt.subplot(nrows, ncols,k+1)
            plt.hist(hc[:,k], bins=30, density=True, alpha=.2, label="cont", color="red");
            plt.hist(hb[:,k], bins=30, density=True, alpha=.2, label="bins", color="blue");
            plt.plot(nr[k], p[k], color="black", label="Poisson")
            plt.grid(); plt.legend();
            plt.axvline(np.mean(hc[:,k]), color="black", alpha=.8)
            plt.axvline(np.mean(hb[:,k]), color="black", alpha=.8)
            plt.title("cont $\mu$=%.2f, $\sigma$=%.3f"%(np.mean(hc[:,k]), np.std(hc[:,k]))+
                      "\nbins $\mu$=%.2f, $\sigma$=%.3f"%(np.mean(hb[:,k]), np.std(hb[:,k])))
        plt.tight_layout()
        plt.show()        
