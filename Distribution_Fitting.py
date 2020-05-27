"""
Created on Mon Jul  1 16:17:06 2019
@author: marcus
"""


import warnings
import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.stats import rv_continuous
import matplotlib
import matplotlib.pyplot as plt
from scipy.special import gamma

class fit_distribution():

    def __init__(self, data, name):
        self.data = data
        self.name =name
        
    def best_fit_distribution(self, data, DISTRIBUTIONS, bins=250, ax=None):
        # Get histogram of original data
        y, x = np.histogram(data, bins=bins, density=True)
        x = (x + np.roll(x, -1))[:-1] / 2.0
    
        # Distributions to check
    # =============================================================================
    #     DISTRIBUTIONS = [        
    #         st.alpha,st.anglit,st.arcsine,st.beta,st.betaprime,st.bradford,st.burr,st.cauchy,st.chi,st.chi2,st.cosine,
    #         st.dgamma,st.dweibull,st.erlang,st.expon,st.exponnorm,st.exponweib,st.exponpow,st.f,st.fatiguelife,st.fisk,
    #         st.foldcauchy,st.foldnorm,st.frechet_r,st.frechet_l,st.genlogistic,st.genpareto,st.gennorm,st.genexpon,
    #         st.genextreme,st.gausshyper,st.gamma,st.gengamma,st.genhalflogistic,st.gilbrat,st.gompertz,st.gumbel_r,
    #         st.gumbel_l,st.halfcauchy,st.halflogistic,st.halfnorm,st.halfgennorm,st.hypsecant,st.invgamma,st.invgauss,
    #         st.invweibull,st.johnsonsb,st.johnsonsu,st.ksone,st.kstwobign,st.laplace,st.levy,st.levy_l,st.levy_stable,
    #         st.logistic,st.loggamma,st.loglaplace,st.lognorm,st.lomax,st.maxwell,st.mielke,st.nakagami,st.ncx2,st.ncf,
    #         st.nct,st.norm,st.pareto,st.pearson3,st.powerlaw,st.powerlognorm,st.powernorm,st.rdist,st.reciprocal,
    #         st.rayleigh,st.rice,st.recipinvgauss,st.semicircular,st.t,st.triang,st.truncexpon,st.truncnorm,st.tukeylambda,
    #         st.uniform,st.vonmises,st.vonmises_line,st.wald,st.weibull_min,st.weibull_max,st.wrapcauchy
    #     ]
    # =============================================================================
     
        # Common Distributions for Insurance claims cost:          
    # =============================================================================
    #         DISTRIBUTIONS = [ st.expon, st.beta, st.burr, st.halfnorm, st.pareto, st.genpareto, st.invweibull, st.gamma,
    #                           st.gengamma, st.uniform, st.norm]
    # =============================================================================
                   
        # Best holders
        best_distribution = st.norm
        best_params = (0.0, 1.0)
        best_sse = np.inf
    
        # Estimate distribution parameters from data
        for distribution in DISTRIBUTIONS:
            
            # Try to fit the distribution
            try:
                # Ignore warnings from data that can't be fit
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
    
                    # fit dist to data
                    params = distribution.fit(data)
    
                    # Separate parts of parameters
                    arg = params[:-2]
                    loc = params[-2]
                    scale = params[-1]
    
                    # Calculate fitted PDF and error with fit in distribution
                    pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                    sse = np.sum(np.power(y - pdf, 2.0))
    
                    # if axis pass in add to plot
                    try:
                        if ax:
                            pd.Series(pdf, x).plot(ax=ax,label=distribution.name)
                        end
                    except Exception:
                        pass
    
                    # identify if this distribution is better
                    if best_sse > sse > 0:
                        best_distribution = distribution
                        best_params = params
                        best_sse = sse
    
            except Exception:
                pass
    
        return (best_distribution.name, best_params)
    
    def make_pdf(self, dist, params, size=10000):
        """Generate distributions's Probability Distribution Function """
    
        # Separate parts of parameters
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]
    
        # Get sane start and end points of distribution
        start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
        end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)
    
        # Build PDF and turn into pandas Series
        x = np.linspace(start, end, size)
        y = dist.pdf(x, loc=loc, scale=scale, *arg)
        pdf = pd.Series(y, x)
    
        return pdf
    
    
    def compare(self,dist):
        # unpack
        DISTRIBUTIONS=[]
        for each in dist:
            ls = getattr(st, each)
            DISTRIBUTIONS.append(ls)
  
        if len(DISTRIBUTIONS) ==1:
            best_fit_name, best_fit_params = self.best_fit_distribution(self.data, DISTRIBUTIONS, 250)
           
        else:
            # Plot for comparison distribution
            plt.figure(figsize=(12,8))
            ax = self.data.plot(kind='hist', bins=250, density=True, alpha=0.5)
            # Save plot limits
            dataYLim = ax.get_ylim()
            
            # Find best fit distribution
            best_fit_name, best_fit_params = self.best_fit_distribution(self.data, DISTRIBUTIONS, 250, ax)
            
            # Update plots
            ax.set_ylim(dataYLim)
            ax.set_title('%s Distribution' %(self.name))
            ax.set_xlabel('$AUD')
            ax.legend(loc='upper right')
            ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('${x:,.0f}'))
            ax.set_ylabel('Probability')
        
        return best_fit_name, best_fit_params
