"""
Created on Mon Jul  1 16:17:06 2020
"""


import warnings
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib
import matplotlib.pyplot as plt

class fit_distribution():

    def __init__(self, data, name):
        self.data = data
        self.name =name
        
    def best_fit_distribution(self, data, dist, bins=250, ax=None):
        # Get histogram of original data
        y, x = np.histogram(data, bins=bins, density=True)
        x = (x + np.roll(x, -1))[:-1] / 2.0
    
        # Best holders
        best_distribution = st.norm
        best_params = (0.0, 1.0)
        best_sse = np.inf
    
        # Estimate distribution parameters from data
        for distribution in dist:
            
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
    
    def make_pdf(self, distribution, params, size=10000):
        """Generate distributions's Probability Distribution Function """
    
        # Separate parts of parameters
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]
    
        # Get same start and end points of distribution
        start = distribution.ppf(0.01, *arg, loc=loc, scale=scale) if arg else distribution.ppf(0.01, loc=loc, scale=scale)
        end = distribution.ppf(0.99, *arg, loc=loc, scale=scale) if arg else distribution.ppf(0.99, loc=loc, scale=scale)
    
        # Build PDF and turn into pandas Series
        x = np.linspace(start, end, size)
        y = distribution.pdf(x, loc=loc, scale=scale, *arg)
        pdf = pd.Series(y, x)
    
        return pdf
    
    
    def compare(self,dist_name):
        # unpack
        dist=[]
        for each in dist_name:
            ls = getattr(st, each)
            dist.append(ls)
  
        if len(dist) ==1:
            best_fit_name, best_fit_params = self.best_fit_distribution(self.data, dist, 250)
           
        else:
            # Plot for comparison distribution
            plt.figure(figsize=(12,8))
            ax = self.data.plot(kind='hist', bins=250, density=True, alpha=0.5)
            # Save plot limits
            dataYLim = ax.get_ylim()
            
            # Find best fit distribution
            best_fit_name, best_fit_params = self.best_fit_distribution(self.data, dist, 250, ax)
            
            # Update plots
            ax.set_ylim(dataYLim)
            ax.set_title('%s Distribution' %(self.name))
            ax.set_xlabel('$AUD')
            ax.legend(loc='upper right')
            ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('${x:,.0f}'))
            ax.set_ylabel('Probability')
        
        return best_fit_name, best_fit_params
