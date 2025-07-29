# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2025-07-16 14:52:34
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2025-07-24 10:23:42

import numpy as np
from matplotlib.pyplot import *
from itertools import product
from scipy.stats import pearsonr
import pandas as pd
from model import Model

class ParameterSweep:

    def __init__(self, base_params_wake=None, base_params_opto=None):
        """
        Generic parameter sweep class
        
        Parameters:
        -----------
        base_params_wake : dict
            Base parameters for wake model (default: I_lmn=1.0, D_lmn=0.1)
        base_params_opto : dict
            Base parameters for opto model (default: I_lmn=0.0, w_psb_lmn_=0.0)
        """
        self.base_params_wake = base_params_wake or {'I_lmn': 1.0, 'D_lmn': 0.1}
        self.base_params_opto = base_params_opto or {'I_lmn': 0.0, 'w_psb_lmn_': 0.0}
        self.results = None
    
    def sweep(self, param_dict):
        """
        Perform parameter sweep
        
        Parameters:
        -----------
        param_dict : dict
            Dictionary with parameter names as keys and lists of values as values
            Example: {'w_adn_trn': [0.1, 0.2, 0.3], 'beta_adn': [0.5, 1.0, 1.5]}
        
        Returns:
        --------
        tuple : (adn_scores, lmn_scores) as numpy arrays with shape matching param_dict dimensions
        """
        # Get parameter values and shapes
        param_names = list(param_dict.keys())
        param_values = list(param_dict.values())
        param_shapes = [len(values) for values in param_values]
        
        # Initialize score arrays
        adn_scores = np.full(param_shapes, np.nan)
        lmn_scores = np.full(param_shapes, np.nan)
        
        total = np.prod(param_shapes)
        print(f"Starting sweep with {total} parameter combinations...")
        
        count = 0
        # Use np.ndindex to iterate over all indices
        for idx in np.ndindex(*param_shapes):
            print(count)
            if count % 10 == 0:
                print(f"Progress: {count}/{total}")
            count += 1
            
            # Get current parameter values for this index
            current_params = {}
            for i, name in enumerate(param_names):
                current_params[name] = param_values[i][idx[i]]
            
            try:
                # Create models with current parameters
                wake_params = {**self.base_params_wake, **current_params}
                opto_params = {**self.base_params_opto, **current_params}
                
                m_wake = Model(**wake_params)
                m_wake.run()
                
                m_opto = Model(**opto_params)
                m_opto.run()
                
                # Calculate scores (same as original code)
                scores = self._calculate_scores(m_wake, m_opto)
                
                # Store results at current index
                adn_scores[idx] = scores['adn']
                # lmn_scores[idx] = scores['lmn']
                
            except Exception as e:
                print(f"Error with {current_params}: {e}")
                # NaN values already initialized
                pass
        
        print("Sweep completed!")
        return adn_scores
    
    def _calculate_scores(self, m_wake, m_opto):
        """Calculate correlation scores (same as original code)"""
        popcoh = {}
        # for k in ['adn', 'lmn']:    
        for k in ['adn']:
            popcoh[k] = {}
            for n, m in zip(['wak', 'opto'], [m_wake, m_opto]):
                tmp = np.corrcoef(getattr(m, f"r_{k}").T)
                popcoh[k][n] = tmp[np.triu_indices(tmp.shape[0], 1)]
        
        scores = {}
        # for st in ['adn', 'lmn']:
        for st in ['adn']:
            r, p = pearsonr(popcoh[st]['wak'], popcoh[st]['opto'])
            scores[st] = r
        
        return scores



# Example usage:
if __name__ == "__main__":
    # Create sweep instance
    sweep = ParameterSweep()
    
    # Define parameters to sweep
    params = {
        'w_trn_adn_': np.linspace(0.0, 0.1, 10),
        # 'beta_adn': [1, 5, 10, 15]
        "w_lmn_adn_" : np.linspace(0.9, 2.0, 3)
    }
    
    # Run sweep
    results= sweep.sweep(params)
    
    figure()
    for i, v in enumerate(params['w_lmn_adn_']):
        plot(params['w_trn_adn_'], results[:,i], 'o-', label=f"w_lmn_adn_={v}")

    axhline(0.4)
    legend()
    xlabel("W trn -> adn")
    show()





