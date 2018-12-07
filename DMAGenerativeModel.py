from datetime import datetime, timedelta
import re
import json
from math import inf, sqrt
import pandas as pd
from scipy.special import erf
import numpy as np
import sys
from fbprophet import Prophet
sys.path.insert(0, 'utilities/')
from redshift_helper import *
from pomegranate import *



class DMAGenerativeModel():
    """
    This class applies a generative model to predict the expected lift in weighted
    minutes viewed on a per DMA level if promos are shown in the DMAs.  


    Parameters
    ----------
    shows : list
        List of shows to condsider
    
    days_out : int
        Number of days in the future the promo will be shown.

    
    Attributes
    ----------
    possible_dmas_original_ : list
        A list of possible DMAs from the original SQL query
    
    possible_dmas_filtered_ : list
        A list of DMAs after appropriate filtering
    
    ratios_ : dict
        A dictionary giving the expected lift and variance for each DMA
    
    df1_ : pandas DF
        A dataframe giving all the zero minute events
   
    df2_ : pandas DF
        A dataframe giving all the non-zero minute events
    
    fitted_vals_ : dict
        A dictionary giving the ML fitted parameters for each DMA in the form of:
        (w0, w1, w2, mu, sigma, lam, best_fit_model, (NN)/(NN+N_o))

    
    Author : Douglas Rubin

    """


    def __init__(self, shows = [], days_out = 45):
        self.shows = shows
        self.forecasts = pd.read_csv('data/all_intab_forecasts.csv', index_col=0)
        self.forecasts['ds'] = pd.to_datetime(self.forecasts['ds'])
        self.forecast_date = datetime.now().date() + timedelta(days=days_out)
        
    def __ParseSQL(self, shows):
        """
        Parse string into readable SQL commands
        """

        shows = [f"'{x}'" for x in shows]

        with open('utilities/SQL_commands.sql') as f:
            commands = [x for x in f if x[0] !='-' and x != '\n']

        commands = ''.join(commands)
        commands = commands.replace('\n', ' ')


        commands = re.sub(' +',' ', commands)
        commands = commands.replace('PROGS_TO_REPLACE', '(' + ', '.join(shows) + ')')
        commands = commands.split(';')

        commands = [x.lstrip().rstrip() for x in commands]
        commands = [x for x in commands if x != '']

        return commands
    
    
    def FetchData(self, credsfile = 'data/credentials.txt'):
        """
        Queries the redshift database to return the appropriate data
        
        Parameters
        ----------
        credsfile : str
            location of credentials

        Returns
        -------
        None

        """

        ###read in credentials
        f = open(credsfile, "r")
        creds = f.read().splitlines()
        f.close()

        ###connect to redshift
        cur, con = redshift_connect(creds[0], '\\' + creds[1], creds[2], creds[3], creds[4])

        ###get SQL commands
        commands = self.__ParseSQL(self.shows)

        ### aggregate data into temp tables on redshift

        for command in commands[0:-1]:
            cur.execute(command)

         ### read relevent data into pandas dataframes

        #### get number of zero minute events
        df1 = pd_df(cur, commands[-1])
        df1['dma_name'] = df1['dma_name'].apply(lambda x: x.lstrip().rstrip())
        self.df1_ = df1

        ### get all non-zero events
        df2 = pd_df(cur, 'select * from wmv')
        df2['dma_name'] = df2['dma_name'].apply(lambda x: x.lstrip().rstrip())
        self.df2_ = df2
        
        ### close up connections
        cur.close()
        con.close()

        possible_dmas = set(df2.dma_name.unique()).intersection(set(df1.dma_name.unique()))

        self.possible_dmas_original_ = possible_dmas.copy()


        return None
    
    
    def FilterData(self, thresh1 = 200, thresh2 = 30, thresh3 = 30):
        """
        Queries the redshift database to return the appropriate data
        
        Parameters
        ----------
        thresh1 : int
            minimum number of data points in histogram to consider for fitting
        
        thresh2 : int
            minimum number of unique PIDs that comprise the histogram to consider
            for fitting
       
        thresh3 : int
            minimum number of forecasted PIDs in DMA at time of promo placement
        
        Returns
        -------
        None

        """
        
        self.possible_dmas_filtered_ = self.possible_dmas_original_.copy()

        ### filter out DMAs by total number of data points in histogram
        DMAs_to_remove1 = set()
        for DMA in self.possible_dmas_filtered_:
            counts_wo_class_1 = len(self.df2_[self.df2_.dma_name==DMA].wmv)
            if counts_wo_class_1 < thresh1:
                DMAs_to_remove1.add(DMA)

                
        ### filter out DMAs by number of unique PIDS in the histogram
        pid_counts = self.df2_.groupby('dma_name').nunique()['pid']
        
        DMAs_to_remove2 = set()
        for DMA in self.possible_dmas_filtered_:         
            if pid_counts[DMA] < thresh2:
                DMAs_to_remove2.add(DMA)


        ### filter by forecasted number of pids in DMA
        subdf = self.forecasts[(self.forecasts.ds == self.forecast_date)]
        DMAs_to_remove3 = list(subdf[subdf.yhat<thresh3].DMA)
                
                
        ### filter 
        DMAs_to_remove = DMAs_to_remove1.union(DMAs_to_remove2).union(DMAs_to_remove3)
        for DMA in DMAs_to_remove:
            if DMA in self.possible_dmas_filtered_:
                self.possible_dmas_filtered_.remove(DMA)
                            
        return 
    
    def MLfitting(self, filter_val = 15):
        """
        Runs through all DMAs and performs the mixture model ML fits
        
        Parameters
        ----------
        filter_val  : int
            Experimentally found value.  This is the percentile of data, below which, the data is not considered in the fit
            This hacky fix fixes a lot of poorly fitted distributions

        Returns
        -------
        None

        """


        self.fitted_vals_ = {}
        for DMA in self.possible_dmas_filtered_:
            res = self.__MLFitting(DMA, filter_val)
            self.fitted_vals_[DMA] = res


        ### filter the few DMAs who do not fit this model well and thus give spurious results
        means = np.array([self.fitted_vals_[dma][3] for dma in self.fitted_vals_])
        med = np.median(means)

        dmas_to_remove = set()
        for dma in self.fitted_vals_:
            if self.fitted_vals_[dma][3] <= med/10. or self.fitted_vals_[dma][3] >= med*10.:
            	dmas_to_remove.add(dma)

        for dma in dmas_to_remove:
            self.possible_dmas_filtered_.remove(dma)
            del self.fitted_vals_[dma]


            
        return None
            
            
    def ComputeRatios(self, Delta):
        """
        Computes expectation and variance of the desired ratios for all DMAs
        """
        self.ratios_ = {}
        for DMA in self.possible_dmas_filtered_:
            res = self.__ComputeExpVar(DMA, Delta)
            self.ratios_[DMA] = res
            
        return self.ratios_
            
    
    
    def __MLFitting(self, DMA, filter_val):

        """
        Performs mixture model ML fitting for a DMA
        
        Parameters
        ----------
        DMA  : str
            The DMA to fit
        
        filter_val  : int
            Experimentally found value.  This is the percentile of data, below which, the data is not considered in the fit
            This hacky fix fixes a lot of poorly fitted distributions

        Returns
        -------
        ML fitted parameters and best model : tuple

        """

        
        thresh_val = np.percentile(self.df2_[self.df2_.dma_name == DMA]['wmv'], filter_val)
        
        N_o = self.df1_[self.df1_.dma_name==DMA].reset_index()['sum'][0]
        NN = len(self.df2_[(self.df2_.dma_name==DMA)])
        
        
        data = self.df2_[(self.df2_.dma_name==DMA) & (self.df2_.wmv >= thresh_val)].wmv.reset_index(drop=True).values[np.newaxis].T

        best_log_liklihood = -inf
        for _ in range(10):
            model = GeneralMixtureModel.from_samples([NormalDistribution, ExponentialDistribution],\
                                                     n_components=2, X=data)
            log_liklihood = sum(model.log_probability(data))
            if log_liklihood >= best_log_liklihood:
                best_log_liklihood = log_liklihood
                best_model = json.loads(model.to_json())
                best_fit_model = model

        if best_model['distributions'][0]['name'] == 'NormalDistribution': 
            mu = best_model['distributions'][0]['parameters'][0]
            sigma = best_model['distributions'][0]['parameters'][1]
            lam = best_model['distributions'][1]['parameters'][0]
            w1 = best_model['weights'][0]
            w2 = best_model['weights'][1]
        else:
            mu = best_model['distributions'][1]['parameters'][0]
            sigma = best_model['distributions'][1]['parameters'][1]
            lam = best_model['distributions'][0]['parameters'][0]
            w1 = best_model['weights'][1]
            w2 = best_model['weights'][0]

        

        w0 = N_o/(N_o+NN)
        w1 = w1*((NN)/(NN+N_o))
        w2 = w2*((NN)/(NN+N_o))

        #w0 = w_delta, w1 = w_gauss, w2 = w_exp, ..., re-weighting factor 
        return (w0, w1, w2, mu, sigma, lam, best_fit_model, (NN)/(NN+N_o))
    
    
    def __ComputeExpVar(self, DMA, Delta):
        """
        Computes expected lift and variance.
        
        Parameters
        ----------
        DMA  : str
            The DMA to fit
       
        Delta  : float
            Roughly the conversion rate


        Returns
        -------
        The expectation and variance of the desired ratios : tuple

        """

        
        
        phi1, phi2, phi3, mu, sigma, lambd, best_fit_model, f = self.fitted_vals_[DMA]
        
        res = self.forecasts[(self.forecasts.DMA == DMA) & (self.forecasts.ds == self.forecast_date)]
        
        ### take average of upper and lower bounds to compute sigma_NDMA
        E_NDMA = res['yhat'].values[0]
        sigma_NDMA = ((res['yhat'] - res['yhat_lower']).values[0]+(res['yhat_upper'] - res['yhat']).values[0])/2
        Var_NDMA = sigma_NDMA**2
        

        g1 = mu/2*(erf(mu/(sqrt(2)*sigma))+1) + sigma/(sqrt(2*np.pi))*np.exp(-mu**2/(2*sigma**2))
        g2 = ((mu**2+sigma**2)/2)*(erf(mu/(sqrt(2)*sigma))+1) + mu*sigma/(sqrt(2*np.pi))*np.exp(-mu**2/(2*sigma**2))

        EW = phi2/lambd + phi3*g1
        EW_2 = phi2*2/(lambd**2) + phi3*g2
        VarW = EW_2 - EW**2

        EWtot = E_NDMA*EW
        VarWtot = E_NDMA*VarW + Var_NDMA*(EW**2)

        EWpr = phi2*(1 - Delta)/lambd + phi3*g1 + Delta*((phi1/2)*(1/lambd+g1)+phi2*g1)
        EWpr_2 = phi2*(1 - Delta)*2/(lambd**2) + phi3*g2 + Delta*((phi1/2)*(2/(lambd**2)+g2)+phi2*g2)
        VarWpr = EWpr_2 - EWpr**2

        EWtotpr = E_NDMA*EWpr
        VarWtotpr = E_NDMA*VarWpr + Var_NDMA*(EWpr**2)

        EWWpr = 2*phi2*(1-Delta)/(lambd**2) + Delta*phi2*g1/(lambd) + phi3*g2
        CovWWpr = EWWpr - EW*EWpr
        CovWtotWtotpr = E_NDMA*CovWWpr + Var_NDMA*EWpr*EW

        ERatio = EWtotpr/EWtot - CovWtotWtotpr/(EWtot**2) + VarWtot*EWtotpr/(EWtot**3)
        VarRatio = ((EWtotpr/EWtot)**2)*(VarWtotpr/((EWtotpr)**2) - 2*CovWtotWtotpr/(EWtotpr*EWtot)+VarWtot/(EWtot**2))


        return (ERatio-1, VarRatio)

    def UpdateForecasts(self, credsfile = 'data/credentials.txt'):

        """
        Updates all time series forecast of the number of respondents in each DMA (usining facebook's prophet) and writes 
        all forecasts to CSV.  These forecasts are used in this generative model.  Note this takes about an, so should only 
        be updated occasionally.  There are no inputs and the fuction does not return anthing.  It merely computs the 
        forecasts and writes the results to 'data/all_intab_forecasts.csv'

        """

        ###read in credentials
        f = open(credsfile, "r")
        creds = f.read().splitlines()
        f.close()



        ###connect to redshift
        cur, con = redshift_connect(creds[0], '\\' + creds[1], creds[2], creds[3], creds[4])

        ### fetch data of number of people intab for all DMAs across time

        command = \
          "select date, RTRIM(dma_name) as dma_name, count(distinct(pid)) as total_intab \
            from( \
                 (select A.date, A.pid, B.household_number \
                    from dev.nielsen_in_tab A \
                  INNER JOIN dev.nielsen_market_breaks B \
                  ON A.pid = B.pid) AS C \
          INNER JOIN dev.l5m_dmas \
          ON dev.l5m_dmas.hhid =c.household_number) \
          group by date, dma_name \
          order by date, dma_name;"

        df = pd_df(cur, command)

        ### close up connections
        cur.close()
        con.close()

        df1['dma_name'] = df1['dma_name'].apply(lambda x: x.lstrip().rstrip())

        ### cut off data before 2016

        df['year'] = df['date'].apply(lambda x: x.year)
        df = df[df.year>2015]
        del df['year']

        ### use prophet to forcast all DMAs, save the figure for each forecast and save a csv of all forecasts

        forecasts = []

        for name in set(df.dma_name):

            df_to_fit = df[df.dma_name == name].reset_index(drop=True)
            del df_to_fit['dma_name']
            df_to_fit.columns = ['ds', 'y']
            m = Prophet(interval_width=0.68)
            m.fit(df_to_fit)

            future = m.make_future_dataframe(periods=365)

            forecast = m.predict(future)
            stuff = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
            stuff['DMA'] = name
            forecasts.append(stuff)
            
            
        df_empty = pd.DataFrame({'ds':[], 'yhat':[], 'yhat_lower':[], 'yhat_upper':[], 'DMA':[]})
        final_df = df_empty.append(forecasts).reset_index(drop=True)

        final_df.to_csv('data/all_intab_forecasts.csv')

        return 


def SortRatios(ratios):
    """
    Sorts the output of ComputeRatios(Delta) by expected lift
    
    Parameters
    ----------
    ratios : dict
        output of ComputeRatios(Delta)

    Returns
    -------
    The sorted ratios: tuple

    """

    ### sort them
    dmas = np.array(list(ratios.keys()))
    expt = np.array([ratios[dma][0] for dma in dmas])
    err = np.sqrt(np.array([ratios[dma][1] for dma in dmas]))

    arr1inds = list(expt.argsort())
    dmas = dmas[arr1inds[::-1]]
    expt = expt[arr1inds[::-1]]
    err = err[arr1inds[::-1]]
    expt = expt*100
    err = err*10
    
    return (expt, err, dmas)