import numpy as np
import os
import time
import sys
from math import log10, floor, ceil

class EmptyDistribution():

    """Documentation for class EmptyDistribution

    Parent object. The object will not have the same types of attributes depending 
    on the choice of distribution structure.
    
    """
    
    def __init__(self,bintype='linear',nbpd=10,nppb=4,nbins=50,nd=4,fill_last_decade=False):

        """Constructor for class EmptyDistribution.
        Arguments:
        - bintype [linear, log, invlogQ, linQ]: bin structure.
        - nbins: 'number of linear bins' used for all types of statistics. Default is 50.
        - nbpd: number of bins per (log or invlog) decade. Default is 10.
        - nppb: minimum number of data points per bin. Default is 4.
        - nd: (maximum) number of decades for inverse-logarithmic bins. Default is 4.
        - fill_last_decade: boolean to fill up largest percentiles for 'invlog' bin type
        """

        self.bintype = bintype
        self.nbins = nbins
        self.nbpd = nbpd
        self.nppb = nppb
        self.nd = nd
        self.fill_last_decade = fill_last_decade

        # Remove unnecessary attributes
        if self.bintype == 'linear':

            self.nbpd = None
            self.nppb = None
            self.fill_last_decade = None

        elif self.bintype in ['log','invlogQ']:

            # self.nbins = None
            pass

        elif self.bintype == 'linQ':

            self.nlb = None
            self.nbpd = None
            self.nppb = None
            self.fill_last_decade = None

        else:

            raise Exception("ERROR: unknown bintype")
        
    def __repr__(self):
        """Creates a printable version of the Distribution object. Only prints the 
        attribute value when its string fits is small enough."""

        out = '< Distribution object:\n'
        # print keys
        for k in self.__dict__.keys():
            out = out+' . %s: '%k
            if sys.getsizeof(getattr(self,k).__str__()) < 80:
                # show value
                out = out+'%s\n'%str(getattr(self,k))
            else:
                # show type
                out = out+'%s\n'%getattr(self,k).__class__
        out = out+' >'

        return out
    
    def __str__(self):
        """Override string function to print attributes
        """
        # method_names = []
        # str_out = '-- Attributes --'
        str_out = ''
        for a in dir(self):
            if '__' not in a:
                a_str = str(getattr(self,a))
                if 'method' not in a_str:
                    str_out = str_out+("%s : %s\n"%(a,a_str))
        #         else:
        #             method_names.append(a)
        # print('-- Methods --')
        # for m in method_names:
        #     print(m)
        return str_out


class Distribution(EmptyDistribution):
    """     
    """

    def __init__(self,name='',bintype='linear',nbpd=10,nppb=4,nbins=50,nd=None,\
        fill_last_decade=False,distribution=None,overwrite=False):
        """Constructor for class Distribution.
        Arguments:
        - name: name of reference variable
        - bintype [linear, log, invlog]: bin structure,
        - nbins: number of bins used for all types of statistics. Default is 50.
        - nbpd: number of bins per log or invlog decade. Default is 10.
        - nppb: minimum number of data points per bin. Default is 4.
        - nd: maximum number of decades in invlogQ bin type. Default is 4
        """

        EmptyDistribution.__init__(self,bintype,nbpd,nppb,nbins,nd,fill_last_decade)
        self.name = name
        self.size = 0
        self.vmin = None
        self.vmax = None
        self.rank_edges = None
        self.ranks = None
        self.percentiles = None
        self.bins = None
        self.density = None
        self.bin_locations_stored = False
        self.overwrite = overwrite

        if distribution is not None: # then copy it in self
            for attr in distribution.__dict__.keys():
                setattr(self,attr,getattr(distribution,attr)) 
    
    def __repr__(self):
        """Creates a printable version of the Distribution object. Only prints the 
        attribute value when its string fits is small enough."""
        return super().__repr__()

    def __str__(self):
        return super().__str__()
    
    def set_sample_size(self,sample):

        if sample.size == 0:
            raise Exception("")
        else:
            self.size = sample.size

    def set_bounds(self,sample=None,vmin=None,vmax=None,minmode='positive',\
        overwrite=False):

        """Compute and set minimum and maximum values
        Arguments:
        - sample: 1D numpy array of data values."""

        # Find minimum value
        if vmin is None:	
            if minmode is None:
                vmin = np.nanmin(sample)
            elif minmode == 'positive':
                vmin = np.nanmin(sample[sample > 0])
        # Find maximum value
        if vmax is None:
            vmax = np.nanmax(sample)
            
        if self.vmin is None or overwrite:
            self.vmin = vmin
        if self.vmax is None or overwrite:
            self.vmax = vmax

    def get_inv_log_ranls(self,out=False):

        """Percentile ranks regularly spaced on an inverse-logarithmic axis (zoom on 
        largest percentiles of the distribution).
        Arguments:
            - fill_last_decade: True (default is False) if want to plot
            up to 99.99 or 99.999, not some weird number in the middle of a decade.
        Sets:
            - ranks: 1D numpy.array of floats"""

        # k indexes bins
        if self.nd is None:
            n_decades = log10(self.size/self.nppb) 		# Number of decades from data size
        else:
            n_decades = self.nd                        # Prescribed number of decades
        dk = 1/self.nbpd
        if self.fill_last_decade:
            k_max = floor(n_decades)				 	# Maximum bin index
        else:
            k_max = int(n_decades*self.nbpd)*dk # Maximum bin index
        scale_invlog = np.arange(0,k_max+dk,dk)
        ranks_invlog = np.subtract(np.ones(scale_invlog.size),
            np.power(10,-scale_invlog))*100

        # store ranks
        self.ranks = ranks_invlog
        # calculate bin edges in rank-space
        self.rank_edges = np.hstack([[0],np.convolve(self.ranks,[0.5,0.5],mode='valid'),[None]])
        # get number of bins
        self.nbins = self.ranks.size # in this case, define nbins from - no no no no noooo, recode this
        
        if out:
            return ranks_invlog
        
    def get_lin_ranks(self):

        """Percentile ranks regularly spaced on a linear axis of percentile ranks"""

        self.rank_edges = np.linspace(0,100,self.nbins+1) # need nbins as input
        self.ranks = np.convolve(self.rank_edges,[0.5,0.5],mode='valid') # center of rank 'bins'

    def compute_percentiles_and_bins_from_ranks(self,sample,crop=False,store=True,output=False):

        """Compute percentiles of the distribution and histogram bins from 
        percentile ranks. 
        Arguments:
            - sample: 1D numpy array of values
            - ranks: 1D array of floats between 0 and 1
        Sets:
            - ranks, cropped by one at beginning and end
            - percentiles (or bin centers)
            - bins (edges)
        """

        sample_no_nan = sample[np.logical_not(np.isnan(sample))]
        if sample_no_nan.size == 0:
            percentiles = np.array([np.nan]*self.nbins)
        else:
            percentiles = np.percentile(sample_no_nan,list(self.ranks))
            
        # calculate center bins (not minimum edge nor maximum edge)
        bins = np.array([np.nan]*(self.nbins+1))
        bins[1:-1] = np.percentile(sample_no_nan,list(self.rank_edges[1:-1]))

        if not crop:
            bins[0] = self.vmin
            bins[-1] = self.vmax

        if store:
            self.percentiles = percentiles
            self.bins = bins
        
        if output:
            return self.percentiles, self.bins

    def define_percentiles_on_inv_log_q(self,sample):

        """Defines percentiles and histogram bins on inverse-logarithmic ranks.
        Arguments:
            - sample: 1D numpy array of values
        Sets:
            - ranks
            - percentiles
            - bins
        """
        
        self.size = sample.size
        # First compute invlog ranks including its edge values
        self.get_inv_log_ranls()
        # Then compute final stats
        self.compute_percentiles_and_bins_from_ranks(sample) # keep crop=False to get manually-set bounds

    def define_percentiles_on_lin_q(self,sample,vmin=None,vmax=None):

        """Define percentiles and histogram bins on linear ranks.
        Arguments:
            - sample: 1D numpy array of values
        Sets:
            - ranks
            - percentiles
            - bins
        """

        self.set_bounds(sample=sample,vmin=vmin,vmax=vmax)
        # Compute linear ranks
        self.get_lin_ranks()
        # Then compute final stats
        self.compute_percentiles_and_bins_from_ranks(sample)

    def define_log_bins(self,sample,vmin=None,vmax=None,minmode='positive'):

        """Define logarithmic bin centers and edges from sample values.
        Arguments:
            - sample: 1D numpy array of values
            - n_bins_per_decade: number of ranks/bins per logarithmic decade
            - vmin and vmax: extremum values
        Computes:
            - centers (corresponding percentiles, or bin centers)
            - breaks (histogram bin edges)"""

        self.set_bounds(sample,vmin,vmax,minmode)
        kmin = floor(log10(self.vmin))
        kmax = ceil(log10(self.vmax))
        self.bins = np.logspace(kmin,kmax,(kmax-kmin)*self.nbpd)
        self.percentiles = np.convolve(self.bins,[0.5,0.5],mode='valid')
        self.nbins = self.percentiles.size

    def define_linear_bins(self,sample,vmin=None,vmax=None,minmode='positive'):

        """Define linear bin centers and edges from sample values.
        Arguments:
            - sample: 1D numpy array of values
            - vmin and vmax: extremum values
        Computes:
            - percentiles (or bin centers)
            - bins (edges)
        """

        self.set_bounds(sample,vmin,vmax,minmode)
        self.bins = np.linspace(self.vmin,self.vmax,self.nbins+1)
        self.percentiles = np.convolve(self.bins,[0.5,0.5],mode='valid')
        
        assert(self.percentiles.size == self.nbins), "wrong number of bins: #(percentiles)=%d and #(bins)=%d"%(self.percentiles.size,self.nbins)

    def compute_percentile_ranks_from_binsnear_bins(self,sample):

        """Computes percentile ranks corresponding to percentile values.
        Arguments:
            - sample: 1D numpy array of values
        Computes:
            - ranks: 1D numpy.ndarray"""
        
        self.ranks = 100*np.array(list(map(lambda x:(sample < x).sum()/self.size, \
            self.percentiles)))

    def ranks_percentiles_and_bins(self,sample,vmin=None,vmax=None,minmode='positive'):

        """Preliminary step to compute probability densities. Define 
        ranks, percentiles, bins from the sample values and binning structure.
        Arguments:
            - sample: 1D numpy array of values
        Computes:
            - ranks, percentiles and bins"""

        self.set_sample_size(sample)
        self.set_bounds(sample,vmin,vmax,minmode)

        if self.bintype == 'linear':

            self.define_linear_bins(sample,vmin,vmax,minmode)
            self.compute_percentile_ranks_from_binsnear_bins(sample)

        elif self.bintype == 'log':

            self.define_log_bins(sample,vmin,vmax,minmode)
            self.compute_percentile_ranks_from_binsnear_bins(sample)

        elif self.bintype == 'invlogQ':

            self.get_inv_log_ranls()
            self.compute_percentiles_and_bins_from_ranks(sample)

        elif self.bintype == 'linQ':

            self.define_percentiles_on_lin_q(sample)

        else:

            raise Exception("ERROR: unknown bintype")

    def compute_distribution(self,sample,vmin=None,vmax=None,minmode=None):

        """Compute ranks, bins, percentiles and corresponding probability densities.
        Arguments:
            - sample: 1D numpy array of values
        Computes:
            - ranks, percentiles, bins and probability densities"""

        if not self.overwrite:
            pass

        # Compute ranks, bins and percentiles
        self.ranks_percentiles_and_bins(sample,vmin,vmax,minmode)
        # Compute probability density
        density, _ = np.histogram(sample,bins=self.bins,density=True)
        self.density = density
        # Number fraction of points below chosen vmin
        self.frac_below_vmin = np.sum(sample < self.vmin)/np.size(sample)
        # Number fraction of points above chosen vmax
        self.frac_above_vmax = np.sum(sample > self.vmax)/np.size(sample)

    def index_of_rank(self,rank):
    
        """Returns the index of the closest rank in numpy.array ranks"""

        dist_to_rank = np.absolute(np.subtract(self.ranks,rank*np.ones(self.ranks.shape)))
        mindist = dist_to_rank.min()
        return np.argmax(dist_to_rank == mindist)

    def rank_id(self,rank):

        """Convert rank (float) to rank id (string)
        """

        return "%2.4f"%rank

    def bin_index(self,percentile=None,rank=None):

        """Returns the index of bin corresponding to percentile or rank 
        of interest
        """

        if percentile is not None:
            # Find first bin edge to be above the percentile of interest
            i_perc = np.argmax(self.bins > percentile)
            if i_perc == 0: # Then percentile is outside the range of stored bins
                return None
            return i_perc-1 # Offset by 1

        if rank is not None:
            return self.index_of_rank(rank)
        # raise WrongArgument("no percentile or rank is provided in bin_index")
        return None

    def format_dimensions(self,sample):
        """Reshape the input data, test the validity and returns the dimensions
        and formatted data.

        Arguments:
        - sample: here we assume data is horizontal, formats it in shape (Ncolumns,)
        Controls if it matches the data used for the control distribution. If
        not, aborts.
        """

        # Get shape
        sshape = sample.shape
        # Initialize default output
        sample_out = sample
        # Get dimensions and adjust output shape
        if len(sshape) > 1: # reshape
            sample_out = np.reshape(sample,np.prod(sshape))
        Npoints, = sample_out.shape
        
        # Test if sample size is correct to access sample points
        if Npoints != self.size:
            raise Exception("Error: used different sample size")

        return sample_out

    def compute_fraction(self,mask):
        """BF addition July 2023. Computes number fraction of True in mask, used for subsampling the data before calculation of the distribution"""
        
        if np.any(mask == slice(None)):
            
            self.frac = 1.

        else:
            
            self.frac = np.sum(np.array(mask,dtype=int))/mask.size
            
    def get_bin_locations(self,sample,bins,sizemax=50,verbose=False,method='shuffle_mask'):
        """Find indices of bins in the sample data, to get a mapping of extremes 
        and fetch locations later
        """
        
        sample = self.format_dimensions(sample)
        
        # Initalize and find bin locations
        nbins = len(bins)
        bin_locations = [[] for _ in range(nbins)]
        bin_sample_size = [0 for _ in range(nbins)]
            
        if method == 'shuffle_mask':

            if verbose: print('bin #: ',end='')
            # compute mask for each bin, randomize and store 'sizemax' first indices
            for i_bin in range(nbins-1):

                if verbose: print('%d..'%i_bin,end='')

                # compute mask
                mask = np.logical_and(sample.flatten() >= bins[i_bin],
                            sample.flatten() < bins[i_bin+1])
                # get all indices
                ind_mask = np.where(mask)[0]
                # shuffle
                np.random.seed(int(round(time.time() * 1000)) % 1000)
                np.random.shuffle(ind_mask)
                
                # save sample size in bin
                bin_sample_size[i_bin] = ind_mask.size # count all points there
                # select 'sizemax' first elements
                bin_locations[i_bin] = ind_mask[:sizemax]

#         elif method == 'random':

#             # Here, look at all points, in random order
            
#             indices = list(range(self.size))
#             np.random.shuffle(indices)

#             bins_full = []
#             for i_ind in range(len(indices)):

#                 i = indices[i_ind]

#                 # Find corresponding bin
#                 i_bin = self.bin_index(percentile=sample[i])

#                 # Store only if bin was found
#                 if i_bin is not None:

#                     # Keep count
#                     self.bin_sample_size[i_bin] += 1
#                     # Store only if there is still room in stored locations list
#                     if len(self.bin_locations[i_bin]) < sizemax:
#                         self.bin_locations[i_bin].append(i)
#                     elif i_bin not in bins_full:
#                         bins_full.append(i_bin)
#                         bins_full.sort()
#                         if verbose:
#                             print("%d bins are full (%d iterations)"%(len(bins_full),i_ind))
                            
        else:
            
            raise ValueError('option "%s" is not implemented for methodget_bin_locations'%method)

            if verbose: print()

        if verbose:
            print()
            
        return bins, bin_locations, bin_sample_size
            
    def store_bin_locations(self,sample,sizemax=50,verbose=False,method='shuffle_mask'):
        """Find indices of bins in the sample data, to get a mapping or extremes 
        and fetch locations later
        """

        if self.bin_locations_stored and not self.overwrite:
            pass

        if verbose:
            print("Finding bin locations...")

        # compute
        bins, bin_locations, bin_sample_size = self.get_bin_locations(sample,self.bins,sizemax=sizemax,verbose=verbose,method=method)
        
        # store
        self.bin_locations = bin_locations
        self.bin_sample_size = bin_sample_size
        
        # mark stored
        self.bin_locations_stored = True

    def store_bin_locations_global(self,sample,global_bins,sizemax=50,verbose=False,method='shuffle_mask'):
        """Find indices of bins from a global distribution from which 'sample' is just a subset"""
        
        if self.global_bin_locations_stored and not self.overwrite:
            pass
        
        if verbose:
            print("Finding bin locations...")
            
        # compute
        global_bins, global_bin_locations, global_bin_sample_size = self.get_bin_locations(sample,global_bins,sizemax=sizemax,verbose=verbose,method=method)
        
        # store
        self.global_bins = global_bins
        self.global_bin_locations = global_bin_locations
        self.global_bin_sample_size = global_bin_sample_size
        
        # mark stored
        self.global_bin_locations_stored = True

    def compute_mean(self,sample,out=False):
        """Compute mean of input sample"""
        
        result = np.mean(sample)
        setattr(self,"mean",result)
        
        if out:
            return result
        
    def compute_individual_percentiles(self,sample,ranks,out=False):
        """Computes percentiles of input sample and store in object attribute"""

        if isinstance(ranks,float) or isinstance(ranks,int):
            ranks = [ranks]
        
        result = []

        for r in ranks:
            # calculate percentile
            p = np.percentile(sample,r)
            result.append(p)
            # save
            setattr(self,"perc%2.0f"%r,p)

        if out:
            return result

    def compute_inv_cdf(self,sample,out=False):
        """Calculate 1-CDF on inverse-logarithmic ranks: fraction of rain mass falling 
        above each percentile"""
        
        self.invCDF = np.ones(self.nbins)*np.nan
        sample_sum = np.nansum(sample)
        for iQ in range(self.nbins):
            rank = self.ranks[iQ]
            perc = self.percentiles[iQ]
            if not np.isnan(perc):
                self.invCDF[iQ] = np.nansum(sample[sample>perc])/sample_sum

        if out:
            return self.invCDF

    def bootstrap_percentiles(self,sample,nd_resample=10,n_bootstrap=50):
        """Perform bootstrapping to evaluate the interquartile range around each
        percentile, for the ranks stored.

        Arguments:
        - sample: np array in Nt,Ny,Nx format
        - nd_resample: number of time indices to randomly select for resampling
        - n_boostrap: number of times to calculate the distribution
        """

        sshape = sample.shape
        d_time = 0

        # calculate and store distribution n_bootstrap times
        perc_list = []
        for i_b in range(n_bootstrap):

            # select days randomly
            indices = list(range(sshape[d_time]))
            np.random.shuffle(indices)
            ind_times = indices[:nd_resample]
            resample = np.take(sample,ind_times,axis=d_time)

            # calculate percentiles on resample
            perc, bins = self.compute_percentiles_and_bins_from_ranks(resample,
                                            store=False,output=True)

            perc_list.append(perc)

        # combine distributions into statistics and save
        perc_array = np.vstack(perc_list)
        self.percentiles_sigma = np.std(perc_array,axis=0)
        self.percentiles_P5 = np.percentile(perc_array,5,axis=0)
        self.percentiles_Q1 = np.percentile(perc_array,25,axis=0)
        self.percentiles_Q2 = np.percentile(perc_array,50,axis=0)
        self.percentiles_Q3 = np.percentile(perc_array,75,axis=0)
        self.percentiles_P95 = np.percentile(perc_array,95,axis=0)
        
    def get_cdf(self):
        """Compute the cumulative density function from the probability density,
        as: fraction of points below vmin + cumulative sum of density*bin_width
        Output is the probability of x < x(bin i), same size as bins (bin edges)"""
        
        # array of bin widths
        bin_width = np.diff(self.bins)
        # CDF from density and bin width
        cdf_base = np.cumsum(bin_width*self.density)
        # readjust  to account for the fraction of points outside the range [vmin,vmax]
        fmin = self.frac_below_vmin
        fmax = self.frac_above_vmax
        cdf = fmin + np.append(0,cdf_base*(1-fmin-fmax))
        
        return cdf
