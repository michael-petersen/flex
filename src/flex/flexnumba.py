import numpy as np

import numpy as np
from numba import njit

@njit
def laguerre_eval(n, alpha, x_vals):
    result = np.empty(len(x_vals))
    for i in range(len(x_vals)):
        x = x_vals[i]
        if n == 0:
            result[i] = 1.0
        elif n == 1:
            result[i] = 1.0 + alpha - x
        else:
            Lnm2 = 1.0
            Lnm1 = 1.0 + alpha - x
            for k in range(2, n + 1):
                L = ((2 * k - 1 + alpha - x) * Lnm1 - (k - 1 + alpha) * Lnm2) / k
                Lnm2 = Lnm1
                Lnm1 = L
            result[i] = Lnm1
    return result


@njit
def laguerre_eval_one(n, x_vals):
    """
    Compute the Laguerre polynomial L_n^1(x) for a single n and multiple
    x values.
    
    Parameters:
        n (int): Order of the Laguerre polynomial.
        x_vals (1D array): Input x values.
    Returns:
        1D array of Laguerre polynomial values for each x in x_vals.
     """
    result = np.empty(len(x_vals))
    for i in range(len(x_vals)):
        x = x_vals[i]
        if n == 0:
            result[i] = 1.0
        elif n == 1:
            result[i] = 2.0 - x
        else:
            Lnm2 = 1.0
            Lnm1 = 2.0 - x
            for k in range(2, n + 1):
                L = ((2 * k - x) * Lnm1 - k * Lnm2) / k
                Lnm2 = Lnm1
                Lnm1 = L
            result[i] = Lnm1
    return result

@njit
def laguerre_all_orders_one(nmax, x_vals):
    """
    Compute L_n^1(x) for all n = 0..nmax and all x in x_vals.

    Parameters:
        nmax (int): Maximum order n.
        x_vals (1D array): Input x values.

    Returns:
        2D array of shape (nmax+1, len(x_vals)) with L_n^1(x).
    """
    nx = len(x_vals)
    result = np.empty((nmax + 1, nx))

    for i in range(nx):
        x = x_vals[i]
        result[0, i] = 1.0
        if nmax >= 1:
            result[1, i] = 2.0 - x
        for n in range(2, nmax + 1):
            result[n, i] = (2 * n - x) * result[n - 1, i]/ n - result[n - 2, i] 

    return result

class FLEXY:
    """
    FLEX class for calculating Laguerre basis amplitudes.

    This class provides methods for calculating Laguerre basis amplitudes based on Weinberg & Petersen (2021).

    Parameters:
        rscl (float): Scale parameter for the Laguerre basis.
        mass (array-like): Mass values for particles.
        phi (array-like): Angular phi values.
        velocity (array-like): Velocity values.
        R (array-like): Radial values.
        mmax (int): Maximum order parameter for m.
        nmax (int): Maximum order parameter for n.

    Methods:
        gamma_n(nrange, rscl): Calculate the Laguerre alpha=1 normalisation.
        G_n(R, nrange, rscl): Calculate the Laguerre basis.
        n_m(): Calculate the angular normalisation.
        laguerre_amplitudes(): Calculate Laguerre amplitudes for the given parameters.
        laguerre_reconstruction(rr, pp): Calculate Laguerre reconstruction.

    Attributes:
        rscl (float): Scale parameter for the Laguerre basis.
        mass (array-like): Mass values for particles.
        phi (array-like): Angular phi values.
        velocity (array-like): Velocity values.
        R (array-like): Radial values.
        mmax (int): Maximum order parameter for m.
        nmax (int): Maximum order parameter for n.
        coscoefs (array-like): Cosine coefficients.
        sincoefs (array-like): Sine coefficients.
        reconstruction (array-like): Laguerre reconstruction result.
    """

    def __init__(self, rscl, mmax, nmax, R, phi, mass=1., velocity=1.):
        """
        Initialize the LaguerreAmplitudes instance with parameters.

        Args:
            rscl (float): Scale parameter for the Laguerre basis.
            mmax (int): Maximum Fourier harmonic order.
            nmax (int): Maximum Laguerre order.
            R (array-like): Radial values.
            velocity (array-like): Velocity values.
            mass (integer or array-like): Mass values for particles.
            phi (integer or array-like): Angular phi values.

        """
        self.rscl = rscl
        self.mmax = mmax
        self.nmax = nmax
        self.R = R
        self.phi = phi
        self.mass = mass
        self.velocity = velocity


        # run the amplitude calculation
        self.laguerre_amplitudes()

    def _gamma_n(self,nrange, rscl):
        """
        Calculate the Laguerre alpha=1 normalisation.

        Args:
            nrange (array-like): Range of order parameters.
            rscl (float): Scale parameter for the Laguerre basis.

        Returns:
            array-like: Laguerre alpha=1 normalisation values.
        """
        return (rscl / 2.) * np.sqrt(nrange + 1.)

    def _G_n(self, R, nrange, rscl):
        """
        Calculate the Laguerre basis.

        Args:
            R (array-like): Radial values.
            nrange (array-like): Range of order parameters.
            rscl (float): Scale parameter for the Laguerre basis.

        Returns:
            array-like: Laguerre basis values.
        """
        R = np.asarray(R)
        x = 2 * R / rscl
        gamma_values = np.array([self._gamma_n(n, rscl) for n in nrange])

        # Preallocate output array for speed
        laguerrevalues = np.empty((len(nrange), len(R)))

        for i, n in enumerate(nrange):
            laguerrevalues[i] = laguerre_eval_one(n, x) / gamma_values[i]

        return np.exp(-R / rscl) * laguerrevalues

    def _n_m(self):
        """
        Calculate the angular normalisation.

        Returns:
            array-like: Angular normalisation values.
        """
        deltam0 = np.zeros(self.mmax+1)
        deltam0[0] = 1.0
        #return np.power((deltam0 + 1) * np.pi / 2., -0.5)
        return np.power((deltam0 + 1) * np.pi / 2.,-1.)
        #return np.power(deltam0+1,-1.0)#np.power((deltam0 + 1), -0.5)

    def laguerre_amplitudes_newaxis(self):
        """
        Calculate Laguerre amplitudes for the given parameters.

        Returns:
            tuple: Tuple containing the cosine and sine amplitudes.
        """

        G_j = self._G_n(self.R, np.arange(0,self.nmax,1), self.rscl)

        nmvals = self._n_m()
        cosm = np.array([nmvals[m]*np.cos(m*self.phi) for m in np.arange(0,self.mmax+1,1)])
        sinm = np.array([nmvals[m]*np.sin(m*self.phi) for m in np.arange(0,self.mmax+1,1)])

        # broadcast to sum values
        self.coscoefs = np.nansum(cosm[:, np.newaxis, :] * G_j[np.newaxis, :, :] * self.mass * self.velocity,axis=2)
        self.sincoefs = np.nansum(sinm[:, np.newaxis, :] * G_j[np.newaxis, :, :] * self.mass * self.velocity,axis=2)


    def laguerre_amplitudes(self):
        """
        Calculate Laguerre amplitudes for the given parameters.

        Returns:
            none. Attributes are added containing the cosine and sine amplitudes.
        """

        G_j = self._G_n(self.R, np.arange(0,self.nmax,1), self.rscl)

        nmvals = self._n_m()
        cosm = np.array([nmvals[m]*np.cos(m*self.phi) for m in np.arange(0,self.mmax+1,1)])
        sinm = np.array([nmvals[m]*np.sin(m*self.phi) for m in np.arange(0,self.mmax+1,1)])

        #if scalar:
        if np.isscalar(self.mass) and np.isscalar(self.velocity):
            scale = self.mass * self.velocity  # scalar
            self.coscoefs = scale * np.einsum('mn,jn->mj', cosm, G_j)
            self.sincoefs = scale * np.einsum('mn,jn->mj', sinm, G_j)   
        else:
            # vector case
            self.coscoefs = np.einsum('mn,jn,n->mj', cosm, G_j, self.mass * self.velocity)
            self.sincoefs = np.einsum('mn,jn,n->mj', sinm, G_j, self.mass * self.velocity)

    def laguerre_reconstruction(self, rr, pp):
        """
        Reconstruct a function using Laguerre amplitudes.

        Args:
            rr (array-like): Radial values.
            pp (array-like): Angular phi values.

        This method reconstructs a function using the Laguerre amplitudes calculated with the `laguerre_amplitudes` method.

        Returns:
            array-like: The reconstructed function values.
        """
        nmvals = self._n_m()
        G_j = self._G_n(rr, np.arange(0, self.nmax, 1), self.rscl)

        fftotal = 0.
        for m in range(0, self.mmax+1):
            for n in range(0, self.nmax):
                fftotal += self.coscoefs[m, n] * np.cos(m * pp) * G_j[n]
                fftotal += self.sincoefs[m, n] * np.sin(m * pp) * G_j[n]

        self.reconstruction = fftotal * 0.5 #/ np.pi
