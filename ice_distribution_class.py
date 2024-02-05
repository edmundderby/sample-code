import numpy as np
from scipy import integrate
from scipy import sparse
from scipy.interpolate import splrep, splev, interp1d
from matplotlib import pyplot as plt
from IPython import display
import pylab as pl
import time

class IceDistModel:
    def __init__(
        self,
        thickness_minimum=1e-100,
        thickness_maximum=10,
        thickness_interval=1e-2,
        H_c=0.1,
        timestep=20*60,
        duration_of_year=365*24*3600,
        delta_F_0=0,
        F_B=0,
        initial_distribution=None,
        T_ml=0,
        flux_data='mu71',
        ridging_rate=1e-8,
        rifting_rate=1e-8,
        kernel='linear',
        kernel_decay_rate=1,
        instantly_freeze_open_water=True,
        use_sk_k1_k2_values=False,
                 ):
        
        #this creates everything needed to start the model off
        #sets all constants that are called in the model
        #defines the timestep and sets how long the annual cycle of the model is (by default 1 year)
        #defines the thickness coordinates
        #initialises the thickness distribution, either as the steady state solution to the Stefan growth problem or as the open water distribution
        #initialises other relevant quantities like temperature and growth rate functions
        #sets k_1 and k_2, by default as the values in Toppaladoddi et al. 2015
        #calculates the forcing splines for EW09 data
        
        #set constants for use in ice model
        #thermodynamic constants
        self._k_i = 2.2 #[W m^-1 K^-1] - thermal conductivity of ice
        self._rho_i = 917 #[kg m^-3] - density of ice
        self._L_i = 333.4e3 #[J kg^-1] - latent heat of fusion of ice
        self._c_i = 3965 #[J kg^-1 K^-1] - heat capacity of ice
        self._K_i = self._k_i/(self._rho_i*self._c_i) #[m^2 s^-1] - thermal diffusivity of ice
        #properties of ocean mixed layer
        self._c_ml = 4e6 #[J m^-3 K^-1] - heat capacity of mixed layer
        self._H_ml = 50 #[m] - thickness of mixed layer
        #albedo properties
        self._h_a = 0.67 #[m] - the transition thickness scale of the albedo
        self._a_ml = 0.2 #[unitless] - albedo of mixed layer
        self._a_i = 0.68 #[unitless] - albedo of thick sea ice
        
        #initialise model time
        self._time = 0 #[s] - time elapsed in model
        self._time_in_year = 0
        self._delta_t = timestep
        self._duration_of_year = duration_of_year
        self._timesteps_in_year = int(duration_of_year/timestep) #number of timesteps in 1 year
        
        #create thickness coordinates
        self._delta_h = thickness_interval
        self._thickness_coordinates = np.arange(thickness_minimum,thickness_maximum,self._delta_h)
        #set cut off thickness for definition of open water boundary condition
        self._H_c = H_c
        
        self._instantly_freeze_open_water = instantly_freeze_open_water

        #define q_ss and H_ss for the steady state Stefan growth solution
        q_ss = 1.838
        H_ss = 0.777
        
        #check if an initial thickness distribution has been specified
        if initial_distribution is not None and initial_distribution.size!=self._thickness_coordinates.size:
            print(f'The specified initial thickness distribution doesn\'t have the right coordinates.')
        if initial_distribution is not None and initial_distribution.size==self._thickness_coordinates.size:
            initial_thickness_distribution = initial_distribution
            self._thickness_distribution = initial_thickness_distribution
            self._open_water_fraction = self.calculate_open_water_fraction()

            self._T_ml = 0
        else:
            #otherwise initialise the thickness distribution in this steady state form
            initial_thickness_distribution = self._thickness_coordinates**q_ss * np.exp(-self._thickness_coordinates/H_ss)
            self._thickness_distribution = initial_thickness_distribution/(integrate.trapz(initial_thickness_distribution,self._thickness_coordinates))
            #initialise open water fraction
            self._open_water_fraction = 0 #this should be 0, but set as a small number to avoid singularity errors
            #initialise mixed layer temperature
            self._T_ml = 0
            if T_ml>0:
                initial_thickness_distribution[:] = 0
                self._thickness_distribution[:] = 0
                self._T_ml = T_ml
                self._open_water_fraction = 1
        
        #initialise temperature field - by default set to zero for all thicknesses
        self._temperature = np.zeros(self._thickness_coordinates.size)
        #initialise growth rate - by default set to zero for all thicknesses
        self._growth_rate = np.zeros(self._thickness_coordinates.size)
        self._growth_rate_old = np.zeros(self._thickness_coordinates.size)
        
        #set Stefan number of ice
        S=334.3e3/(3965*18.44)
        #set mechanical constants
        if use_sk_k1_k2_values:
            self._k_1 = 0.048 * 1.5*0.1/1e5
            self._k_2 = 0.025 * 1.5**2 *0.1/1e5
        else:
            self._k_1 = 6.05e-7/(q_ss*S)
            self._k_2 = 6.05e-7/(q_ss*S*H_ss)
        
        #calculate initial phi
        self._phi = self.calculate_phi()
        
        #calculate the albedo for each thickness coordinate
        self._albedo = (self._a_ml+self._a_i)/2 + (self._a_ml-self._a_i)/2 * np.tanh(-self._thickness_coordinates/(self._h_a))
        
        #set constant export rate for sea ice
        self._export_rate = 0.1/(365*24*3600)
        
        #generate splines for EW09 data to calculate forcing values
        self._flux_data = flux_data
        self.calculate_forcing_splines()
        self._sigma_0 = 316 #[W m^-2]
        self._sigma_T = 3.9
        #set forcing perturbation value
        self._delta_F_0 = delta_F_0
        self._F_B = F_B
        
        #set mechanical rate for convolution mechanics
        self._ridging_rate = ridging_rate
        self._rifting_rate = rifting_rate
        
        #set coagulation/fragmentation mechanics parameters
        self._coag_frag_kernel = kernel
        self._coag_frag_kernel_decay_rate = kernel_decay_rate
        
        #creaet matrix for coagulation kernel
        x_grid, y_grid = np.meshgrid(self._thickness_coordinates,self._thickness_coordinates)
        if self._coag_frag_kernel=='linear':
            def K(x,y):
                return np.ones_like(x)
        elif self._coag_frag_kernel=='additive':
            def K(x,y):
                return x+y
        elif self._coag_frag_kernel=='multiplicative':
            def K(x,y):
                return x*y
        elif self._coag_frag_kernel=='arithmetic_mean_exp_decay':
            def K(x,y):
                return np.exp(-self._coag_frag_kernel_decay_rate*(x+y)/2)
        elif self._coag_frag_kernel=='geometric_mean_exp_decay':
            def K(x,y):
                return np.exp(-self._coag_frag_kernel_decay_rate*np.sqrt(x*y))
        elif self._coag_frag_kernel=='harmonic_mean_exp_decay':
            def K(x,y):
                return np.exp(-self._coag_frag_kernel_decay_rate*x*y/(x+y))
        self._K_matrix = self._ridging_rate*K(x_grid,y_grid)
    
    def calculate_forcing_splines(self):
        if self._flux_data=='mu71':
            #this calculates the forcing splines for the EW09 data
            #set forcing parameters at the beginning of each month according to supplementary material
            #the first value is repeated as the last value to ensure smooth periodicity
            F_S_mu71 = 15.9 * np.array([0, 0, 1.9, 9.9, 17.7, 19.2, 13.6, 9.0, 3.7, 0.4, 0, 0, 0]) #[W m^-2]
            F_L_mu71 = 15.9 * np.array([10.4, 10.3, 10.3, 11.6, 15.1, 18.0, 19.1, 18.7, 16.5, 13.9, 11.2, 10.9, 10.4]) #[W m^-2]
            F_SH_mu71 = 15.9 * np.array([1.18, 0.76, 0.72, 0.29, -0.45, -0.39, -0.30, -0.40, -0.17, 0.10, 0.56, 0.79, 1.18]) #[W m^-2]
            F_LH_mu71 = 15.9 * np.array([0, -0.02, -0.03, -0.09, -0.46, -0.70, -0.64, -0.66, -0.39, -0.19, -0.01, -0.01, 0]) #[W m^-2]
            #interpolate forcing parameters for every model timestep
            duration_of_month = self._duration_of_year/12
            times_of_mu71_flux_data = np.arange(duration_of_month/2, self._duration_of_year+3*duration_of_month/2, duration_of_month)
            times_of_model_steps_in_year = np.linspace(0,self._duration_of_year,self._timesteps_in_year)
            self._F_S_spline = splrep(times_of_mu71_flux_data, F_S_mu71, k=3, per=1)
            self._F_L_spline = splrep(times_of_mu71_flux_data, F_L_mu71, k=3, per=1)
            self._F_SH_spline = splrep(times_of_mu71_flux_data, F_SH_mu71, k=3, per=1)
            self._F_LH_spline = splrep(times_of_mu71_flux_data, F_LH_mu71, k=3, per=1)
        elif self._flux_data=='ew09':
            #this calculates the forcing splines for the EW09 data
            #set forcing parameters at the beginning of each month according to supplementary material
            #the first value is repeated as the last value to ensure smooth periodicity
            F_S_ew09 = np.array([0, 0, 30, 160, 280, 310, 220, 140, 59, 6.4, 0, 0, 0]) #[W m^-2]
            F_0_ew09 = np.array([120, 120, 130, 94, 64, 61, 57, 54, 56, 64, 82, 110, 120]) #[W m^-2]
            F_T_ew09 = np.array([3.1, 3.2, 3.3, 2.9, 2.6, 2.6, 2.6, 2.5, 2.5, 2.6, 2.7, 3.1, 3.1]) #[W m^-2]
            #interpolate forcing parameters for every model timestep
            duration_of_month = self._duration_of_year/12
            times_of_ew09_flux_data = np.arange(duration_of_month/2, self._duration_of_year+3*duration_of_month/2, duration_of_month)
            times_of_model_steps_in_year = np.linspace(0,self._duration_of_year,self._timesteps_in_year)
            self._F_S_spline = splrep(times_of_ew09_flux_data, F_S_ew09, k=3, per=1)
            self._F_0_spline = splrep(times_of_ew09_flux_data, F_0_ew09, k=3, per=1)
            self._F_T_spline = splrep(times_of_ew09_flux_data, F_T_ew09, k=3, per=1)
    
    def interpolate_forcing(self):
        #this calculates the forcing at the model's current time, given the forcing splines
        
        #calculate time in year
        time_in_year = self._time%self._duration_of_year
        if self._flux_data=='mu71':
            F_S = splev(time_in_year,self._F_S_spline)
            F_L = splev(time_in_year,self._F_L_spline)
            F_SH = splev(time_in_year,self._F_SH_spline)
            F_LH = splev(time_in_year,self._F_LH_spline)
            self._F_S = F_S
            self._F_L = F_L
            self._F_SH = F_SH
            self._F_LH = F_LH
            return F_S,F_L,F_SH,F_LH
        elif self._flux_data=='ew09':
            F_S = splev(time_in_year,self._F_S_spline)
            F_0 = splev(time_in_year,self._F_0_spline)
            F_T = splev(time_in_year,self._F_T_spline)
            self._F_S = F_S
            self._F_0 = F_0
            self._F_T = F_T
            return F_S,F_0,F_T
    
    def calculate_phi(self):
        #this calculates phi for the model's timestep for use in the numerical method, see Toppaladoddi et al. 2015
        
        phi = self._k_1 - self._growth_rate
        self._phi = phi
        return phi
    
    def calculate_open_water_fraction(self):
        #this calculates the open water fraction of the system
        
        open_water_fraction = 1-integrate.trapz(self._thickness_distribution,self._thickness_coordinates)
        if open_water_fraction>1:
            open_water_fraction=1
        
        self._open_water_fraction = open_water_fraction
        return open_water_fraction
    
    def calculate_mean_thickness(self):
        #this calculates the mean thickness across the entire grid box and across the ice covered area
        #overall_mean_thickness - mean thickness across entire grid box (i.e. open water areas count for thickness of 0)
        #ice_mean_thickness - mean thickness only across ice-covered areas, this will therefore always be larger
        
        overall_mean_thickness = integrate.trapz(self._thickness_distribution*self._thickness_coordinates,self._thickness_coordinates) #mean thickness over the entire model domain
        ice_mean_thickness = integrate.trapz(self._thickness_distribution*self._thickness_coordinates,self._thickness_coordinates)/integrate.trapz(self._thickness_distribution,self._thickness_coordinates) #mean thickness within the ice covered region
        
        self._overall_mean_thickness = overall_mean_thickness
        self._ice_mean_thickness = ice_mean_thickness
        
        return overall_mean_thickness, ice_mean_thickness
    
    def calculate_mean_albedo(self):
        #this calculates the mean albedo of the current model state
        
        #This is a combination of the mean albedo of the thickness distribution and the albedo of the open water
        mean_albedo = self._open_water_fraction*self._albedo[0] + integrate.trapz(self._thickness_distribution*self._albedo,self._thickness_coordinates)
        return mean_albedo
    
    def calculate_temperature(self):
        #this calculates surface temperature for each ice thickness using the method outlined in EW09
        
        #Assumes forcings described in the form used in EW09.
        if self._flux_data=='mu71':
            ramp_expression = ((1-self._albedo)*self._F_S-self._sigma_0+self._F_L+self._F_LH+self._F_SH+self._delta_F_0)/(-self._k_i/self._thickness_coordinates - self._sigma_T)
        elif self._flux_data=='ew09':
            ramp_expression = ((1-self._albedo)*self._F_S-self._F_0+self._delta_F_0)/(-self._k_i/self._thickness_coordinates - self._F_T)
        temperature = np.zeros_like(self._thickness_coordinates)
        temperature[(ramp_expression>0)] = -ramp_expression[(ramp_expression>0)]
        temperature[(ramp_expression<=0)] = 0
        
        self._temperature = temperature
        
        return temperature
        
    def calculate_growth_rate(self):
        #this calculates the thermodynamic growth rate for each thickness in a method analogous to EW09
        #heat flux into open water adds an additional contribution to the basal heat flux
        #when ice is not present the mixed layer is heated
        
        #when ice is present 100% of heat input to open water contributes to an additional heat flux at ice bottom
        #when ice is not present 100% of heat input goes into changing temperature of open water
        #on the onset of freezing the open water flux is set to zero

        #calculate total energy flux into the ice pack (valid assuming a linear temperature profile)
        if self._flux_data=='mu71':
            energy_flux = (1-self._albedo)*self._F_S - self._sigma_0 + self._F_L + self._F_LH + self._F_SH - self._sigma_T*self._temperature + self._delta_F_0 + self._F_B
        elif self._flux_data=='ew09':
            energy_flux = (1-self._albedo)*self._F_S - self._F_0 + self._delta_F_0 - self._F_T*self._temperature + self._F_B
        #calculate thermodynamic growth rate
        growth_rate = -1/(self._rho_i*self._L_i)*(energy_flux)
        
        #calculate energy flux into open water
        open_water_albedo = (self._a_ml+self._a_i)/2 + (self._a_ml-self._a_i)/2 * np.tanh(self._c_ml*self._H_ml*self._T_ml/(self._rho_i*self._L_i*self._h_a))
        
        if self._flux_data=='mu71':
            open_water_flux = (1-open_water_albedo)*self._F_S - self._sigma_0 + self._F_L + self._F_LH + self._F_SH - self._sigma_T*self._T_ml + self._delta_F_0
        elif self._flux_data=='ew09':
            open_water_flux = (1-open_water_albedo)*self._F_S - self._F_0 + self._delta_F_0 - self._F_T*self._T_ml
        if self._open_water_fraction>=1:
            mixed_layer_flux = open_water_flux*self._open_water_fraction
        else:
            if open_water_flux>0:
                bottom_flux = open_water_flux*self._open_water_fraction/(1-self._open_water_fraction)
                growth_rate -= 1/(self._rho_i*self._L_i)*bottom_flux
                #no energy is left to heat the mixed layer
                mixed_layer_flux = 0
            else:
                mixed_layer_flux = open_water_flux*self._open_water_fraction
        
        self._growth_rate = growth_rate
        self._mixed_layer_flux = mixed_layer_flux
        
        return growth_rate, mixed_layer_flux
    
    def export_ice(self):
        #this models the export of ice out of the Arctic according to an export rate (specified when the model is called)
        #ice export is modeled as being equally likely for ice of all thicknesses - therefore if over a timestep ice concentration is
        #reduced by 10% - this is modeled simply by scaling the distribution by 0.9
        
        if self._T_ml<=0:
            self._thickness_distribution = self._thickness_distribution*(1-self._export_rate*self._delta_t)
            self.calculate_open_water_fraction()
        #calculate new open water_fraction, mean albedo and mean thickness
        self.calculate_open_water_fraction()
        self.calculate_mean_thickness()
        self.calculate_mean_albedo()
    
    def advance_distribution(self):
        
        self._k1_k2_error = (self._k_2*self._thickness_distribution[0]-self._k_1*integrate.trapz(self._thickness_distribution,self._thickness_coordinates))*self._delta_t
        
        self._mean_thickness_from_PDE = 0
        
        if self._T_ml<=0 and (
            integrate.trapz((self._growth_rate-self._k_1)*self._thickness_distribution,self._thickness_coordinates)-self._k_2*self._thickness_distribution[0]
        )*self._delta_t+integrate.trapz(self._thickness_coordinates*self._thickness_distribution,self._thickness_coordinates)>0:
            #implicit scheme for advancing the ice thickness distribution PDE
            self._mean_thickness_before_PDE = integrate.trapz(self._thickness_distribution*self._thickness_coordinates,self._thickness_coordinates)
            #the boundary condition at h=0+ conserves the area under the curve with the flux of probability out of the system
            #calculate phi
            self.calculate_phi()
            #define c_1 and c_2
            c_1 = self._delta_t/(4*self._delta_h)
            c_2 = self._k_2*self._delta_t/(2*(self._delta_h)**2)
            domain_size = self._thickness_coordinates.size
            #solves the Fokker-Planck equation for  sea ice thickness using a matrix method
            A_diagonals = [
                self._phi[:-1]/2 - self._k_2/self._delta_h, 
                np.ones(domain_size)*(2*self._k_2/self._delta_h+self._delta_h/self._delta_t),
                -self._phi[1:]/2 - self._k_2/self._delta_h
            ]
            B_diagonals = [
                np.zeros(domain_size), 
                np.ones(domain_size)*self._delta_h/self._delta_t,
                np.zeros(domain_size)
            ]
            if self._instantly_freeze_open_water:
                if self._T_ml<=0 and self._mixed_layer_flux<0:
                    A_diagonals[1][0] = 1
                    A_diagonals[2][0] = 0
                    B_diagonals[1][0] = 0
                    B_diagonals[2][0] = 0
                else:
                    A_diagonals[1][0] = 1 - self._delta_t/self._H_c *((self._k_1-self._growth_rate[0])/2-self._k_2/self._delta_h)
                    A_diagonals[2][0] = -self._delta_t/self._H_c * ((self._k_1-self._growth_rate[1])/2+self._k_2/self._delta_h)
                    B_diagonals[1][0] = 1
                    B_diagonals[2][0] = 0
            else:
                A_diagonals[1][0] = 1 - self._delta_t/self._H_c *((self._k_1-self._growth_rate[0])/2-self._k_2/self._delta_h)
                A_diagonals[2][0] = -self._delta_t/self._H_c * ((self._k_1-self._growth_rate[1])/2+self._k_2/self._delta_h)
            A_diagonals[1][-1] = 1
            A_diagonals[0][-1] = 0
            A = sparse.diags(A_diagonals,[-1,0,1],(domain_size,domain_size),format="csr")
            B_diagonals[1][-1] = 0
            B_diagonals[0][-1] = 0
            B = sparse.diags(B_diagonals,[-1,0,1],(domain_size,domain_size))

            #solve the matrix equation for new g
            rhs = B*self._thickness_distribution
            new_thickness_distribution = sparse.linalg.spsolve(A, rhs)
            self._mean_thickness_after_PDE = integrate.trapz(new_thickness_distribution*self._thickness_coordinates,self._thickness_coordinates)
            self._mean_thickness_from_PDE = self._mean_thickness_after_PDE-self._mean_thickness_before_PDE
        else:
            new_thickness_distribution = self._thickness_distribution*0
        
        #change mixed layer temperature if energy is input to mixed layer, or it is above freezing
        self._T_ml += self._mixed_layer_flux*self._delta_t/(self._c_ml*self._H_ml)
        #if T_ml is reduced below 0, form ice uniformly between 0 and H_c sufficiently to increase to freezing point
        self._mean_thickness_from_new_ice = 0
        if self._T_ml<0 and self._open_water_fraction<1:
            
            mean_thickness_before_new_ice = integrate.trapz(new_thickness_distribution*self._thickness_coordinates,self._thickness_coordinates)
            
            #set T_ml to zero and convert energy into thin ice
            A = 1-integrate.trapz(new_thickness_distribution,self._thickness_coordinates)
            h_f = (1/A)*-self._c_ml*self._H_ml*self._T_ml/(self._rho_i*self._L_i)
            sigma = h_f/6 #set sigma to a fraction of mu - in this case to ensure up to 6 standard deviations below are >0
            g_new_ice = A * 1/(sigma*(2*np.pi)**0.5) * np.exp(-0.5*(self._thickness_coordinates-h_f)**2/(sigma**2))
            g_new_ice = A*g_new_ice/integrate.trapz(g_new_ice,self._thickness_coordinates)
            new_thickness_distribution += g_new_ice
            new_thickness_distribution[0] = 0
            new_thickness_distribution = new_thickness_distribution/integrate.trapz(new_thickness_distribution,self._thickness_coordinates)
            self._T_ml = 0
            
            mean_thickness_after_new_ice = integrate.trapz(new_thickness_distribution*self._thickness_coordinates,self._thickness_coordinates)
            
            self._mean_thickness_from_new_ice = mean_thickness_after_new_ice-mean_thickness_before_new_ice
        
        #if mixed layer goes above freezing and system is NOT ice free, uniformly melt ice cover sufficiently to cool mixed layer to freezing
        if self._T_ml>0 and self._open_water_fraction<1:
            #if mixed layer energy is less than energy in ice and the melt will keep open water fraction below 1
            if self._c_ml*self._H_ml*self._T_ml<self._rho_i*self._L_i*self._overall_mean_thickness:
                self._thickness_distribution -= self._thickness_distribution*self._c_ml*self._H_ml*self._T_ml/(self._rho_i*self._L_i*self._overall_mean_thickness)
                self.calculate_open_water_fraction()
                self._thickness_distribution[0] = self._open_water_fraction/self._H_c
                self._T_ml = 0
            else:
                open_water_fraction = 1
                #ice_mean_thickness = 0
                #overall_mean_thickness = 0
                mean_albedo = self._albedo[0]
                
        self._thickness_distribution = new_thickness_distribution
        #calculate new open water_fraction, mean albedo and mean thickness
        self.calculate_open_water_fraction()
        self.calculate_mean_thickness()
        self.calculate_mean_albedo()
                
        self.export_ice()
        
        return new_thickness_distribution
    
    def grow_ice(self):
        if self._T_ml<=0 and integrate.trapz(self._growth_rate*self._thickness_distribution)*self._delta_t+integrate.trapz(self._thickness_coordinates*self._thickness_distribution)>0:# or self._growth_rate[0]>0:
            #implicit scheme for advancing the ice thickness distribution PDE
            #the boundary condition at h=0+ conserves the area under the curve with the flux of probability out of the system
            #calculate phi
            self.calculate_phi()
            #define c_1 and c_2
            c_1 = -(self._growth_rate+np.abs(self._growth_rate))/2*self._delta_t/self._delta_h
            c_2 = -(self._growth_rate-np.abs(self._growth_rate))/2*self._delta_t/self._delta_h
            domain_size = self._thickness_coordinates.size
            
            f_i_plus_half = (self._growth_rate+np.roll(self._growth_rate,-1))/2
            f_i_minus_half = (self._growth_rate+np.roll(self._growth_rate,1))/2
            
            #solves the Fokker-Planck equation for  sea ice thickness using a matrix method
            A_diagonals = [np.zeros(domain_size-1), 
                           np.ones(domain_size),
                           np.zeros(domain_size-1)]
            B_lower_diagonal = np.zeros(domain_size)
            B_main_diagonal = np.zeros(domain_size)
            B_upper_diagonal = np.zeros(domain_size)
            B_lower_diagonal[self._growth_rate<0] = 0
            B_main_diagonal[self._growth_rate<0] = (1+self._delta_t/self._delta_h*self._growth_rate)[self._growth_rate<0]
            B_upper_diagonal[self._growth_rate<0] = (-self._delta_t/self._delta_h*np.roll(self._growth_rate,-1))[self._growth_rate<0]
            B_lower_diagonal[self._growth_rate>0] = (self._delta_t/self._delta_h*np.roll(self._growth_rate,1))[self._growth_rate>0]
            B_main_diagonal[self._growth_rate>0] = (1-self._delta_t/self._delta_h*self._growth_rate)[self._growth_rate>0]
            B_upper_diagonal[self._growth_rate>0] = 0
            B_diagonals = [B_lower_diagonal[1:], #crop first element from lower diagonal as it is not featured in the matrix
                           B_main_diagonal,
                           B_upper_diagonal[:-1]] #crop last element from upper diagonal as it is not featured in the matrix
            
            if self._growth_rate[-1]<0:
                #set far field value of G to 1
                A_diagonals[1][-1] = 1
                A_diagonals[0][-1] = 0
                B_diagonals[1][-1] = 0
                B_diagonals[0][-1] = 0
            if self._growth_rate[0]>0:
                #immediately freeze all open water - this sets g(0)=0
                A_diagonals[1][0] = 1#1+self._delta_t/(2*self._H_c)*self._growth_rate[0]
                A_diagonals[2][0] = 0#self._delta_t/(2*self._H_c)*self._growth_rate[1]
                B_diagonals[1][0] = 0#1
                B_diagonals[2][0] = 0
                
            A = sparse.diags(A_diagonals,[-1,0,1],(domain_size,domain_size),format="csr")
            B = sparse.diags(B_diagonals,[-1,0,1],(domain_size,domain_size))
            
            #solve the matrix equation for new g
            rhs = B*self._thickness_distribution
            new_thickness_distribution = sparse.linalg.spsolve(A, rhs)
            new_thickness_distribution[new_thickness_distribution<0] = 0
        else:
            #new_thickness_distribution = self._thickness_distribution
            #self._thickness_distribution = new_thickness_distribution
            new_thickness_distribution = self._thickness_distribution*0
        
        #change mixed layer temperature if energy is input to mixed layer, or it is above freezing
        self._T_ml += self._mixed_layer_flux*self._delta_t/(self._c_ml*self._H_ml)
        A = 1-integrate.trapz(new_thickness_distribution,self._thickness_coordinates)
        #if T_ml is reduced below 0, form ice uniformly between 0 and H_c sufficiently to increase to freezing point
        self._mean_thickness_from_new_ice = 0
        if self._T_ml<0:
            mean_thickness_before_new_ice = integrate.trapz(new_thickness_distribution*self._thickness_coordinates,self._thickness_coordinates)
            #set T_ml to zero and convert energy into thin ice
            h_f = (1/A)*-self._c_ml*self._H_ml*self._T_ml/(self._rho_i*self._L_i)
            sigma = h_f/6 #set sigma to a fraction of mu - in this case to ensure up to 6 standard deviations below are >0
            g_new_ice = np.zeros_like(self._thickness_coordinates)
            g_new_ice = (A * 1/(sigma*(2*np.pi)**0.5) * np.exp(-0.5*(self._thickness_coordinates-h_f)**2/(sigma**2)))
            #g_new_ice = A * 1/(sigma*(2*np.pi)**0.5) * np.exp(-0.5*(self._thickness_coordinates-h_f)**2/(sigma**2))
            g_new_ice = A*g_new_ice/integrate.trapz(g_new_ice,self._thickness_coordinates)
            new_thickness_distribution += g_new_ice
            #new_thickness_distribution[0] = 0
            new_thickness_distribution = new_thickness_distribution/integrate.trapz(new_thickness_distribution,self._thickness_coordinates)
            
            mean_thickness_after_new_ice = integrate.trapz(new_thickness_distribution*self._thickness_coordinates,self._thickness_coordinates)
            
            self._mean_thickness_from_new_ice = mean_thickness_after_new_ice-mean_thickness_before_new_ice
            
            self._T_ml = 0
        
        #if mixed layer goes above freezing and system is NOT ice free, uniformly melt ice cover sufficiently to cool mixed layer to freezing
        if self._T_ml>0 and A<1:#self._open_water_fraction<1:
            #if mixed layer energy is less than energy in ice and the melt will keep open water fraction below 1
            if self._c_ml*self._H_ml*self._T_ml<self._rho_i*self._L_i*self._overall_mean_thickness:
                self._thickness_distribution -= self._thickness_distribution*self._c_ml*self._H_ml*self._T_ml/(self._rho_i*self._L_i*self._overall_mean_thickness)
                self.calculate_open_water_fraction()
                self._thickness_distribution[0] = self._open_water_fraction/self._H_c
                self._T_ml = 0
            else:
                open_water_fraction = 1
                #ice_mean_thickness = 0
                #overall_mean_thickness = 0
                mean_albedo = self._albedo[0]
                
        self._thickness_distribution = new_thickness_distribution
        #calculate new open water_fraction, mean albedo and mean thickness
        self.calculate_open_water_fraction()
        self.calculate_mean_thickness()
        self.calculate_mean_albedo()
        
        self._growth_rate_old = np.copy(self._growth_rate)
        
        return new_thickness_distribution
    
    def do_coagulation_fragmentation_mechanics(self):
        domain_size = self._thickness_coordinates.size
        xs = self._thickness_coordinates
        Δx = self._delta_h
        g = self._thickness_distribution
        
        if self._open_water_fraction<0.99 and self._T_ml==0:
            start_time = time.time()
            if self._ridging_rate>0:
                coagulation = ([0.5*sum([self._K_matrix[j,i-j]*g[j]*g[i-j] for j in range(i)])*Δx for i in range(np.size(xs))]
                      - g*sum([self._K_matrix[i]*g[i] for i in range(np.size(xs))])*Δx)
            else:
                coagulation = 0
            if False:#self._rifting_rate>0:
                fragmentation = (sum([K_frag(xs,xs[i],g)*
                            [g[i+j] if (i+j)<(np.size(g)-1) else g[-1] for j in range(np.size(xs))]
                                  for i in range(np.size(xs))])*Δx
                      - 0.5*g*[sum([K_frag(xs[i-j],xs[j],g) for j in range(i)])*Δx for i in range(np.size(xs))])
            else:
                fragmentation = 0
            mechanics = coagulation+fragmentation
            self._mechanics_rate = mechanics
            g_new = np.copy(g)+mechanics*self._delta_t
            
            # I artificially suppress any local maximum with a peak of less than 10^(-3.5)
            # this is because these can spuriously be produced by the mechanics scheme at large thicknesses. These can then grow
            # and prevent the system from reaching an ice-free state
        else:
            g_new = g
            self._mechanics_rate = np.zeros_like(self._thickness_coordinates)
        end_time = time.time()
        #print(end_time-start_time)
        self._thickness_distribution = g_new
        
    def do_convolution_mechanics(self):
        domain_size = self._thickness_coordinates.size
        if self._open_water_fraction<1:
            #solve for ice mechanical effects
            ridging_rate = self._ridging_rate*(1-self._open_water_fraction)
            rifting_rate = self._rifting_rate*self._open_water_fraction
            g = self._thickness_distribution
            h = self._thickness_coordinates
            #calculate the self-convolution of g
            #cutoff after 'domain_size' elements because I only want values up to h_max
            g_convolution = np.convolve(g,g,'full')[:domain_size]*self._delta_h
            
            A = self._open_water_fraction
            
            ridging_term = ridging_rate * (-2*(1-A)*g+g_convolution)
            rifting_term = rifting_rate * ((integrate.trapz(g,h)-integrate.cumtrapz(g,h,initial=0)) - 0.5*h*g)
            mechanical_term = ridging_term + rifting_term
                        
            new_thickness_distribution = self._thickness_distribution + self._delta_t*mechanical_term
            #new_thickness_distribution[0] += 1/self._H_c*(ridging_rate*(1-A)**2-2*rifting_rate*A*(1-A)+rifting_rate/2*integrate.trapz(h*g,h))*self._delta_t
            #self._open_water_fraction += (ridging_rate*(1-A)**2-2*rifting_rate*A*(1-A)+rifting_rate/2*integrate.trapz(h*g,h))*self._delta_t
            #self._open_water_fraction += -integrate.trapz(mechanical_term,h)*self._delta_t
            
            self.calculate_open_water_fraction()
            #new_thickness_distribution[0] = self._open_water_fraction/self._H_c
            self._thickness_distribution = new_thickness_distribution
            #print(self._thickness_distribution[0]*self._H_c)
            #print(self._open_water_fraction)
            #print('-----------')
    
    def do_convolution_rifting(self):
        domain_size = self._thickness_coordinates.size
        if self._open_water_fraction<1:
            #solve for ice mechanical effects
            rifting_rate = self._rifting_rate*self._open_water_fraction
            g = self._thickness_distribution
            h = self._thickness_coordinates
            A = self._open_water_fraction
            
            mechanical_term = 2*rifting_rate*(integrate.trapz(g,h)-integrate.cumtrapz(g,h,initial=0)) - rifting_rate*h*g
                        
            new_thickness_distribution = self._thickness_distribution + self._delta_t*mechanical_term
            #new_thickness_distribution[0] += self._H_c*(rifting_rate*(1-A)**2*self._delta_t)
            self._open_water_fraction += -rifting_rate*integrate.trapz(g*h,h)*self._delta_t
            
            self._thickness_distribution = new_thickness_distribution
            self.calculate_open_water_fraction()
    
    def do_convolution_ridging(self):
        domain_size = self._thickness_coordinates.size
        if self._open_water_fraction<1:
            #solve for ice mechanical effects
            ridging_rate = self._ridging_rate*(1-self._open_water_fraction)
            g = self._thickness_distribution
            #calculate the self-convolution of g
            #cutoff after 'domain_size' elements because I only want values up to h_max
            g_convolution = np.convolve(g,g,'full')[:domain_size]*self._delta_h

            A = self._open_water_fraction

            mechanical_term = ridging_rate*(-2*(1-A)*g+g_convolution)

            new_thickness_distribution = self._thickness_distribution + self._delta_t*mechanical_term
            #new_thickness_distribution[0] += self._H_c*(ridging_rate*(1-A)**2*self._delta_t)
            self._open_water_fraction += ridging_rate*(1-A)**2*self._delta_t

            self._thickness_distribution = new_thickness_distribution
            self.calculate_open_water_fraction()
    
    def normalize_first_moment(self):
        #this rescales the x and y axes to change the first moment of the distribution without changing the area under the curve
        #i.e. this changes the mean thickness without changing the area fraction of ice
        #at each timestep the mean thickness erroneously increases by first_moment_change = (k_2*g(0)-k_1*(1-A))*delta_t
        #to correct for this g(h) can be transformed to a*g(a*h) where a is a scaling factor
        #you can calculate that a=1/(1-first_moment_change/overall_mean_thickness)
        
        #NOTE: this currently corrects using the advanced versions of the distribution and open water fraction. The values BEFORE advancement should really be used.
        
        if self._open_water_fraction<1:
            first_moment_change = self._delta_t*(self._k_2*self._thickness_distribution[0]-self._k_1*(1-self._open_water_fraction))
            scaling_factor =1/(1-first_moment_change/self._overall_mean_thickness)

            scaled_distribution = self._thickness_distribution*scaling_factor
            scaled_coordinates = self._thickness_coordinates/scaling_factor

            interpolator = interp1d(scaled_coordinates,scaled_distribution,fill_value='extrapolate')
            
            unsquashed_distribution = np.copy(self._thickness_distribution)
            
            squashed_distribution = interpolator(self._thickness_coordinates)
            #set all extrapolated values of the distribution to 0 to avoid any funny behaviour
            squashed_distribution[self._thickness_coordinates>scaled_coordinates.max()] = 0

            #THREE options for keeping y intercept the same
            #1. just keep g[0] the same
            #self._thickness_distribution[1:] = squashed_distribution[1:]
            #2. to avoid coordinate dependency, keep all g below a certain thickness the same
            #self._thickness_distribution[self._thickness_coordinates>self._H_c] = squashed_distribution[self._thickness_coordinates>self._H_c]
            #3. between 0 and H_c make dist a combination of squashed and unsquashed dist. Contribution of squashed increase linearly as h increases.
            #self._thickness_distribution[self._thickness_coordinates>self._H_c] = squashed_distribution[self._thickness_coordinates>self._H_c]
            #self._thickness_distribution[self._thickness_coordinates<self._H_c] = ((self._H_c-self._thickness_coordinates)/self._H_c*unsquashed_distribution+(self._thickness_coordinates)/self._H_c*squashed_distribution)[self._thickness_coordinates<self._H_c]
            
            return squashed_distribution
    
    def advance_time(self):
        #this moves the model time forward by one timestep
        new_time = self._time + self._delta_t
        self._time = new_time
        return new_time