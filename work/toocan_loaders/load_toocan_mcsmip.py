### Importation des bibliotheques
import sys
import gzip

class MCS_IntParameters(object): 

	def __init__(self):
		self.DCS_number       		= 0
		self.INT_qltyDCS	 		= 0
		self.INT_classif			= 0
		self.INT_duration           = 0 
		self.INT_UTC_timeInit       = 0.
		self.INT_localtime_Init		= 0
		self.INT_lonInit			= 0
		self.INT_latInit			= 0
		self.INT_UTC_timeEnd		= 0
		self.INT_localtime_End		= 0
		self.INT_lonEnd				= 0
		self.INT_latEnd				= 0
		self.INT_velocityAvg		= 0		
		self.INT_distance			= 0
		self.INT_lonmin				= 0
		self.INT_latmin				= 0
		self.INT_lonmax				= 0
		self.INT_latmax				= 0
		self.INT_TbMin	            = 0
		self.INT_surfmaxPix_241K	= 0
		self.INT_surfmaxkm2_241K 	= 0 
		self.INT_surfmaxkm2_220K 	= 0
		self.INT_surfmaxkm2_210K  	= 0
		self.INT_surfmaxkm2_200K 	= 0
		self.INT_surfcumkm2_241K 	= 0 
		self.INT_classif_JIRAK 	    = 0

		self.INT_Succ_40000km2 	    = 0
		self.INT_surfprecip_2mmh 	= 0
		self.INT_minPeakPrecip 	    = 0
		self.INT_totalRainVolume 	= 0
		self.INT_PF_maxAREA_0mmh 	= 0
		self.INT_classif_MCS 	    = 0

## Ajout B. Fildier ##

	def __repr__(self):
		"""Creates a printable version of the Distribution object. Only prints the 
		attribute value when its string fits is small enough."""

		out = '< MCS_IntParameters object:\n'
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

## Fin ajout B. Fildier ##

## ajout M Carenso ## 
	def __eq__(self, other):
		"""Determine if two MCS_IntParameters instances are equal."""
		if not isinstance(other, MCS_IntParameters):
            # Don't attempt to compare against unrelated types.
			return NotImplemented

		return (
            self.DCS_number == other.DCS_number and
            self.INT_qltyDCS == other.INT_qltyDCS and
            self.INT_classif == other.INT_classif and
            self.INT_duration == other.INT_duration and
            self.INT_UTC_timeInit == other.INT_UTC_timeInit and
            self.INT_localtime_Init == other.INT_localtime_Init and
            self.INT_lonInit == other.INT_lonInit and
            self.INT_latInit == other.INT_latInit and
            self.INT_UTC_timeEnd == other.INT_UTC_timeEnd and
            self.INT_localtime_End == other.INT_localtime_End and
            self.INT_lonEnd == other.INT_lonEnd and
            self.INT_latEnd == other.INT_latEnd and
            self.INT_velocityAvg == other.INT_velocityAvg and
            self.INT_distance == other.INT_distance and
            self.INT_lonmin == other.INT_lonmin and
            self.INT_latmin == other.INT_latmin and
            self.INT_lonmax == other.INT_lonmax and
            self.INT_latmax == other.INT_latmax and
            self.INT_TbMin == other.INT_TbMin and
            self.INT_surfmaxPix_241K == other.INT_surfmaxPix_241K and
            self.INT_surfmaxkm2_241K == other.INT_surfmaxkm2_241K and
            self.INT_surfmaxkm2_220K == other.INT_surfmaxkm2_220K and
            self.INT_surfmaxkm2_210K == other.INT_surfmaxkm2_210K and
            self.INT_surfmaxkm2_200K == other.INT_surfmaxkm2_200K and
            self.INT_surfcumkm2_241K == other.INT_surfcumkm2_241K and
            self.INT_classif_JIRAK == other.INT_classif_JIRAK and
            self.INT_Succ_40000km2 == other.INT_Succ_40000km2 and
            self.INT_surfprecip_2mmh == other.INT_surfprecip_2mmh and
            self.INT_minPeakPrecip == other.INT_minPeakPrecip and
            self.INT_totalRainVolume == other.INT_totalRainVolume and
            self.INT_PF_maxAREA_0mmh == other.INT_PF_maxAREA_0mmh and
            self.INT_classif_MCS == other.INT_classif_MCS
        )

## Fin ajout M Carenso ## 

class MCS_Lifecycle(object):

	def __init__(self):
		self.QCgeo_IRimage		= []
		self.LC_tbmin			= []
		self.LC_tbavg_241K 		= []
		self.LC_tbavg_210K      = []
		self.LC_tbavg_200K      = []
		self.LC_tb_90th         = []
		self.LC_UTC_time		= []
		self.LC_localtime		= []
		self.LC_lon				= []
		self.LC_lat				= []
		self.LC_x				= []
		self.LC_y				= []
		self.LC_velocity		= []
		self.LC_sminor_241K	    = []
		self.LC_smajor_241K	    = []
		self.LC_ecc_241K	    = []
		self.LC_orientation_241K= []
		self.LC_sminor_220K		= []
		self.LC_smajor_220K		= []
		self.LC_ecc_220K	    = []
		self.LC_orientation_220K	= []
		self.LC_surfPix_241K		= []
		self.LC_surfPix_210K		= []
		self.LC_surfkm2_241K		= []
		self.LC_surfkm2_220K		= []   
		self.LC_surfkm2_210K  		= []   
		self.LC_surfkm2_200K		= []   

		self.LC_surfprecip_2mmh		= []   
		self.LC_PF_rainrate 		= []   
		self.LC_PF_rainrate_0mmh	= []   
		self.LC_PF_rainrate_5mmh	= []   
		self.LC_PF_rainrate_10mmh	= []   
		self.LC_PF_AREA_0mmh    	= []   
		self.LC_PF_AREA_5mmh	    = []   
		self.LC_PF_AREA_10mmh   	= []   

## Ajout B. Fildier ##

	def __repr__(self):
		"""Creates a printable version of the Distribution object. Only prints the 
		attribute value when its string fits is small enough."""

		out = '< MCS_Lifecycle object:\n'
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

## Fin ajout B. Fildier ##

		
def load_TOOCAN(FileTOOCAN):	# SAUF POUR : Summer/OBSv7 | Summer/SCREAMv1 | Winter/OBSv7 | Winter/SCREAMv1

	lunit=gzip.open(FileTOOCAN,'rt')

	# New method for reading the Header , plus souple sur la taille du header    
	##########################
	Headers = []
	while True:
		line_h = lunit.readline()
		if line_h.startswith("#"):
			Headers.append(line_h.strip())
		else:
			break

	header_col1 = lunit.readline()   # ==>     DCS_number           INT_qltyDCS           INT_classif          INT_duration      INT_UTC_timeInit    INT_localtime_Init           INT_lonInit           INT_latInit       INT_UTC_timeEnd     INT_localtime_End            INT_lonEnd            INT_latEnd       INT_velocityAvg          INT_distance            INT_lonmin            INT_latmin            INT_lonmax            INT_latmax             INT_TbMin   INT_surfmaxPix_235K   INT_surfmaxkm2_235K   INT_surfmaxkm2_220K   INT_surfmaxkm2_210K   INT_surfmaxkm2_200K   INT_surfcumkm2_235K     INT_classif_JIRAK    INT_classif_MADDOX        INT_TSnumber_IBTRACS        INT_TSnature_IBTRACS   INT_TSmindistance_IBTRACS
	header_col2 = lunit.readline()   #        QCgeo_IRimage            LC_tbmin       LC_tbavg_235K       LC_tbavg_208K       LC_tbavg_200K          LC_tb_90th         LC_UTC_time        LC_localtime              LC_lon              LC_lat                LC_x                LC_y         LC_velocity      LC_sminor_235K      LC_smajor_235K         LC_ecc_235K LC_orientation_235K      LC_sminor_220K      LC_smajor_220K         LC_ecc_220K LC_orientation_220K     LC_surfPix_235K     LC_surfPix_210K     LC_surfkm2_235K     LC_surfkm2_220K     LC_surfkm2_210K     LC_surfkm2_200K
	header_col3 = lunit.readline()   #
	
	for head in Headers:
		print(head)
	##########################

	data = []
	iMCS = -1
	lines = lunit.readlines()

	for iline in lines: 
		Values = iline.split()

		if(Values[0] == '==>'):

			#
			# Read the integrated parameters of the convective systems
			###########################################################
			data.append(MCS_IntParameters())
			iMCS = iMCS+1
			data[iMCS].DCS_number 			= int(Values[1])	    # Label of the convective system in the segmented images
			data[iMCS].INT_qltyDCS			= int(Values[2])	    # Quality control of the convective system 
			data[iMCS].INT_classif			= int(Values[3])	    # classif
			data[iMCS].INT_duration			= float(Values[4])    	# duration of the convective system (MCSMIP: duration in hours, )
			data[iMCS].INT_UTC_timeInit		= int(Values[5])		# time TU of initiation of the convective system
			data[iMCS].INT_localtime_Init	= int(Values[6])		# local time of inititiation
			data[iMCS].INT_lonInit			= float(Values[7])		# longitude of the center of mass at inititiation
			data[iMCS].INT_latInit			= float(Values[8])		# latitude of the center of mass at inititiation
			data[iMCS].INT_UTC_timeEnd		= int(Values[9])		# time TU of dissipation of the convective system
			data[iMCS].INT_localtime_End	= int(Values[10])		# local hour of dissipation
			data[iMCS].INT_lonEnd			= float(Values[11])		# longitude of the center of mass at dissipation
			data[iMCS].INT_latEnd			= float(Values[12])		# latitude of the center of mass at dissipation
			data[iMCS].INT_velocityAvg		= float(Values[13])		# average velocity during its life cycle(m/s)
			data[iMCS].INT_distance			= float(Values[14])		# distance covered by the convective system during its life cycle(km)
			data[iMCS].INT_lonmin			= float(Values[15])		# longitude min of the center of mass during its life cycle
			data[iMCS].INT_latmin           = float(Values[16])     # latitude min of the center of mass during its life cycle
			data[iMCS].INT_lonmax			= float(Values[17])		# longitude max of the center of mass during its life cycle
			data[iMCS].INT_latmax			= float(Values[18])		# latitude max of the center of mass during its life cycle
			data[iMCS].INT_TbMin			= float(Values[19])		# minimum Brigthness temperature (K)
			data[iMCS].INT_surfmaxPix_241K	= int(Values[20])		# maximum surface for a 235K threshold of the convective system during its life cycle (pixel)
			data[iMCS].INT_surfmaxkm2_241K	= float(Values[21])		# maximum surfacefor a 235K threshold of the convective system during its life cycle (km2)
			data[iMCS].INT_surfmaxkm2_220K	= float(Values[22])		# maximum surfacefor a 235K threshold of the convective system during its life cycle (km2)
			data[iMCS].INT_surfmaxkm2_210K	= float(Values[23])		# maximum surfacefor a 235K threshold of the convective system during its life cycle (km2)
			data[iMCS].INT_surfmaxkm2_200K	= float(Values[24])		# maximum surfacefor a 235K threshold of the convective system during its life cycle (km2)
			data[iMCS].INT_surfcumkm2_241K	= float(Values[25]) 	# integrated cumulated surface for a 235K threshold of the convective system during its life cycle (km2)		
			data[iMCS].INT_classif_JIRAK    = float(Values[26]) 	# classif jirak

			data[iMCS].INT_Succ_40000km2    = float(Values[27])
			data[iMCS].INT_surfprecip_2mmh  = float(Values[28])
			data[iMCS].INT_minPeakPrecip    = float(Values[29])
			data[iMCS].INT_totalRainVolume  = float(Values[30])
			data[iMCS].INT_PF_maxAREA_0mmh  = float(Values[31])
			data[iMCS].INT_classif_MCS      = int(Values[32])

			data[iMCS].clusters = MCS_Lifecycle()

			inc = 0
		else:
			#
			# Read the parameters of the convective systems 
			#along their life cycles
			##################################################
			data[iMCS].clusters.QCgeo_IRimage.append(int(Values[0]))	    		# quality control on the Infrared image
			data[iMCS].clusters.LC_tbmin.append(float(Values[1]))	    			# min brightness temperature of the convective system at day TU (K)
			data[iMCS].clusters.LC_tbavg_241K.append(float(Values[2]))	    		# average brightness temperature of the convective system at day TU (K) 
			data[iMCS].clusters.LC_tbavg_210K.append(float(Values[3]))              # min brightness temperature of the convective system at day TU (K)
			data[iMCS].clusters.LC_tbavg_200K.append(float(Values[4]))	    	    # min brightness temperature of the convective system at day TU (K)
			data[iMCS].clusters.LC_tb_90th.append(float(Values[5]))	    			# min brightness temperature of the convective system at day TU (K)
			data[iMCS].clusters.LC_UTC_time.append(int(Values[6]))	    			# day TU 
			data[iMCS].clusters.LC_localtime.append(int(Values[7]))	    			# local hour (h)
			data[iMCS].clusters.LC_lon.append(float(Values[8]))	    				# longitude of the center of mass (°)
			data[iMCS].clusters.LC_lat.append(float(Values[9]))	    				# latitude of the center of mass (°)
			data[iMCS].clusters.LC_x.append(int(Values[10]))		    			# column of the center of mass (pixel)
			data[iMCS].clusters.LC_y.append(int(Values[11]))		    			# line of the center of mass(pixel)
			data[iMCS].clusters.LC_velocity.append(float(Values[12]))	    		# instantaneous velocity of the center of mass (m/s)
			data[iMCS].clusters.LC_sminor_241K.append(float(Values[13]))	 
			data[iMCS].clusters.LC_smajor_241K.append(float(Values[14]))	 
			data[iMCS].clusters.LC_ecc_241K.append(float(Values[15]))
			data[iMCS].clusters.LC_orientation_241K.append(float(Values[16])) 
			data[iMCS].clusters.LC_sminor_220K.append(float(Values[17]))	 
			data[iMCS].clusters.LC_smajor_220K.append(float(Values[18]))	 
			data[iMCS].clusters.LC_ecc_220K.append(float(Values[19]))
			data[iMCS].clusters.LC_orientation_220K.append(float(Values[20])) 
			data[iMCS].clusters.LC_surfPix_241K.append(int(Values[21]))	    		# surface of the convective system at time day TU (pixel)
			data[iMCS].clusters.LC_surfPix_210K.append(int(Values[22]))	    		# surface of the convective system at time day TU (pixel)
			data[iMCS].clusters.LC_surfkm2_241K.append(float(Values[23]))  			# surface of the convective system for a 241K threshold
			data[iMCS].clusters.LC_surfkm2_220K.append(float(Values[24]))  			# surface of the convective system for a 220K threshold
			data[iMCS].clusters.LC_surfkm2_210K.append(float(Values[25]))  			# surface of the convective system for a 210K threshold
			data[iMCS].clusters.LC_surfkm2_200K.append(float(Values[26]))  			# surface of the convective system for a 200K threshold

			data[iMCS].clusters.LC_surfprecip_2mmh.append(float(Values[27]))
			data[iMCS].clusters.LC_PF_rainrate.append(float(Values[28]))
			data[iMCS].clusters.LC_PF_rainrate_0mmh.append(float(Values[29]))
			data[iMCS].clusters.LC_PF_rainrate_5mmh.append(float(Values[30]))
			data[iMCS].clusters.LC_PF_rainrate_10mmh.append(float(Values[31]))
			data[iMCS].clusters.LC_PF_AREA_0mmh.append(float(Values[32]))
			data[iMCS].clusters.LC_PF_AREA_5mmh.append(float(Values[33]))
			data[iMCS].clusters.LC_PF_AREA_10mmh.append(float(Values[34]))

	lunit.close()    
	return data    
