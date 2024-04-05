"""FEEDS FramEwork for Early Detection Systems

Provides classes for data managing, triggering model and cross-validation
......
....

"""
__author__ = "Flavio Cannavo and Vittorio Minio"
__credits__ = ["DPC-INGV Allegato B 2021-2023 Task 3 Project"]
__license__ = "GPL"
__version__ = "0.9.1"
__maintainer__ = "Flavio Cannavo and Vittorio Minio"
__email__ = "{flavio.cannavo, vittorio.minio}@ingv.it"
__status__ = "Production"

# Import main packages 
import pandas as pd
import numpy as np
import random
from itertools import groupby

class Data_manager:
    """  
    Class that defines the data batch for single alert checking.
    """        
    time_window = 600  # Time window for the data batch in seconds
    n_samples = 600    # Number of samples in the data batch
    
    minimum_segment_length = 600  # Minimum segment length in seconds
    dataset_periods = None         # Array-like of timestamps UTC [start_time, end_time]
    folder_save =None              # Directory containing the data 
# -----------------------------------------------------------------------------   
    def get_batch_info(self):
        """  
        Retrieve the data batch info for single alert checking.
        
        Returns:
            Tuple (time_window, n_samples): Time window in seconds and number of samples in the batch.
        """   
        return self.time_window, self.n_samples
# -----------------------------------------------------------------------------   
    def get_dataset_periods(self):
        """  
        Retrieve the dataset periods for single alert checking.
        
        Returns:
            Array-like of timestamps UTC [start_time, end_time].
        """   
        return self.dataset_periods.copy()
# -----------------------------------------------------------------------------   
    def get_data(self, tstart, tstop):
        """  
        Retrieve the data within a specified period.
        
        Args:
            tstart (timestamp UTC): Start time of the period.
            tstop (timestamp UTC): End time of the period.
            
        Returns:
            Pandas DataFrame: Dataset indexed by timestamp UTC.
        """   
        return None
# -----------------------------------------------------------------------------   
    def get_batch(self, t):
        """  
        Retrieve the data batch at a specific UTC timestamp.
        
        Args:
            t (timestamp UTC): Timestamp for the data batch.
            
        Returns:
            Pandas DataFrame: Data batch indexed by timestamp UTC.
        """   
        return None
# -----------------------------------------------------------------------------   
    def get_minimum_segment_length(self):
        """  
        Retrieve the minimum segment length.
        
        Returns:
            Minimum segment length in seconds.
        """   
        return self.minimum_segment_length   
# -----------------------------------------------------------------------------
    def check_events(self, events):
        """
        Performs a check on all events trying to exclude those events showing some
        timing problems with the available data.
    
        Args:
            events(ndarray): list of events containig the starttime and the endtime in UTC timestamp.  
            dataset_periods (ndarray): list of periods containig the starttime and the endtime in UTC timestamp.    
    
        output:
            events_checked(ndarray): list of events without problems. 
    
        """
        # Retrivies timing information
        dataset_periods= self.dataset_periods
        minimum_batch_length,_=self.get_batch_info() 
            
        # Checks if there is overlapping among events     
        overlapisok = not(np.any(np.less(np.diff(events.flatten()),0)))
          
        events_checked = np.empty((0,2))
        events_over = np.empty((0,2))
        
        # Activate the next control only if there are non overlapped events
        if overlapisok: 
            
            # Checks if the events are too close to the interval borders
            for period in dataset_periods:
                
                # Searches all events included in the period 
                event_filt = events[np.logical_and(events[:,0]>=period[0], events[:,1]<=period[1]),:]
                
                # Checks if there is almost one event in the period
                if event_filt.shape[0]>0:
                    evchk = event_filt.copy()
                    
                    if np.any(event_filt[:,0] - period[0] < minimum_batch_length):
                        
                        # Deletes the events very nearby the limits (< batch window) 
                        idx_bords= np.where(event_filt[:,0] - period[0] < minimum_batch_length)[0]
                        evchk= np.delete(event_filt.copy(), idx_bords, axis=0)
                    
                    # Stores the events
                    events_checked = np.row_stack((events_checked, evchk))
        else:
            # Retrievies the overlapped events 
             idx_over = np.unique(np.round(np.where(np.less(np.diff(events.flatten()),0))[0]/2))        
             events_over = events[idx_over.astype('int'),:]
             print('Warning! One or more events are overllaped')
        
        #Sorts the events     
        idx = np.argsort(events_checked[:,0])
        events_checked = events_checked[idx]            
        return events_checked, events_over     
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
class EW_Trigger:
    """  
    Class for building the EW model and retrieving warnings.
    """    
    
    parameters = None # parameters used to trigger an alert, can be a objects or list of float values.
    
    def __init__(self, **kwargs):
        """
        Initialize the EW_Trigger class with threshold values.

        Args:
            **kwargs: Arbitrary keyword arguments.
        """
        self.parameters = None
# -----------------------------------------------------------------------------        
    def checkdata(self, databatch, **kwargs):
        """
        Check the data quality.

        Args:
            databatch: Data to analyze, can be a databatch object or a dataset in pandas indexed by UTC timestamp.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            bool: True if the data can be processed.
        """  
        # Placeholder implementation, should be customized by the user
        return True
# -----------------------------------------------------------------------------
    def model(self, databatch, **kwargs):
        """
        Retrieve warnings based on provided data.

        Args:
            databatch: Data to analyze, can be a databatch object or a dataset in pandas indexed by UTC timestamp.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            model object or parameters used to detect alerts.
        """  
        # Placeholder implementation, should be customized by the user
        return False
# -----------------------------------------------------------------------------
    def trigger(self, databatch, **kwargs):
        """
        Retrieve warnings based on provided data.

        Args:
            databatch: Data to analyze, can be a databatch object or a dataset in pandas indexed by UTC timestamp.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            bool: True if a warning is triggered, False otherwise.
        """  
        # Check data quality before invoking the model
        if self.checkdata(databatch, kwargs):
            return self.model(databatch, kwargs)
        return None
# -----------------------------------------------------------------------------
    @classmethod
    def create_model(cls, data_manager, train_periods, train_targets, **kwargs):
        """  
        Constructs an Early Warning System (EWS) based on available data
        and target events. 
        
        Args:
            data_manager (Data_manager): Class to load data from a period.
            train_periods (array of [tstart, tstop]): Array of UTC timestamps used to build a model.
            train_targets: Array Nx2 of timestamps with begins and ends of target events to early detect.
            **kwargs: Arbitrary keyword arguments.           
        Returns:
            EW_Trigger: EWS model.
        """
        # Placeholder implementation, should be customized by the user
        return cls(param=None)   
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
class EW_Xvalidation():
    """
    Class that evaluates model performance using cross-validation.
    """
    
    data_manager = None # Class to load data from a period.
    target_events_times = None # Array Nx2 of timestamps with begins and ends of target events to early detect.
    
    SPLIT_NTRIALS = 400 # Number of trials to split the dataset
# -----------------------------------------------------------------------------    
    def __init__(self,  data_manager, target_events_times):
        """
        Initialize the EW_Xvalidation class with a list of results.

        Args:
            data_manager (Data_manager): Class to load data from a period.
            target_events_times: Array Nx2 of timestamps with begins and ends of target events to early detect.
        """
        
        # Check the target events using the DataManager class method
        self.data_manager = data_manager 
        target_events_checked, events_over = data_manager.check_events(target_events_times)
        
        # Set the adjusted target events
        self.target_events_times = target_events_checked 
# -----------------------------------------------------------------------------
    def _segment_trim(self, t0, t1, events, batch_length, period):
        """
        Trims the segment [t0, t1] based on the provided events, batch length, and period.
    
        Args:
            t0 (float): Start time of the segment.
            t1 (float): End time of the segment.
            events (numpy.ndarray): Array containing event in UTC timestamps.
            batch_length (float): Length of the batch in seconds.
            period (numpy.ndarray): Array representing the period [start, end] in UTC timestamp.
    
        Returns:
            tuple: Trimmed segment in UTC timestamps (t0, t1).
        """
        # Find events that overlap with the start time of the segment
        idx_start_overlap = np.where((events[:, 0] < t0) & (events[:, 1] > t0))[0]
    
        # Adjust t0 based on the earliest overlapping event
        if len(idx_start_overlap) > 0:
            t0 = np.max([np.min(events[idx_start_overlap, 0]) - batch_length, period[0]])
    
        # Find events that overlap with the end time of the segment
        idx_end_overlap = np.where((events[:, 0] < t1) & (events[:, 1] > t1))[0]
    
        # Adjust t1 based on the latest overlapping event
        if len(idx_end_overlap) > 0:
            t1 = events[idx_end_overlap, 1]
    
        return t0, t1     
# -----------------------------------------------------------------------------
    def _check_segment(self, t0, t1, length, nevents, events, minimum_segment_length, batch_length):
        """
        Checks if the segment [t0, t1] meets the criteria for a valid segment.
    
        Args:
            t0 (float): Start time of the segment in UTC timestamps.
            t1 (float): End time of the segment in UTC timestamps.
            length (float): Length of the segment.
            nevents (int): Number of events.
            events (numpy.ndarray): Array containing event in UTC timestamps.
            minimum_segment_length (float): Minimum allowable segment length in seconds.
            batch_length (float): Length of the batch in seconds.
    
        Returns:
            tuple: A boolean indicating whether the segment is valid, and an integer code (-1 to -4) indicating the reason if not.
        """
        # Check if the segment length is less than the fixed minimum length
        if (t1 - t0) < minimum_segment_length:
            return False, -1
    
        # Check if the remaining piece after trimming cannot be shorter than the fixed minimum length
        if length - (t1 - t0) < minimum_segment_length:
            return False, -2
    
        # Find events that fall within the segment time range
        idx_events_within_segment = np.where((events[:, 0] >= t0) & (events[:, 0] <= t1))[0]
    
        if len(idx_events_within_segment) > 0:
            # Check if the number of events in the segment exceeds the specified limit
            if len(idx_events_within_segment) > nevents:
                return False, -3
    
            # Check if the earliest event in the segment is within the batch length from the start of the segment
            if np.min(events[idx_events_within_segment, 0]) <= (t0 + batch_length):
                return False, -4
    
        return True, len(idx_events_within_segment)                 
# -----------------------------------------------------------------------------
    def _searchsplit(self, periods, events, length, nevents, segments, minimum_segment_length=600, batch_length=60, trials=400):
        """
        Search for valid segments within the specified periods.
    
        Args:
            periods (numpy.ndarray): Array containing time periods in UTC timestamps.
            events (numpy.ndarray): Array containing event in UTC timestamps.
            length (float or numpy.ndarray): Length of the segments to be searched.
            nevents (int): Number of events.
            segments (list): List to store the found segments.
            minimum_segment_length (float): Minimum allowable segment length in seconds. Default is 600 seconds.
            batch_length (float): Length of the batch in seconds. Default is 600 seconds.
            trials (int): Number of trials to search for segments. Default is 400 trials.
    
        Returns:
            tuple: List of valid segments, remaining length, and remaining number of events.
        """
        # Checks the classes of the input parameters and selects the correct values
        length = length[0] if isinstance(length, np.ndarray) else length
    
        if np.min(np.shape(periods) == 1):
            periods = periods[0]
    
        if np.ndim(periods) == 1:
            period = periods
    
            # Check if the period is long enough
            if np.diff(period) > minimum_segment_length:
    
                for i in range(trials):
                    # Generate random start and end times within the given period
                    t0 = random.uniform(period[0], period[1] - minimum_segment_length)
                    t1 = random.uniform(t0, np.min([period[1], t0 + length]))
                    
                    # Checks the classes of the input parameters and selects the correct values
                    t0 = t0[0] if isinstance(t0, np.ndarray) else t0
                    t1 = t1[0] if isinstance(t1, np.ndarray) else t1
    
                    # Trims the segment to ensure it is within the event timestamps and the batch length
                    t0, t1 = self._segment_trim(t0, t1, events, batch_length, period)
                    
                    # Checks the classes of the input parameters and selects the correct values
                    t0 = t0[0] if isinstance(t0, np.ndarray) else t0
                    t1 = t1[0] if isinstance(t1, np.ndarray) else t1
    
                    # Check if the segment is valid
                    found, info = self._check_segment(t0, t1, length, nevents, events, minimum_segment_length, batch_length)
                    
                    # Breaks the loop if one segment is found
                    if found:
                        break
    
                if found:
                    # Append the valid segment to the list
                    segments.append([t0, t1])
                    nevents -= info
                    length -= t1 - t0
    
                    # Update the remaining periods based on the newly found segment
                    if t0 > period[0]:
                        if t1 < period[1]:
                            periods = np.array([[period[0], t0], [t1, period[1]]])
                        else:
                            periods = np.array([[period[0], t0]])
                    else:
                        if t1 < period[1]:
                            periods = np.array([[t1, period[1]]])
    
                    # Recursive call to search for more segments
                    return self._searchsplit(
                        periods, events, length, nevents, segments,
                        minimum_segment_length=minimum_segment_length,
                        batch_length=batch_length
                    )
                else:
                    # Return the current state if no valid segment is found
                    return segments, length, nevents
    
            else:
                # Return the current state if the period is not long enough
                return segments, length, nevents
    
        else:
            # Shuffle the periods for randomness
            np.random.shuffle(periods)
            for period in periods:
                # Recursive call for each period
                segments, length, nevents = self._searchsplit(
                    period, events, length, nevents, segments,
                    minimum_segment_length=minimum_segment_length,
                    batch_length=batch_length
                )
    
            return segments, length, nevents   
# -----------------------------------------------------------------------------
    def _do_splitting(self, dataset_periods, events, fraction, num_events, segments,
                      fraction_tol=0.1, nevents_tol=0, batch_length=600, minimum_segment_length=600):
        """
        Perform the splitting of the dataset based on specified criteria.
    
        Args:
            dataset_periods (numpy.ndarray): Array containing dataset periods in UTC timestamps.
            events (numpy.ndarray): Array containing event in UTC timestamps.
            fraction (float): Fraction of the dataset to consider.
            num_events (int): Number of events.
            segments (list): List to store the found segments.
            fraction_tol (float): Tolerance for the fraction of the dataset. Default is 0.1 (1%). 
            nevents_tol (int): Tolerance for the number of events. Default is 0.
            batch_length (float): Length of the batch in seconds. Default is 600 seconds.
            minimum_segment_length (float): Minimum allowable segment length in seconds.Default is 600 seconds.
    
        Returns:
            numpy.ndarray or None: Array containing the found segments or None if no valid segments are found.
        """
        # Initialize the flag to indicate whether valid segments are found
        found = False
    
        # Perform the splitting for a specified number of trials
        for i in range(self.SPLIT_NTRIALS):
            # Calculate the length of the segment based on the specified fraction
            length = fraction * np.diff(dataset_periods).sum()
            nevents = num_events
            segments = []
            
            # Call the internal method _searchsplit to find segments
            segments, length_residual, nevents = self._searchsplit(
                dataset_periods, events, length, nevents, segments,
                batch_length=batch_length, minimum_segment_length=minimum_segment_length
            )
    
            # Calculate the actual fraction of the dataset covered by the found segments
            actual_fraction = length_residual / (fraction * np.diff(dataset_periods).sum())
    
            # Check if the criteria for a valid split are met:
            # nevents_tol behavior: nevents_tol>=0  maximum tolerance, nevents_tol<0 minimum acceptance 
            if (actual_fraction < fraction_tol) and (((nevents_tol >= 0) and (nevents <= nevents_tol))
                                                     or ((nevents_tol < 0) and ((num_events - nevents) >= -nevents_tol))):
                found = True
                break
    
        # Check if valid segments are found
        if found:
            # Sort the found segments based on the starting time
            segments = np.array(segments)
            idx = np.argsort(segments[:, 0])
            segments = segments[idx]
            return segments
        else:
            # Return None if no valid segments are found
            return None               
# -----------------------------------------------------------------------------
    def _events_in_periods(self, segments, events):
        """
        Check which events are contained within specified segments.
    
        Args:
            segments (numpy.ndarray): Array containing segments with start and end times in UTC timestamps.
            events (numpy.ndarray): Array containing event in UTC timestamps.
    
        Returns:
            numpy.ndarray: Boolean array indicating whether each event is contained within any segment.
        """
        # Extract start and end times of events and segments
        event_start_times = events[:, 0]
        event_end_times = events[:, 1]
    
        segment_start_times = segments[:, 0]
        segment_end_times = segments[:, 1]
    
        # Create a boolean array indicating whether each event is contained within any segment
        event_contained = np.logical_and(
            segment_start_times < event_start_times[:, np.newaxis],
            segment_end_times >= event_end_times[:, np.newaxis]
        )
    
        # Check if any of the events are contained within the segments
        idx_events_in = np.any(event_contained, axis=1)
    
        return idx_events_in       
# -----------------------------------------------------------------------------
    def _split_traintest_periods(self, train_nevents=2, train_fraction=0.7, test_fraction=0.2):
        """
        Split the dataset periods into proper chunks and return train and test periods.
    
        Args:
            train_nevents (int): Number of events to include in the training set. Default is 2 events. 
            train_fraction (float): Fraction of the dataset to use for training. Default is 0.7 (70%).
            test_fraction (float): Fraction of the dataset to use for testing. Default is 0.2 (20%).
    
        Returns:
            tuple: Four numpy.ndarray objects representing train_segments, train_events_times, test_segments, and test_events_times.
                   Returns None if the split is unsuccessful.
        """
        # Retrieve dataset information
        dataset_periods = self.data_manager.get_dataset_periods()
        batch_time_window, batch_n_samples = self.data_manager.get_batch_info()
        minimum_seg_len = self.data_manager.get_minimum_segment_length()
    
        splitisok = False
    
        # Attempt multiple splits
        for i in range(int(np.ceil(self.SPLIT_NTRIALS / 10))):
            # Split: randomly select events for training
            train_segments = []
            train_segments = self._do_splitting(
                dataset_periods, self.target_events_times, train_fraction, train_nevents,
                train_segments, batch_length=batch_time_window, minimum_segment_length=minimum_seg_len
            )
            
            # Continue if the training segments are found
            if train_segments is not None:
                # Check events within training segments
                idx_events = self._events_in_periods(train_segments, self.target_events_times)
                train_events_times = self.target_events_times[idx_events]
    
                # Combine dataset and train_segments for remaining periods calculation
                times = np.append(dataset_periods, train_segments.copy(), axis=0)
                remaining_periods = np.reshape(np.sort(times.flatten()), (-1, 2))
    
                # Calculate actual test fraction based on remaining periods
                actual_test_fraction = test_fraction * np.sum(np.diff(dataset_periods)) / np.sum(np.diff(remaining_periods))
    
                # Calculate the number of events for testing
                test_nevents = self.target_events_times.shape[0] - train_events_times.shape[0]
    
                # Split remaining periods for testing
                test_segments = self._do_splitting(
                    remaining_periods, self.target_events_times, actual_test_fraction, test_nevents,
                    train_segments, batch_length=batch_time_window, minimum_segment_length=minimum_seg_len, nevents_tol=-1
                )
                
                # Continue if the testing segments are found
                if test_segments is not None:
                    splitisok = True
                    # Check events within testing segments
                    idx_events_in = self._events_in_periods(test_segments, self.target_events_times)
                    test_events_times = self.target_events_times[idx_events_in]
                    break
    
        # Return None if the split is unsuccessful
        if not splitisok:
            return None, None, None, None
    
        return train_segments, train_events_times, test_segments, test_events_times      
# -----------------------------------------------------------------------------
    def _simulate(self, model, test_segments, simulation_step=600):
        """
        Simulate model triggers on test segments.
    
        Args:
            model: Model object with a trigger method retrivied from EW_Trigger.
            test_segments (numpy.ndarray): Array representing test segments in the form of [tstart, tend] in UTC timestamps.
            simulation_step (int): Time step for simulation in seconds. Default is 600 seconds.
    
        Returns:
            numpy.ndarray: Array containing simulated warnings over time in the form [timeline (UTC timestamps), simulation (0 or 1)].
        """
        # Get batch information
        batch_time_window, batch_n_samples = self.data_manager.get_batch_info()
    
        simulation = []
        timeline = []
    
        # Iterate through test segments
        for segment in test_segments:
            t = segment[0] + batch_time_window + 1
    
            # Simulate triggers over the segment
            while t < segment[1]:
                databatch = self.data_manager.get_batch(t)
                # Retrivies the response 
                warning = -1 if (ew := model.trigger(databatch)) is None else int(ew)
                #Stores the results
                timeline.append(t)
                simulation.append(warning)
    
                t += simulation_step
        # Creates the array time-warnings
        simulation = np.column_stack([timeline, simulation])
        return simulation
      
# -----------------------------------------------------------------------------
    @staticmethod
    def calculate_performances(simulation, events_times, hysteresis):
        """
        Calculate performance metrics based on calculated warnings and actual event times.
    
        Args:
            simulation (numpy.ndarray): warnings in the form [timeline (UTC timestamps), simulation (0 or 1)].
            events_times (list): List of actual event times in UTC timestamps.
            hysteresis (float): Hysteresis value in seconds.
    
        Returns:
              TPR (float): True Positive Rate.
               FDR (float): False Discovery Rate.
               TA (float): Total Alert Time in seconds.
               leadTimes (list): Lead times for detected events in seconds.
               tFound (list): Times for correctly detected events in UTC timestamps.
               tDiscarded (list): Times for discarded events in UTC timestamps.
               tFalse (list): Times for falsely detected events UTC timestamps.
        """
        def apply_hysteresis_to_simulation(simulation, hysteresis):
            """
            Apply hysteresis to the calculated warnings.
    
            Args:
                simulation (numpy.ndarray): warnings in the form [timeline (UTC timestamps), simulation (0 or 1)].
                hysteresis (int): Hysteresis value in seconds.
    
            Returns:
                simulation(numpy.ndarray): Updated warnings. 
            """
            # Make a copy of the simulation array to avoid modifying the original array
            simulation = simulation.copy()
            
            # Find indices where the warning value is 1 (indicating an alert)
            itrig = np.where(simulation[:,1]>0)[0]
            
            # Iterate over the indices where warnings are triggered
            for i in itrig:
                    
                # Adds hysteresis to the warning time series
                pin = np.where((simulation[:,0]>=simulation[i,0]) &
                               (simulation[:,0]<=(simulation[i,0]+hysteresis)))[0]
                
                # Sets the warning values within the hysteresis window to 1 (indicating an alert) 
                if len(pin)>0:
                    simulation[pin,1] = 1                    
                    simulation = np.row_stack((simulation, [simulation[i,0]+hysteresis, 1]))        
            
            # Sorts the warnings in time 
            simulation = np.row_stack((simulation, [simulation[0,0]-hysteresis, 0]))
            idxsorted = np.argsort(simulation[:,0])
            simulation = simulation[idxsorted]
            
            return simulation
    
        def simulation_groupby(simulation):
            """
               Extracts time intervals of alerts from warning time series.
            
               Args:
                   simulation (numpy.ndarray): Warnings in the form [timeline (UTC timestamps), simulation (0 or 1)].
            
               Returns:
                   alerts (list): List of time intervals in UTC timestamps where warnings are triggered.
            """ 
            # Make a copy of the simulation array to avoid modifying the original array
            simulation = simulation.copy() 
            
            # Set values less than 0 to 0 (indicating  a no alert)
            simulation[simulation[:,1]<0,1] = 0
            
            # Initialize the indexes and the list of alerts
            idx = 0           
            alerts = []
            
            # Group consecutive values of 1 in alerts periods 
            for key, sub in groupby(simulation[:,1]):
                
                # If the key is 1 (indicating an alert), append the corresponding interval to the list of alerts
                ele = len(list(sub))               
                if key==1:
                    alerts.append([simulation[idx,0], simulation[idx + ele - 1,0]])
                
                # Update the index
                idx += ele
               
            return alerts

        def intervals_overlaps(x, y):
            """
               Check if two intervals are overlapped.
            
               Args:
                   x (list): First interval [start, end].
                   y (list): Second interval [start, end].
            
               Returns:
                   bool: True if the intervals overlap, False otherwise.
            """
            return min(x[1], y[1]) - max(x[0], y[0]) >= 0

        # Convert inputs to NumPy arrays for efficient operations
        events_times = np.array(events_times)
        simulation = np.array(simulation)
    
        # Apply hysteresis to the warning time series 
        simulation = apply_hysteresis_to_simulation(simulation, hysteresis)
    
        # Initialize variables for performance metrics
        Ntot = events_times.shape[0]
        Nok = 0
        Ndiscarded = 0
        dTs = []
        tMissed = []
        tFound = []
        tDiscarded = []
    
        # Iterate over target events
        for event_times in events_times:
            # Find indices of warning data within the time window of the current event
            iewin = np.where((simulation[:,0] >= event_times[0]) & (simulation[:,0] <= event_times[1]))[0]
    
            # Check if no alert is within the time window
            if len(iewin) == 0:
                Ndiscarded += 1
                tDiscarded.append(event_times)
                Ntot -= 1
            else:
                # Find indices of triggered events within the time window
                iew = np.where(simulation[iewin,1] > 0)[0]
    
                # Check if no triggered events are found
                if len(iew) == 0:
                    tMissed.append(event_times)
                else:
                    Nok += 1
                    # Find the index of the last non-triggered event before the current event
                    itrigg = np.where(simulation[:iewin[iew[0]],1] < 1)[0][-1] + 1
                    # Calculate lead time and record the detected event time
                    dTs.append(event_times[0] - simulation[itrigg,0])
                    tFound.append(event_times)
    
        # Initialize variables for false positive detection analysis
        Nfalse = 0
        tFalse = []
        
        # Extract time intervals of alerts 
        alerts_times = simulation_groupby(simulation)
    
        # Iterate over time intervals of alerts
        for alert_times in alerts_times:
            # Check if the alert overlaps with any target event
            itok = np.where([intervals_overlaps(alert_times, e) for e in events_times])[0]
    
            # If no overlap is found, record the false positive
            if len(itok) == 0:
                Nfalse += 1
                tFalse.append(alert_times)
    
        # Calculate performance metrics
        TPR = Nok / Ntot if Ntot > 0 else np.nan # True Positive Rate
        leadTimes = dTs # Lead times 
        FDR = Nfalse / len(alerts_times) if len(alerts_times) > 0 else np.nan # False Discovery Rate
        TA = np.diff(alerts_times).sum() #Time of Alerts
    
        return TPR, FDR, TA, leadTimes, tFound, tDiscarded, tFalse        
# -----------------------------------------------------------------------------
    def crossvalidate(self, ew_trigger, train_fraction=0.7, train_nevents=2, 
                      test_fraction=0.2, simulation_step=600, nfolds=1, hysteresis=3600):
        """
        Creates K models using cross-validation and calculates their performances 
    
        Args:
            ew_trigger (EW_Trigger): Early Warning trigger class used to create the model for the alert detection.
            train_fraction (float): Fraction of data used for training. Default is 0.7 (70%).
            train_nevents (int): Number of events to include in the training set.
            test_fraction (float): Fraction of data used for testing. Default is 0.2 (20%).
            simulation_step (int): Simulation step in seconds. Default is 600 seconds.
            nfolds (int): Number of folds for cross-validation. Default is 1 fold.
            hysteresis (int): Hysteresis value in seconds. Default is 3600 seconds.
    
        Returns:
            performances (pd.DataFrame): Performance metrics for each fold.
        """
        performances = None
    
        for ix in range(nfolds):
            # Splits the dataset into training and testing sets  
            train_segments, train_events_times, test_segments, test_events_times = \
                     self._split_traintest_periods(train_nevents=train_nevents, train_fraction=train_fraction, 
                                                   test_fraction=test_fraction)
                     
            # Bulids the model for the detctions of the alerts 
            model = ew_trigger.create_model(self.data_manager, train_segments, train_events_times)
    
            # Performs a simulation on th testing set and calculates the performance of the model
            test_simulation = self._simulate(model, test_segments, simulation_step)
            TPR, FDR, TA, LeadTimes, tEventFound, tEventDiscarded, tFalseAlert = \
                self.calculate_performances(test_simulation, test_events_times, hysteresis)
            
            # Computes the Fraction of Time Alerted (FTA)
            FTA = TA / np.diff(test_segments).sum()
    
            # Stores the computed metrics in a pandas DataFrame 
            fold = {
                'index': ix,
                'train_segments': train_segments,
                'train_events_times': train_segments,
                'test_segments': test_segments,
                'test_events_times': test_events_times,
                'TPR': TPR,
                'FDR': FDR,
                'FTA': FTA,
                'LT': LeadTimes,
                'TS': test_simulation
            }
            
            performances = pd.DataFrame.from_dict(fold, orient='index') if performances is None \
                        else pd.concat([performances, pd.DataFrame.from_dict(fold, orient='index')], axis=1)
        # Rename columns
        performances.columns = np.arange(nfolds)
        return performances


