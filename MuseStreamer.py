import os
import pandas as pd
from pylsl import StreamInlet, resolve_byprop  # Module to receive EEG data
import utils 
import numpy as np
import signal
import sys
# import threading
import datetime as dt
import time
from muselsl import muse
# from muselsl.constants import LSL_EEG_CHUNK


def keyboardInterruptHandler(sig, frame):
    print(f"\nKeyboardInterrupt (ID: {sig}) has been caught. Cleaning up...")
    sys.exit()


signal.signal(signal.SIGINT, keyboardInterruptHandler)

# these parameters are remaind from the ancestor of this code (orig: muselsl examples folder)

'''
EXPERIMENTAL PARAMETERS, Modify these to change aspects of the signal processing:

BUFFER_LENGTH - Length of the EEG data buffer (in seconds). This buffer will hold last n seconds of data and be 
                used for calculations
                Orig val: 2
EPOCH_LENGTH - Length of the epochs used to compute the FFT (in seconds)
               Orig val: 1
OVERLAP_LENGTH - Amount of overlap between two consecutive epochs (in seconds) 
                 Orig val: 0.8
SHIFT_LENGTH - Amount to 'shift' the start of each next consecutive epoch 
INDEX_CHANNEL - Index of the channel(s) (electrodes) to be used
                [0 = left ear, 1 = left forehead, 2 = right forehead, 3 = right ear]
                Orig val: [1]
'''
BUFFER_LENGTH = 2
EPOCH_LENGTH = 1
OVERLAP_LENGTH = 0.8
SHIFT_LENGTH = EPOCH_LENGTH - OVERLAP_LENGTH
N_WIN_TEST = int(np.floor((BUFFER_LENGTH - EPOCH_LENGTH) / SHIFT_LENGTH + 1))


INDEX_CHANNEL = [0, 1, 2, 3]
MAX_DF_LENGTH = 100000
EEG_CHUNK = 128

ADDRESS = "00:55:da:b7:44:b0"


def get_full_file():
    return f"streams/muse_stream_{dt.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}.csv"


# def marker_lookup(timeout):
#     print("Looking for a Markers stream...")
#     marker_streams = resolve_byprop('name', 'Markers', timeout=timeout)
#     if marker_streams:
#         inlet_marker = StreamInlet(marker_streams[0])
#     else:
#         inlet_marker = False
#         print("Can't find Markers stream.")
#     return inlet_marker


class Band:
    # Handy little enum to make code more readable
    Delta = 0
    Theta = 1
    Alpha = 2
    Beta = 3


class MuseStreamer:
    def __init__(self, address, print_data=False, print_band_power=False, print_metrics=False):

        self.has_stream = False
        self.stop = False
        self.print_data = print_data
        self.print_band_power = print_band_power
        self.print_metrics = print_metrics

        self.muse = muse.Muse(address=address)
        self.muse.connect()
        self.muse.start()

        self.inlet = self.stream_listener(timeout=5, max_chunklen=20)

        self.ch_names = self.get_ch_names(self.inlet.info().channel_count())
        self.fs = int(self.inlet.info().nominal_srate())  # Should be 256 for muse 2016

        self.data_frame = pd.DataFrame(columns=['Timestamp']+self.ch_names)
        self.filter_state = None

        self.band_buffer = np.zeros((N_WIN_TEST, 4))
        self.smooth_band_powers = None

    def stream_listener(self, timeout=2, max_chunklen=12):
        print('[Looking for an EEG stream...]')
        while not self.has_stream:
            streams = resolve_byprop('type', 'EEG', timeout=timeout)
            if len(streams) < 1:
                self.muse.start()
                print(f'- Can\'t find EEG stream, trying again. timeout = {timeout} second[s]')
            else:
                print(f"[Stream acquired]")
                self.has_stream = True
                print("[Ready to start acquiring data]")
                return StreamInlet(streams[0], max_chunklen)

    def get_ch_names(self, n_chan):
        ch = self.inlet.info().desc().child('channels').first_child()
        ch_names = [ch.child_value('label')]
        for i in range(1, n_chan):
            ch = ch.next_sibling()
            ch_names.append(ch.child_value('label'))
        return ch_names

    def get_data_chunk(self, print_flag=False, return_flag=False):
        eeg_data, timestamp = self.inlet.pull_chunk(timeout=1, max_samples=EEG_CHUNK)
        channel_data = np.array(eeg_data)[:, INDEX_CHANNEL]
        timestamp = np.array(timestamp) + self.inlet.time_correction()

        dc_df = pd.DataFrame(data=np.c_[timestamp, channel_data], columns=['Timestamp']+self.ch_names)
        self.df_size_order(dc_df)

        if print_flag:
            dc_df['Timestamp'] = [str(tm.time()) for tm in list(map(dt.datetime.fromtimestamp, dc_df['Timestamp']))]
            print(dc_df.head())
        if return_flag: return eeg_data, timestamp, dc_df

    def df_size_order(self, dc_df):
        self.data_frame = pd.concat((self.data_frame, dc_df), axis=0)
        self.data_frame = self.data_frame.sort_values(axis=0, by='Timestamp')
        if len(self.data_frame) > MAX_DF_LENGTH:
            self.data_frame = self.data_frame.tail(MAX_DF_LENGTH)

    # TODO: Fix - let choose channels (and opt.average)
    def comp_band_power(self, print_flag=False):
        data_epoch = self.data_frame[self.ch_names]
        band_powers = utils.compute_band_powers(data_epoch, fs=self.fs)

        self.band_buffer, _ = utils.update_buffer(self.band_buffer, np.asarray([band_powers]))
        if print_flag:
            print(f'Delta: {band_powers[Band.Delta]:.5f},   Theta: {band_powers[Band.Theta]:.5f},   '
                  f'Alpha: {band_powers[Band.Alpha]:.5f},   Beta: {band_powers[Band.Beta]:.5f}')
        self.smooth_band_powers = np.mean(self.band_buffer, axis=0)

    # TODO: Fix
    def comp_nf_metrics(self, print_flag=False):
        """
          - Alpha Protocol: Simple redout of alpha power, divided by delta waves in order to rule out noise

          - Beta Protocol: Beta waves have been used as a measure of mental activity and concentration
                          This beta over theta ratio is commonly used as neurofeedback for ADHD

          - Alpha/Theta Protocol: This is another popular neurofeedback metric for stress reduction
                                 Higher theta over alpha is supposedly associated with reduced anxiety
        """

        alpha_metric = self.smooth_band_powers[Band.Alpha] / self.smooth_band_powers[Band.Delta]
        beta_metric = self.smooth_band_powers[Band.Beta] / self.smooth_band_powers[Band.Theta]
        theta_metric = self.smooth_band_powers[Band.Theta] / self.smooth_band_powers[Band.Alpha]
        if print_flag:
            print(f"Alpha/Delta ratio: {alpha_metric:.5f},    "
                  f"Beta/Theta ratio: {beta_metric:.5f},   "
                  f"Theta/Alpha ratio: {theta_metric:.5f}")
        return alpha_metric, beta_metric, theta_metric

    # TODO: Set Routine
    def run_once(self):
        self.get_data_chunk(print_flag=self.print_data)
        # self.comp_band_power(self.print_band_power)
        # self.comp_nf_metrics(self.print_metrics)

    def run_for(self, n):
        for _ in range(n):
            self.run_once()

    def run(self):
        self.muse.start()
        print('-Press Ctrl-C in the console to break the while loop-')
        while not self.stop and self.has_stream:
            try:
                self.run_once()
            except Exception as e:
                return self.exception_handler(e)

    # TODO: write an idle menu
    def menu(self):
        pass

    # TODO: Fix to make it correlate to the new get_data_chunk method
    def make_recording(self, data, timestamps, filename=None):

        # time_correction = self.inlet.time_correction()
        # print('Time correction: ', time_correction)
        # timestamps = np.array(timestamps) + time_correction

        data = np.concatenate(data, axis=0)
        data = np.c_[timestamps, data]
        data = pd.DataFrame(data=data, columns=['timestamps'] + self.ch_names)

        """
        if inlet_marker:
            n_markers = len(markers[0][0])
            for ii in range(n_markers):
                data['Marker%d' % ii] = 0
            # process markers:
            for marker in markers:
                # find index of markers
                ix = np.argmin(np.abs(marker[1] - timestamps))
                for ii in range(n_markers):
                    data.loc[ix, 'Marker%d' % ii] = marker[0][ii]
        """

        if filename is None:  filename = get_full_file()
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):  os.makedirs(directory)

        data.to_csv(filename, float_format='%.3f', index=False)
        print(f'Done - wrote file: {filename}.')

    # TODO: Fix to make it correlate to the new get_data_chunk method
    def record_for(self, duration, filename=None, look_for_markers=False, marker_stream_timeout=2):
        """
        inlet_marker = False
        if look_for_markers: inlet_marker = marker_lookup(marker_stream_timeout)
        markers = []
        """
        all_data = []
        all_timestamps = []
        t_init = time.time()
        print(f'Start recording at time t={t_init:.3f}')

        time_correction = self.inlet.time_correction()
        print('Time correction: ', time_correction)

        while time.time() - t_init < duration:
            try:
                data, timestamp = self.inlet.pull_chunk(timeout=1.0, max_samples=EEG_CHUNK)
                if timestamp:
                    print(f"writing at time: {dt.datetime.fromtimestamp(timestamp[0]).strftime('%H.%M.%S')}")
                    all_data.append(data)
                    all_timestamps.extend(timestamp)
                """    
                if inlet_marker:
                    marker, timestamp = inlet_marker.pull_sample(timeout=0.0)
                    if timestamp:
                        markers.append([marker, timestamp])
                """
            # TODO: handle exceptions
            except KeyboardInterrupt:
                break

        self.make_recording(all_data, all_timestamps, filename)

    # TODO: Fix to make it correlate to the new get_data_chunk method
    def record_cont(self, min_samples_per_file=1000, marker_stream_timeout=2):
        """
        inlet_marker = False
        if look_for_markers: inlet_marker = marker_lookup(marker_stream_timeout)
        markers = []
        """
        time_correction = self.inlet.time_correction()

        print(f'Start recording at time t={time.time():.3f}')
        print('Time correction: ', time_correction)

        while not self.stop and self.has_stream:
            # markers = []
            all_data = []
            all_timestamps = []
            cnt = 0
            try:
                while len(all_data) * EEG_CHUNK < min_samples_per_file:
                    try:
                        data, timestamp = self.inlet.pull_chunk(timeout=1.0, max_samples=EEG_CHUNK)
                        if timestamp:
                            cnt += 1
                            print(f"writing batch number {cnt}/{int(min_samples_per_file / EEG_CHUNK)} "
                                  f"at time: {dt.datetime.fromtimestamp(timestamp[0]).strftime('%H.%M.%S')}")
                            all_data.append(data)
                            all_timestamps.extend(timestamp)
                        '''
                        if inlet_marker:
                            marker, timestamp = inlet_marker.pull_sample(timeout=0.0)
                            if timestamp:
                                markers.append([marker, timestamp])
                        '''
                    except KeyboardInterrupt:
                        break
                self.make_recording(all_data, all_timestamps)
            except Exception as e:
                return self.exception_handler(e)

    def exception_handler(self, e):
        if isinstance(e, IndexError):
            print("[Connection Lost] (raised index error)")
            print(type(e), e)
            self.has_stream = False
            self.inlet = self.stream_listener()
            print("[Connection Re-established]")
            self.run()
        elif isinstance(e, KeyboardInterrupt):
            print(type(e), e)
            self.stop = True
        elif isinstance(e, PermissionError):
            print(type(e), e)
            self.stop = True
        else:
            print(type(e), e)
            return


if __name__ == "__main__":
    ms = MuseStreamer(ADDRESS, print_data=True, print_band_power=False, print_metrics=False)
    # ms.get_data_chunk()
    # ms.run_for(20)
    # ms.run()
    # ms.muse.stop()
    # ms.record_cont(min_samples_per_file=10000)
    # ms.record_for(20)

"""
import matplotlib.pyplot as plt
plt.plot(ms.data_frame['Timestamp'][-500:], ms.data_frame[ms.ch_names[1]][-500:])
plt.show()


plt.cla()
"""