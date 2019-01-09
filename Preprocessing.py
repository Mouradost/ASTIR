import os
import pickle
from datetime import datetime
import numpy as np
import h5py
import time
import pandas as pd
from copy import copy
np.random.seed(1337)
# parameters
DATAPATH = os.path.join(os.getcwd(), 'DB')

class STMatrix(object):
    """docstring for STMatrix"""

    def __init__(self, data, timestamps, T=48, CheckComplete=True):
        super(STMatrix, self).__init__()
        assert len(data) == len(timestamps)
        self.data = data
        self.timestamps = timestamps
        self.T = T
        self.pd_timestamps = string2timestamp(timestamps, T=self.T)
        if CheckComplete:
            self.check_complete()
        # index
        self.make_index()

    def make_index(self):
        self.get_index = dict()
        for i, ts in enumerate(self.pd_timestamps):
            self.get_index[ts] = i

    def check_complete(self):
        missing_timestamps = []
        offset = pd.DateOffset(minutes=24 * 60 // self.T)
        pd_timestamps = self.pd_timestamps
        i = 1
        while i < len(pd_timestamps):
            if pd_timestamps[i-1] + offset != pd_timestamps[i]:
                missing_timestamps.append("(%s -- %s)" % (pd_timestamps[i-1], pd_timestamps[i]))
            i += 1
        for v in missing_timestamps:
            print(v)
        assert len(missing_timestamps) == 0

    def get_matrix(self, timestamp):
        return self.data[self.get_index[timestamp]]

    def save(self, fname):
        pass

    def check_it(self, depends):
        for d in depends:
            if d not in self.get_index.keys():
                return False
        return True

    def create_dataset(self, len_closeness=3, len_trend=3, TrendInterval=7, len_period=3, PeriodInterval=1):
        """current version
        """
        # offset_week = pd.DateOffset(days=7)
        offset_frame = pd.DateOffset(minutes=24 * 60 // self.T)
        XC = []
        XP = []
        XT = []
        Y = []
        timestamps_Y = []
        depends = [range(1, len_closeness+1),
                   [PeriodInterval * self.T * j for j in range(1, len_period+1)],
                   [TrendInterval * self.T * j for j in range(1, len_trend+1)]]
        print('*'*100)
        print(depends)
        i = max(self.T * TrendInterval * len_trend, self.T * PeriodInterval * len_period, len_closeness)
        print('i: ', i)

        while i < len(self.pd_timestamps):
            Flag = True
            for depend in depends:
                if Flag is False:
                    break
                Flag = self.check_it([self.pd_timestamps[i] - j * offset_frame for j in depend])

            if Flag is False:
                i += 1
                continue
            x_c = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame) for j in depends[0]]
            x_p = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame) for j in depends[1]]
            x_t = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame) for j in depends[2]]
            y = self.get_matrix(self.pd_timestamps[i])
            if len_closeness > 0:
                XC.append(np.vstack(x_c))
                print('XC shape [[[[ ', np.shape(XC))
            if len_period > 0:
                XP.append(np.vstack(x_p))
            if len_trend > 0:
                XT.append(np.vstack(x_t))
            Y.append(y)
            timestamps_Y.append(self.timestamps[i])
            i += 1
        XC = np.asarray(XC)
        XP = np.asarray(XP)
        XT = np.asarray(XT)
        Y = np.asarray(Y)
        print("XC shape: ", XC.shape, "XP shape: ", XP.shape, "XT shape: ", XT.shape, "Y shape:", Y.shape)
        return XC, XP, XT, Y, timestamps_Y


class MinMaxNormalization(object):
    '''MinMax Normalization --> [-1, 1]
       x = (x - min) / (max - min).
       x = x * 2 - 1
    '''

    def __init__(self):
        pass

    def fit(self, X):
        self._min = X.min()
        self._max = X.max()
        print("min:", self._min, "max:", self._max)

    def transform(self, X):
        X = 1. * (X - self._min) / (self._max - self._min)
        X = X * 2. - 1.
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        X = (X + 1.) / 2.
        X = 1. * X * (self._max - self._min) + self._min
        return X


class MinMaxNormalization_01(object):
    '''MinMax Normalization --> [0, 1]
       x = (x - min) / (max - min).
    '''

    def __init__(self):
        pass

    def fit(self, X):
        self._min = X.min()
        self._max = X.max()
        print("min:", self._min, "max:", self._max)

    def transform(self, X):
        X = 1. * (X - self._min) / (self._max - self._min)
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        X = 1. * X * (self._max - self._min) + self._min
        return X

def timestamp_str_new(cur_timestampes, T=48):
    os.environ['TZ'] = 'Asia/Shanghai'
    # print cur_timestampes
    if '-' in cur_timestampes[0]:
        return cur_timestampes
    ret = []
    for v in cur_timestampes:
        '''TODO
        Bug here
        '''
        cur_sec = time.mktime(time.strptime("%04i-%02i-%02i" % (int(v[:4]), int(v[4:6]), int(v[6:8])), "%Y-%m-%d")) + (int(v[8:]) * 24. * 60 * 60 // T)
        curr = time.localtime(cur_sec)
        if v == "20151101288" or v == "2015110124":
            print(v, time.strftime("%Y-%m-%d-%H-%M", curr), time.localtime(cur_sec), time.localtime(cur_sec - (int(v[8:]) * 24. * 60 * 60 // T)), time.localtime(cur_sec - (int(v[8:]) * 24. * 60 * 60 // T) + 3600 * 25))
        ret.append(time.strftime("%Y-%m-%d-%H-%M", curr))
    return ret


def string2timestamp_future(strings, T=48):
    strings = timestamp_str_new(strings, T)
    timestamps = []
    for v in strings:
        year, month, day, hour, tm_min = [int(z) for z in v.split('-')]
        timestamps.append(pd.Timestamp(datetime(year, month, day, hour, tm_min)))

    return timestamps


def string2timestamp(strings, T=48):
    timestamps = []

    time_per_slot = 24.0 / T
    num_per_T = T // 24
    for t in strings:
        year, month, day, slot = int(t[:4]), int(t[4:6]), int(t[6:8]), int(t[8:])-1
        timestamps.append(pd.Timestamp(datetime(year, month, day, hour=int(slot * time_per_slot), minute=(slot % num_per_T) * int(60.0 * time_per_slot))))

    return timestamps


def timestamp2string(timestamps, T=48):
    # timestamps = timestamp_str_new(timestamps)
    num_per_T = T // 24
    return ["%s%02i" % (ts.strftime('%Y%m%d'),
                        int(1+ts.to_datetime().hour*num_per_T+ts.to_datetime().minute/(60 // num_per_T))) for ts in timestamps]

def timestamp2vec(timestamps):
    # tm_wday range [0, 6], Monday is 0
    # vec = [time.strptime(t[:8], '%Y%m%d').tm_wday for t in timestamps]  # python2
    vec = [time.strptime(str(t[:8], encoding='utf-8'), '%Y%m%d').tm_wday for t in timestamps]  # python3
    ret = []
    for i in vec:
        v = [0 for _ in range(7)]
        v[i] = 1
        if i >= 5:
            v.append(0)  # weekend
        else:
            v.append(1)  # weekday
        ret.append(v)
    return np.asarray(ret)


def remove_incomplete_days(data, timestamps, T=48):
    # remove a certain day which has not 48 timestamps
    days = []  # available days: some day only contain some seqs
    days_incomplete = []
    i = 0
    while i < len(timestamps):
        if int(timestamps[i][8:]) != 1:
            i += 1
        elif i+T-1 < len(timestamps) and int(timestamps[i+T-1][8:]) == T:
            days.append(timestamps[i][:8])
            i += T
        else:
            days_incomplete.append(timestamps[i][:8])
            i += 1
    print("incomplete days: ", days_incomplete)
    days = set(days)
    idx = []
    for i, t in enumerate(timestamps):
        if t[:8] in days:
            idx.append(i)

    data = data[idx]
    timestamps = [timestamps[i] for i in idx]
    return data, timestamps


def split_by_time(data, timestamps, split_timestamp):
    # divide data into two subsets:
    # e.g., Train: ~ 2015.06.21 & Test: 2015.06.22 ~ 2015.06.28
    assert(len(data) == len(timestamps))
    assert(split_timestamp in set(timestamps))

    data_1 = []
    timestamps_1 = []
    data_2 = []
    timestamps_2 = []
    switch = False
    for t, d in zip(timestamps, data):
        if split_timestamp == t:
            switch = True
        if switch is False:
            data_1.append(d)
            timestamps_1.append(t)
        else:
            data_2.append(d)
            timestamps_2.append(t)
    return (np.asarray(data_1), timestamps_1), (np.asarray(data_2), timestamps_2)


def timeseries2seqs(data, timestamps, length=3, T=48):
    raw_ts = copy(timestamps)
    if type(timestamps[0]) != pd.Timestamp:
        timestamps = string2timestamp(timestamps, T=T)

    offset = pd.DateOffset(minutes=24 * 60 // T)

    breakpoints = [0]
    for i in range(1, len(timestamps)):
        if timestamps[i-1] + offset != timestamps[i]:
            print(timestamps[i-1], timestamps[i], raw_ts[i-1], raw_ts[i])
            breakpoints.append(i)
    breakpoints.append(len(timestamps))
    X = []
    Y = []
    for b in range(1, len(breakpoints)):
        print('breakpoints: ', breakpoints[b-1], breakpoints[b])
        idx = range(breakpoints[b-1], breakpoints[b])
        for i in range(len(idx) - length):
            x = np.vstack(data[idx[i:i+length]])
            y = data[idx[i+length]]
            X.append(x)
            Y.append(y)
    X = np.asarray(X)
    Y = np.asarray(Y)
    print("X shape: ", X.shape, "Y shape:", Y.shape)
    return X, Y

def timeseries2seqs_meta(data, timestamps, length=3, T=48):
    raw_ts = copy(timestamps)
    if type(timestamps[0]) != pd.Timestamp:
        timestamps = string2timestamp(timestamps, T=T)

    offset = pd.DateOffset(minutes=24 * 60 // T)

    breakpoints = [0]
    for i in range(1, len(timestamps)):
        if timestamps[i-1] + offset != timestamps[i]:
            print(timestamps[i-1], timestamps[i], raw_ts[i-1], raw_ts[i])
            breakpoints.append(i)
    breakpoints.append(len(timestamps))
    X = []
    Y = []
    avail_timestamps = []
    for b in range(1, len(breakpoints)):
        print('breakpoints: ', breakpoints[b-1], breakpoints[b])
        idx = range(breakpoints[b-1], breakpoints[b])
        for i in range(len(idx) - length):
            avail_timestamps.append(raw_ts[idx[i+length]])
            x = np.vstack(data[idx[i:i+length]])
            y = data[idx[i+length]]
            X.append(x)
            Y.append(y)
    X = np.asarray(X)
    Y = np.asarray(Y)
    print("X shape: ", X.shape, "Y shape:", Y.shape)
    return X, Y, avail_timestamps


def timeseries2seqs_peroid_trend(data, timestamps, length=3, T=48, peroid=pd.DateOffset(days=7), peroid_len=2):
    raw_ts = copy(timestamps)
    if type(timestamps[0]) != pd.Timestamp:
        timestamps = string2timestamp(timestamps, T=T)

    # timestamps index
    timestamp_idx = dict()
    for i, t in enumerate(timestamps):
        timestamp_idx[t] = i

    offset = pd.DateOffset(minutes=24 * 60 // T)

    breakpoints = [0]
    for i in range(1, len(timestamps)):
        if timestamps[i-1] + offset != timestamps[i]:
            print(timestamps[i-1], timestamps[i], raw_ts[i-1], raw_ts[i])
            breakpoints.append(i)
    breakpoints.append(len(timestamps))
    X = []
    Y = []
    for b in range(1, len(breakpoints)):
        print('breakpoints: ', breakpoints[b-1], breakpoints[b])
        idx = range(breakpoints[b-1], breakpoints[b])
        for i in range(len(idx) - length):
            # period
            target_timestamp = timestamps[i+length]

            legal_idx = []
            for pi in range(1, 1+peroid_len):
                if target_timestamp - peroid * pi not in timestamp_idx:
                    break
                legal_idx.append(timestamp_idx[target_timestamp - peroid * pi])
            # print("len: ", len(legal_idx), peroid_len)
            if len(legal_idx) != peroid_len:
                continue

            legal_idx += idx[i:i+length]

            # trend
            x = np.vstack(data[legal_idx])
            y = data[idx[i+length]]
            X.append(x)
            Y.append(y)
    X = np.asarray(X)
    Y = np.asarray(Y)
    print("X shape: ", X.shape, "Y shape:", Y.shape)
    return X, Y


def timeseries2seqs_3D(data, timestamps, length=3, T=48):
    raw_ts = copy(timestamps)
    if type(timestamps[0]) != pd.Timestamp:
        timestamps = string2timestamp(timestamps, T=T)

    offset = pd.DateOffset(minutes=24 * 60 // T)

    breakpoints = [0]
    for i in range(1, len(timestamps)):
        if timestamps[i-1] + offset != timestamps[i]:
            print(timestamps[i-1], timestamps[i], raw_ts[i-1], raw_ts[i])
            breakpoints.append(i)
    breakpoints.append(len(timestamps))
    X = []
    Y = []
    for b in range(1, len(breakpoints)):
        print('breakpoints: ', breakpoints[b-1], breakpoints[b])
        idx = range(breakpoints[b-1], breakpoints[b])
        for i in range(len(idx) - length):
            x = data[idx[i:i+length]].reshape(-1, length, 32, 32)
            y = np.asarray([data[idx[i+length]]]).reshape(-1, 1, 32, 32)
            X.append(x)
            Y.append(y)
    X = np.asarray(X)
    Y = np.asarray(Y)
    print("X shape: ", X.shape, "Y shape:", Y.shape)
    return X, Y


def bug_timeseries2seqs(data, timestamps, length=3, T=48):
    # have a bug
    if type(timestamps[0]) != pd.Timestamp:
        timestamps = string2timestamp(timestamps, T=T)

    offset = pd.DateOffset(minutes=24 * 60 // T)

    breakpoints = [0]
    for i in range(1, len(timestamps)):
        if timestamps[i-1] + offset != timestamps[i]:
            breakpoints.append(i)
    X = []
    Y = []
    for b in range(1, len(breakpoints)):
        print('breakpoints: ', breakpoints[b-1], breakpoints[b])
        idx = range(breakpoints[b-1], breakpoints[b])
        for i in range(len(idx) - 3):
            x = np.vstack(data[idx[i:i+3]])
            y = data[idx[i+3]]
            X.append(x)
            Y.append(y)
    X = np.asarray(X)
    Y = np.asarray(Y)
    print("X shape: ", X.shape, "Y shape:", Y.shape)
    return X, Y



def load_stdata(fname):
    f = h5py.File(fname, 'r')
    data = f['data'].value
    timestamps = f['date'].value
    f.close()
    return data, timestamps


def stat(fname):
    def get_nb_timeslot(f):
        s = f['date'][0]
        e = f['date'][-1]
        year, month, day = map(int, [s[:4], s[4:6], s[6:8]])
        ts = time.strptime("%04i-%02i-%02i" % (year, month, day), "%Y-%m-%d")
        year, month, day = map(int, [e[:4], e[4:6], e[6:8]])
        te = time.strptime("%04i-%02i-%02i" % (year, month, day), "%Y-%m-%d")
        nb_timeslot = (time.mktime(te) - time.mktime(ts)) / (0.5 * 3600) + 48
        ts_str, te_str = time.strftime("%Y-%m-%d", ts), time.strftime("%Y-%m-%d", te)
        return nb_timeslot, ts_str, te_str

    with h5py.File(fname) as f:
        nb_timeslot, ts_str, te_str = get_nb_timeslot(f)
        nb_day = int(nb_timeslot / 48)
        mmax = f['data'].value.max()
        mmin = f['data'].value.min()
        stat = '=' * 5 + 'stat' + '=' * 5 + '\n' + \
               'data shape: %s\n' % str(f['data'].shape) + \
               '# of days: %i, from %s to %s\n' % (nb_day, ts_str, te_str) + \
               '# of timeslots: %i\n' % int(nb_timeslot) + \
               '# of timeslots (available): %i\n' % f['date'].shape[0] + \
               'missing ratio of timeslots: %.1f%%\n' % ((1. - float(f['date'].shape[0] / nb_timeslot)) * 100) + \
               'max: %.3f, min: %.3f\n' % (mmax, mmin) + \
               '=' * 5 + 'stat' + '=' * 5
        print(stat)


def load_holiday(timeslots, fname=os.path.join(DATAPATH, 'TaxiBJ', 'BJ_Holiday.txt')):
    f = open(fname, 'r')
    holidays = f.readlines()
    holidays = set([h.strip() for h in holidays])
    H = np.zeros(len(timeslots))
    for i, slot in enumerate(timeslots):
        if slot[:8] in holidays:
            H[i] = 1
    print(H.sum())
    return H[:, None]


def load_meteorol(timeslots, fname=os.path.join(DATAPATH, 'TaxiBJ', 'BJ_Meteorology.h5')):
    '''
    timeslots: the predicted timeslots
    In real-world, we dont have the meteorol data in the predicted timeslot, instead, we use the meteoral at previous timeslots, i.e., slot = predicted_slot - timeslot (you can use predicted meteorol data as well)
    '''
    f = h5py.File(fname, 'r')
    Timeslot = f['date'].value
    WindSpeed = f['WindSpeed'].value
    Weather = f['Weather'].value
    Temperature = f['Temperature'].value
    f.close()

    M = dict()  # map timeslot to index
    for i, slot in enumerate(Timeslot):
        M[slot] = i

    WS = []  # WindSpeed
    WR = []  # Weather
    TE = []  # Temperature
    for slot in timeslots:
        predicted_id = M[slot]
        cur_id = predicted_id - 1
        WS.append(WindSpeed[cur_id])
        WR.append(Weather[cur_id])
        TE.append(Temperature[cur_id])

    WS = np.asarray(WS)
    WR = np.asarray(WR)
    TE = np.asarray(TE)

    # 0-1 scale
    WS = 1. * (WS - WS.min()) / (WS.max() - WS.min())
    TE = 1. * (TE - TE.min()) / (TE.max() - TE.min())

    print("shape: ", WS.shape, WR.shape, TE.shape)

    # concatenate all these attributes
    merge_data = np.hstack([WR, WS[:, None], TE[:, None]])

    # print('meger shape:', merge_data.shape)
    return merge_data


def load_data_NYC(T=24, nb_flow=2, len_closeness=None, len_period=None, len_trend=None, len_test=None, meta_data=False):
    assert (len_closeness + len_period + len_trend > 0)
    # load data
    stat(os.path.join(DATAPATH, 'BikeNYC', 'NYC14_M16x8_T60_NewEnd.h5'))
    data, timestamps = load_stdata(os.path.join(DATAPATH, 'BikeNYC', 'NYC14_M16x8_T60_NewEnd.h5'))
    # print(timestamps)
    # remove a certain day which does not have 48 timestamps
    data, timestamps = remove_incomplete_days(data, timestamps, T)
    data = data[:, :nb_flow]
    data[data < 0] = 0.
    data_all = [data]
    timestamps_all = [timestamps]
    # minmax_scale
    data_train = data[:-len_test]
    print('train_data shape: ', data_train.shape)
    mmn = MinMaxNormalization()
    mmn.fit(data_train)
    data_all_mmn = []
    for d in data_all:
        data_all_mmn.append(mmn.transform(d))
    XC, XP, XT = [], [], []
    Y = []
    timestamps_Y = []
    for data, timestamps in zip(data_all_mmn, timestamps_all):
        # instance-based dataset --> sequences with format as (X, Y) where X is a sequence of images and Y is an image.
        st = STMatrix(data, timestamps, T, CheckComplete=False)
        _XC, _XP, _XT, _Y, _timestamps_Y = st.create_dataset(len_closeness=len_closeness, len_period=len_period,
                                                             len_trend=len_trend)
        XC.append(_XC)
        XP.append(_XP)
        XT.append(_XT)
        Y.append(_Y)
        timestamps_Y += _timestamps_Y

    XC = np.vstack(XC)
    XP = np.vstack(XP)
    XT = np.vstack(XT)
    Y = np.vstack(Y)
    print("XC shape: ", XC.shape, "XP shape: ", XP.shape, "XT shape: ", XT.shape, "Y shape:", Y.shape)
    XC_train, XP_train, XT_train, Y_train = XC[:-len_test], XP[:-len_test], XT[:-len_test], Y[:-len_test]
    XC_test, XP_test, XT_test, Y_test = XC[-len_test:], XP[-len_test:], XT[-len_test:], Y[-len_test:]

    timestamp_train, timestamp_test = timestamps_Y[:-len_test], timestamps_Y[-len_test:]
    X_train = []
    X_test = []
    for l, X_ in zip([len_closeness, len_period, len_trend], [XC_train, XP_train, XT_train]):
        if l > 0:
            X_train.append(X_)
    for l, X_ in zip([len_closeness, len_period, len_trend], [XC_test, XP_test, XT_test]):
        if l > 0:
            X_test.append(X_)
    print('train shape:', XC_train.shape, Y_train.shape, 'test shape: ', XC_test.shape, Y_test.shape)
    # load meta feature
    if meta_data:
        meta_feature = timestamp2vec(timestamps_Y)
        metadata_dim = meta_feature.shape[1]
        meta_feature_train, meta_feature_test = meta_feature[:-len_test], meta_feature[-len_test:]
        X_train.append(meta_feature_train)
        X_test.append(meta_feature_test)
    else:
        metadata_dim = None
    for _X in X_train:
        print(_X.shape, )
    print()
    for _X in X_test:
        print(_X.shape, )
    print()
    return X_train, Y_train, X_test, Y_test, mmn, metadata_dim, timestamp_train, timestamp_test


def load_data(T=48, nb_flow=2, len_closeness=None, len_period=None, len_trend=None,
              len_test=None,
              meta_data=True, meteorol_data=True, holiday_data=True, year_start=13, year_finish=17):
    assert(len_closeness + len_period + len_trend > 0)
    data_all = []
    timestamps_all = list()
    for year in range(year_start, year_finish):
        fname = os.path.join(
            DATAPATH, 'TaxiBJ', 'BJ{}_M32x32_T30_InOut.h5'.format(year))
        print("file name: ", fname)
        stat(fname)
        data, timestamps = load_stdata(fname)
        # print(timestamps)
        # remove a certain day which does not have 48 timestamps
        data, timestamps = remove_incomplete_days(data, timestamps, T)
        data = data[:, :nb_flow]
        data[data < 0] = 0.
        data_all.append(data)
        timestamps_all.append(timestamps)
        print("\n")

    # minmax_scale
    data_train = np.vstack(copy(data_all))[:-len_test]
    print('train_data shape: ', data_train.shape)
    mmn = MinMaxNormalization()
    mmn.fit(data_train)
    data_all_mmn = [mmn.transform(d) for d in data_all]
    XC, XP, XT = [], [], []
    Y = []
    timestamps_Y = []
    for data, timestamps in zip(data_all_mmn, timestamps_all):
        print('-'*100)
        print(np.shape(data))
        print(np.shape(timestamps))
        print('-' * 100)
        # instance-based dataset --> sequences with format as (X, Y) where X is
        # a sequence of images and Y is an image.
        st = STMatrix(data, timestamps, T, CheckComplete=False)
        _XC, _XP, _XT, _Y, _timestamps_Y = st.create_dataset(
            len_closeness=len_closeness, len_period=len_period, len_trend=len_trend)
        print('_XC{}, _XP{}, _XT{}, _Y{}, _timestamps_Y{}'.format(np.shape(_XC), np.shape(_XP), np.shape(_XT), np.shape(_Y), np.shape(_timestamps_Y)))
        XC.append(_XC)
        XP.append(_XP)
        XT.append(_XT)
        Y.append(_Y)
        timestamps_Y += _timestamps_Y

    meta_feature = []
    if meta_data:
        # load time feature
        time_feature = timestamp2vec(timestamps_Y)
        meta_feature.append(time_feature)
    if holiday_data:
        # load holiday
        holiday_feature = load_holiday(timestamps_Y)
        meta_feature.append(holiday_feature)
    if meteorol_data:
        # load meteorol data
        meteorol_feature = load_meteorol(timestamps_Y)
        meta_feature.append(meteorol_feature)

    meta_feature = np.hstack(meta_feature) if len(
        meta_feature) > 0 else np.asarray(meta_feature)
    metadata_dim = meta_feature.shape[1] if len(
        meta_feature.shape) > 1 else None
    if metadata_dim < 1:
        metadata_dim = None
    if meta_data and holiday_data and meteorol_data:
        print('time feature:', time_feature.shape, 'holiday feature:', holiday_feature.shape,
              'meteorol feature: ', meteorol_feature.shape, 'mete feature: ', meta_feature.shape)

    XC = np.vstack(XC)
    XP = np.vstack(XP)
    XT = np.vstack(XT)
    Y = np.vstack(Y)
    print("XC shape: ", XC.shape, "XP shape: ", XP.shape,
          "XT shape: ", XT.shape, "Y shape:", Y.shape)

    XC_train, XP_train, XT_train, Y_train = XC[
        :-len_test], XP[:-len_test], XT[:-len_test], Y[:-len_test]
    XC_test, XP_test, XT_test, Y_test = XC[
        -len_test:], XP[-len_test:], XT[-len_test:], Y[-len_test:]
    timestamp_train, timestamp_test = timestamps_Y[
        :-len_test], timestamps_Y[-len_test:]

    X_train = []
    X_test = []
    for l, X_ in zip([len_closeness, len_period, len_trend], [XC_train, XP_train, XT_train]):
        if l > 0:
            X_train.append(X_)
    for l, X_ in zip([len_closeness, len_period, len_trend], [XC_test, XP_test, XT_test]):
        if l > 0:
            X_test.append(X_)
    print('train shape:', XC_train.shape, Y_train.shape,
          'test shape: ', XC_test.shape, Y_test.shape)

    if metadata_dim is not None:
        meta_feature_train, meta_feature_test = meta_feature[
            :-len_test], meta_feature[-len_test:]
        X_train.append(meta_feature_train)
        X_test.append(meta_feature_test)
    for _X in X_train:
        print(_X.shape, )
    print()
    for _X in X_test:
        print(_X.shape, )
    print()
    return X_train, Y_train, X_test, Y_test, mmn, metadata_dim, timestamp_train, timestamp_test

def save_preprocess_data(filePath, X_train, Y_train, X_test, Y_test, mmn, metadata_dim, timestamp_train, timestamp_test, DB_name='TaxiBJ'):
    os.makedirs(filePath, exist_ok=True)
    np.save(os.path.join(filePath, 'XC'), X_train[0])
    np.save(os.path.join(filePath, 'XP'), X_train[1])
    np.save(os.path.join(filePath, 'XT'), X_train[2])
    np.save(os.path.join(filePath, 'Y_train'), Y_train)
    np.save(os.path.join(filePath, 'XC_Test'), X_test[0])
    np.save(os.path.join(filePath, 'XP_Test'), X_test[1])
    np.save(os.path.join(filePath, 'XT_Test'), X_test[2])
    np.save(os.path.join(filePath, 'Y_test'), Y_test)
    with open(os.path.join(filePath, 'mmn'), 'wb') as fpkl:
        for obj in [mmn]:
            pickle.dump(obj, fpkl)
    np.save(os.path.join(filePath, 'metadata_dim'), metadata_dim)
    np.save(os.path.join(filePath, 'timestamp_train'), timestamp_train)
    np.save(os.path.join(filePath, 'timestamp_test'), timestamp_test)
    if DB_name == 'TaxiBJ':
        np.save(os.path.join(filePath, 'External'), X_train[3])
        np.save(os.path.join(filePath, 'External_Test'), X_test[3])

def load_preprocess_data(filePath, DB_name='TaxiBJ'):
    X_train, X_test = [], []
    X_train.append(np.load(os.path.join(filePath, 'XC.npy')))
    X_train.append(np.load(os.path.join(filePath, 'XP.npy')))
    X_train.append(np.load(os.path.join(filePath, 'XT.npy')))
    Y_train = np.load(os.path.join(filePath, 'Y_train.npy'))
    X_test.append(np.load(os.path.join(filePath, 'XC_Test.npy')))
    X_test.append(np.load(os.path.join(filePath, 'XP_Test.npy')))
    X_test.append(np.load(os.path.join(filePath, 'XT_Test.npy')))
    Y_test = np.load(os.path.join(filePath, 'Y_test.npy'))
    with open(os.path.join(filePath, 'mmn'), 'rb') as fpkl:
        mmn = pickle.load(fpkl)
    metadata_dim = np.load(os.path.join(filePath, 'metadata_dim.npy'))
    timestamp_train = np.load(os.path.join(filePath, 'timestamp_train.npy'))
    timestamp_test = np.load(os.path.join(filePath, 'timestamp_test.npy'))
    if DB_name == 'TaxiBJ':
        X_test.append(np.load(os.path.join(filePath, 'External_Test.npy')))
        X_train.append(np.load(os.path.join(filePath, 'External.npy')))

    return X_train, Y_train, X_test, Y_test, mmn, metadata_dim, timestamp_train, timestamp_test

def prepare_data_as_a_sequence(X_train, X_test, len_closeness=3, len_period=3, len_trend=3, channel=2):
    X_Train, X_Test = X_train, X_test
    if len_closeness > 0 and len_period > 0 and len_trend > 0:
        id_closeness, id_period, id_trend = 0, 1, 2
    if len_closeness == 0 and len_period > 0 and len_trend > 0:
        id_closeness, id_period, id_trend = 0, 0, 1
    if len_closeness > 0 and len_period == 0 and len_trend > 0:
        id_closeness, id_period, id_trend = 0, 1, 1
    if len_closeness > 0 and len_period > 0 and len_trend == 0:
        id_closeness, id_period, id_trend = 0, 1, 2
    if len_closeness > 0:
        X_Train[id_closeness] = np.reshape(X_Train[id_closeness],
                                (np.shape(X_Train[id_closeness])[0], len_closeness, channel, np.shape(X_Train[id_closeness])[2], np.shape(X_Train[id_closeness])[3]))
    if len_period > 0:
        X_Train[id_period] = np.reshape(X_Train[id_period],
                                (np.shape(X_Train[id_period])[0], len_period, channel, np.shape(X_Train[id_period])[2], np.shape(X_Train[id_period])[3]))
    if len_trend > 0:
        X_Train[id_trend] = np.reshape(X_Train[id_trend],
                                (np.shape(X_Train[id_trend])[0], len_trend, channel, np.shape(X_Train[id_trend])[2], np.shape(X_Train[id_trend])[3]))
    if len_closeness > 0:
        X_Test[id_closeness] = np.reshape(X_Test[id_closeness],
                               (np.shape(X_Test[id_closeness])[0], len_closeness, channel, np.shape(X_Test[id_closeness])[2], np.shape(X_Test[id_closeness])[3]))
    if len_period > 0:
        X_Test[id_period] = np.reshape(X_Test[id_period],
                               (np.shape(X_Test[id_period])[0], len_period, channel, np.shape(X_Test[id_period])[2], np.shape(X_Test[id_period])[3]))
    if len_trend > 0:
        X_Test[id_trend] = np.reshape(X_Test[id_trend],
                               (np.shape(X_Test[id_trend])[0], len_trend, channel, np.shape(X_Test[id_trend])[2], np.shape(X_Test[id_trend])[3]))
    return X_Train, X_Test

def get_DB(DB_name, len_closeness, len_period, len_trend):
    preprocessed_DB_path = os.path.join(os.getcwd(), 'DB/databases_Cleaned/paper_preprocess/{}'.format(DB_name),
                                        'len_closeness_{}_len_period_{}_len_trend_{}'.format(len_closeness, len_period,
                                                                                             len_trend))
    if os.path.isdir(preprocessed_DB_path):
        X_train, Y_train, X_test, Y_test, mmn, metadata_dim, timestamp_train, timestamp_test = load_preprocess_data(
            preprocessed_DB_path,
            DB_name=DB_name)
    else:
        os.makedirs(preprocessed_DB_path, exist_ok=True)
        if DB_name == 'TaxiBJ':
            X_train, Y_train, X_test, Y_test, mmn, metadata_dim, timestamp_train, timestamp_test = load_data(
                T=48, nb_flow=2, len_closeness=len_closeness, len_period=len_period, len_trend=len_trend,
                len_test=48 * (7 * 4))
        else:
            X_train, Y_train, X_test, Y_test, mmn, metadata_dim, timestamp_train, timestamp_test = load_data_NYC(
                T=24, nb_flow=2, len_closeness=len_closeness, len_period=len_period, len_trend=len_trend, len_test=240)
        #save_preprocess_data(filePath=preprocessed_DB_path, X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test, mmn=mmn, metadata_dim=metadata_dim, timestamp_train=timestamp_train, timestamp_test=timestamp_test, DB_name=DB_name)

    return X_train, Y_train, X_test, Y_test, mmn, metadata_dim, timestamp_train, timestamp_test


if __name__ == '__main__':


    DB_names = ['BikeNYC']
    lens_closeness = 1
    lens_period = 1
    lens_trend = 2
    channel = 2

    for len_closeness in range(0, lens_closeness):
        for len_period in range(0, lens_period):
            for len_trend in range(1, lens_trend):
                for DB_name in DB_names:
                    print('*' * 50)
                    print('Data Base : {}\nlen_closeness : {}\nlen_period : {}\nlen_trend : {}'.format(DB_name, len_closeness, len_period, len_trend))
                    print('*' * 50)
                    try:
                        X_train, Y_train, X_test, Y_test, mmn, metadata_dim, timestamp_train, timestamp_test = get_DB(DB_name, len_closeness, len_period, len_trend)
                    except Exception as e:
                        print('+'*50)
                        print('Error {} {} {}'.format(len_closeness, len_period, len_trend))
                        print(e)
                        print('+' * 50)
                        preprocessed_DB_path = os.path.join(os.getcwd(),
                                                            'DB/databases_Cleaned/paper_preprocess/{}'.format(DB_name),
                                                            'len_closeness_{}_len_period_{}_len_trend_{}'.format(
                                                                len_closeness, len_period,
                                                                len_trend))
                        os.removedirs(preprocessed_DB_path)
                    print('-'*100)
