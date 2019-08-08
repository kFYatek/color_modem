# -*- coding: utf-8 -*-
import numpy
import scipy.signal


def chroma_precorrect_lowpass(wp, ws, gpass, gstop):
    b, a = scipy.signal.iirdesign(wp, ws, gpass, gstop, ftype='butter')
    shift = int(numpy.round(scipy.signal.group_delay((b, a), [0.0])[1]))
    return lambda x: scipy.signal.lfilter(b, a, numpy.concatenate((x, numpy.zeros(shift))))[shift:]
