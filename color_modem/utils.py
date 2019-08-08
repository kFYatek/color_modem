# -*- coding: utf-8 -*-

import numpy
import scipy.signal


def _make_filter(b, a, wp, btype, shift, phase_shift):
    wp = numpy.atleast_1d(wp)
    if len(wp) > 1 and btype.lower() not in {'bs', 'bandstop', 'bands', 'stop', 'bandstop'}:
        shiftfreq = numpy.average(wp)
    else:
        shiftfreq = 0.0

    if shift:
        shift = int(numpy.round(scipy.signal.group_delay((b, a), [shiftfreq], fs=2.0)[1]))
    else:
        shift = 0

    if shift == 0:
        filter = lambda x: scipy.signal.lfilter(b, a, x)
    else:
        filter = lambda x: scipy.signal.lfilter(b, a, numpy.concatenate((x, numpy.zeros(shift))))[shift:]

    if phase_shift:
        phase_shift = (numpy.angle(
            scipy.signal.freqz(b, a, worN=[shiftfreq], fs=2.0)[1][0]) + shift * numpy.pi * shiftfreq) % (2.0 * numpy.pi)
        return filter, phase_shift
    else:
        return filter


def iirfilter(N, Wn, rp=None, rs=None, btype='band', ftype='butter', shift=True, phase_shift=False):
    b, a = scipy.signal.iirfilter(N, Wn, rp, rs, btype, ftype=ftype)
    return _make_filter(b, a, Wn, btype, shift, phase_shift)


def iirdesign(wp, ws, gpass, gstop, ftype='butter', shift=True, phase_shift=False):
    b, a = scipy.signal.iirdesign(wp, ws, gpass, gstop, ftype=ftype)
    return _make_filter(b, a, wp, 'band', shift, phase_shift)


def iirdesign_wc(wc, wp, ws, gpass, gstop, ftype='butter', shift=True, phase_shift=False):
    return iirdesign([wc - wp, wc + wp], [wc - ws, wc + ws], gpass, gstop, ftype, shift, phase_shift)


def irrsplitter(wc, wp, ws, gpass, gstop, ftype='butter', shift=True, pass_phase_shift=False, stop_phase_shift=False):
    def invert_db(db):
        return -(20.0 * numpy.log10(1.0 - 10.0 ** (-db / 20.0)))

    bpass = iirdesign_wc(wc, wp, ws, gpass, gstop, ftype, shift, pass_phase_shift)
    bstop = iirdesign_wc(wc, ws, wp, invert_db(gstop), invert_db(gpass), ftype, shift, stop_phase_shift)

    return (bpass if pass_phase_shift else (bpass,)) + (bstop if stop_phase_shift else (bstop,))
