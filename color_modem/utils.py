# -*- coding: utf-8 -*-

import fractions

import numpy
import scipy.signal


class FilterFunction(object):
    def __init__(self, b, a, wp, btype, shift):
        self._b = b
        self._a = a
        wp = numpy.atleast_1d(wp)
        if len(wp) > 1 and btype.lower() not in {'bs', 'bandstop', 'bands', 'stop', 'bandstop'}:
            shiftfreq = numpy.average(wp)
        else:
            shiftfreq = 0.0

        if shift:
            self._shift = int(numpy.round(scipy.signal.group_delay((b, a), [shiftfreq], fs=2.0)[1]))
        else:
            self._shift = 0

        self.phase_shift = (numpy.angle(
            scipy.signal.freqz(b, a, worN=[shiftfreq], fs=2.0)[1][0]) + self._shift * numpy.pi * shiftfreq) % (
                                   2.0 * numpy.pi)

    def __call__(self, x):
        if self._shift == 0:
            return scipy.signal.lfilter(self._b, self._a, x)
        elif self._shift > 0:
            return scipy.signal.lfilter(self._b, self._a, numpy.concatenate((x, x[-1] * numpy.ones(self._shift))))[
                   self._shift:]
        else:
            return scipy.signal.lfilter(self._b, self._a, numpy.concatenate((x[0] * numpy.ones(-self._shift), x)))[
                   :self._shift]


def iirfilter(N, Wn, rp=None, rs=None, btype='band', ftype='butter', shift=True):
    b, a = scipy.signal.iirfilter(N, Wn, rp, rs, btype, ftype=ftype)
    return FilterFunction(b, a, Wn, btype, shift)


def iirdesign(wp, ws, gpass, gstop, ftype='butter', shift=True):
    smallest = numpy.nextafter(0.0, 1.0)
    largest = numpy.nextafter(1.0, 0.0)
    b, a = scipy.signal.iirdesign(numpy.maximum(wp, smallest), numpy.minimum(ws, largest), gpass, gstop, ftype=ftype)
    btype = 'band'
    if len(numpy.atleast_1d(wp)) > 1 and len(numpy.atleast_1d(ws)) > 1 and ws[0] > wp[0]:
        btype = 'bandstop'
    return FilterFunction(b, a, wp, btype, shift)


def iirdesign_wc(wc, wp, ws, gpass, gstop, ftype='butter', shift=True):
    return iirdesign([wc - wp, wc + wp], [wc - ws, wc + ws], gpass, gstop, ftype, shift)


def iirsplitter(wc, wp, ws, gpass, gstop, ftype='butter', shift=True):
    def invert_db(db):
        return -(20.0 * numpy.log10(1.0 - 10.0 ** (-db / 20.0)))

    bpass = iirdesign_wc(wc, wp, ws, gpass, gstop, ftype, shift)
    bstop = iirdesign_wc(wc, ws, wp, invert_db(gstop), invert_db(gpass), ftype, shift)
    return bpass, bstop


class ConstantFrequencyCarrier(object):
    @property
    def line_shift(self):
        return 2.0 * numpy.pi * ((self.config.fsc / (
                self.line_config.line_standard.frame_rate * self.line_config.line_standard.total_lines)) % 1.0)

    @property
    def frame_shift(self):
        return 2.0 * numpy.pi * ((self.config.fsc / self.line_config.line_standard.frame_rate) % 1.0)

    @property
    def frame_cycle(self):
        return fractions.Fraction(
            self.config.fsc / self.line_config.line_standard.frame_rate).limit_denominator().denominator

    def start_phase(self, frame, line):
        reference_line = min(self.line_config.line_standard.odd_field_first_active_line,
                             self.line_config.line_standard.even_field_first_active_line)
        frame %= self.frame_cycle
        frame_shift = (frame * self.frame_shift) % (2.0 * numpy.pi)
        line_shift = ((self.line_config.analog_line(line) - reference_line) * self.line_shift) % (2.0 * numpy.pi)
        return (frame_shift + line_shift) % (2.0 * numpy.pi)
