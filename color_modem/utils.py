# -*- coding: utf-8 -*-

import fractions

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
    elif shift > 0:
        filter = lambda x: scipy.signal.lfilter(b, a, numpy.concatenate((x, x[-1] * numpy.ones(shift))))[shift:]
    else:
        filter = lambda x: scipy.signal.lfilter(b, a, numpy.concatenate((x[0] * numpy.ones(-shift), x)))[:shift]

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
    btype = 'band'
    if len(numpy.atleast_1d(wp)) > 1 and len(numpy.atleast_1d(ws)) > 1 and ws[0] > wp[0]:
        btype = 'bandstop'
    return _make_filter(b, a, wp, btype, shift, phase_shift)


def iirdesign_wc(wc, wp, ws, gpass, gstop, ftype='butter', shift=True, phase_shift=False):
    smallest = numpy.nextafter(0.0, 1.0)
    largest = numpy.nextafter(1.0, 0.0)
    return iirdesign([max(wc - wp, smallest), min(wc + wp, largest)], [max(wc - ws, smallest), min(wc + ws, largest)],
                     gpass, gstop, ftype, shift, phase_shift)


def iirsplitter(wc, wp, ws, gpass, gstop, ftype='butter', shift=True, pass_phase_shift=False, stop_phase_shift=False):
    def invert_db(db):
        return -(20.0 * numpy.log10(1.0 - 10.0 ** (-db / 20.0)))

    bpass = iirdesign_wc(wc, wp, ws, gpass, gstop, ftype, shift, pass_phase_shift)
    bstop = iirdesign_wc(wc, ws, wp, invert_db(gstop), invert_db(gpass), ftype, shift, stop_phase_shift)

    return (bpass if pass_phase_shift else (bpass,)) + (bstop if stop_phase_shift else (bstop,))


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
