# -*- coding: utf-8 -*-

import collections
import fractions

import numpy
import scipy.signal

from color_modem import utils


class QamConfig(collections.namedtuple('QamConfig', ['fsc', 'bandwidth3db', 'bandwidth20db', 'lines'])):
    def __new__(cls, *args, **kwargs):
        self = QamConfig.__base__.__new__(cls, *args, **kwargs)
        if self.lines not in {525, 625}:
            raise RuntimeError('%d is not a supported line count' % (self.lines,))
        return self

    @property
    def framerate(self):
        return (30000.0 / 1001.0) if self.lines == 525 else 25.0

    @property
    def line_shift(self):
        return 2.0 * numpy.pi * ((self.fsc / (self.framerate * self.lines)) % 1.0)

    @property
    def field_shift(self):
        return numpy.pi * (((self.lines + 1) * self.fsc / (self.framerate * self.lines)) % 2.0)

    @property
    def frame_shift(self):
        return 2.0 * numpy.pi * ((self.fsc / self.framerate) % 1.0)

    @property
    def frame_cycle(self):
        return fractions.Fraction(self.fsc / self.framerate).denominator

    def is_alternate_line(self, frame, line):
        # For 525-line system:
        #
        # NOTE: Lines 21-22, 283-285 and 263 are defined to be active by the
        # analogue specifications, but are by convention not encoded digitally.
        #
        # digital line 0 ~ analog line 23 (odd fields 1 & 3)
        # digital line 1 ~ analog line 286 (even fields 2 & 4)
        # digital line 2 - analog line 24 (odd fields 1 & 3)
        # digital line 3 - analog line 287 (even fields 2 & 4)
        #
        # For 625-line system:
        #
        # digital line 0 ~ analog line 23 (even fields 1 & 3)
        # digital line 1 ~ analog line 336 (odd fields 2 & 4)
        # digital line 2 - analog line 24 (even fields 1 & 3)
        # digital line 3 - analog line 337 (odd fields 2 & 4)
        return (((line % 2) ^ ((line // 2) % 2)) == frame % 2) == (self.lines == 525)


class QamColorModem(object):
    def __init__(self, wc, wp, ws, gpass, gstop):
        self.carrier_phase_step = 0.5 * numpy.pi * wc
        self._chroma_precorrect_lowpass = utils.iirdesign(wp, ws, gpass, gstop)
        self._extract_chroma2x, self.extract_chroma_phase_shift, self._remove_chroma2x = \
            utils.irrsplitter(0.5 * wc, 0.5 * wp, 0.5 * ws, gpass, gstop, pass_phase_shift=True)
        self._demod_lowpass = utils.iirfilter(6, wc - 0.5 * ws, rs=48.0, btype='lowpass', ftype='cheby2')

    def _modulate_chroma(self, start_phase, u, v):
        assert len(u) == len(v)
        u = self._chroma_precorrect_lowpass(u)
        v = self._chroma_precorrect_lowpass(v)
        phase = numpy.linspace(start=start_phase, stop=start_phase + len(u) * 2.0 * self.carrier_phase_step, num=len(u),
                               endpoint=False) % (2.0 * numpy.pi)
        return numpy.sin(phase) * u + numpy.cos(phase) * v

    def modulate(self, start_phase, y, u, v):
        assert len(y) == len(u) == len(v)
        y = numpy.array(y, copy=False)
        chroma = self._modulate_chroma(start_phase, u, v)
        return y + chroma

    def extract_chroma(self, composite):
        composite2x = scipy.signal.resample_poly(composite, up=2, down=1)
        chroma2x = self._extract_chroma2x(composite2x)
        return scipy.signal.resample_poly(chroma2x, up=1, down=2)

    def demodulate(self, start_phase, composite, strip_chroma=True):
        shifted_phase = start_phase + self.extract_chroma_phase_shift
        composite2x = scipy.signal.resample_poly(composite, up=2, down=1)
        chroma2x = self._extract_chroma2x(composite2x)
        phase = numpy.linspace(start=shifted_phase, stop=shifted_phase + len(chroma2x) * self.carrier_phase_step,
                               num=len(chroma2x), endpoint=False) % (2.0 * numpy.pi)
        u2x = 2.0 * numpy.sin(phase) * chroma2x
        v2x = 2.0 * numpy.cos(phase) * chroma2x
        u2x = self._demod_lowpass(u2x)
        v2x = self._demod_lowpass(v2x)
        u = scipy.signal.resample_poly(u2x, up=1, down=2)
        v = scipy.signal.resample_poly(v2x, up=1, down=2)
        y = composite
        if strip_chroma:
            y = scipy.signal.resample_poly(self._remove_chroma2x(composite2x), up=1, down=2)
        return y, u, v


class AbstractQamColorModem(object):
    def __init__(self, fs, config):
        self.config = config
        self.qam = QamColorModem(2.0 * config.fsc / fs, 2.0 * config.bandwidth3db / fs, 2.0 * config.bandwidth20db / fs,
                                 3.0, 20.0)

    def calculate_start_phase(self, frame, line):
        frame %= self.config.frame_cycle
        frame_shift = (frame * self.config.frame_shift) % (2.0 * numpy.pi)
        field_shift = (line % 2) * self.config.field_shift
        line_shift = ((line // 2) * self.config.line_shift) % (2.0 * numpy.pi)
        return (frame_shift + field_shift + line_shift) % (2.0 * numpy.pi)

    def modulate(self, frame, line, r, g, b):
        return self.modulate_yuv(frame, line, *self.encode_yuv(r, g, b))

    def demodulate(self, frame, line, *args, **kwargs):
        return self.decode_yuv(*self.demodulate_yuv(frame, line, *args, **kwargs))


"""
Comb filtering calculations:

LS - line shift

PAL
===

PAL-BDGHIK:     LS = %pi*1879/1250 ~= %pi*3/2
PAL-M:          LS = %pi/2
PAL-N:          LS = %pi*629/1250 ~= %pi/2

n%2 == 0
PAL(n) = Y(x) + U(x)*sin(wt(x)+LS*n) - V(x)*cos(wt(x)+LS*n)
PAL(n+1) = Y(x) + U(x)*sin(wt(x)+LS*(n+1)) + V(x)*cos(wt(x)+LS*(n+1))

PAL(n+1) - PAL(n) = U(x)*(sin(wt(x)+(n+1)*LS) - sin(wt(x)+n*LS)) + V(x)*(cos(wt(x)+(n+1)*LS) + cos(wt(x)+n*LS))
PAL(n+1) - PAL(n) = U(x)*2*sin(LS/2)*cos(wt(x) + LS*(n + 1/2)) + V(x)*2*cos(LS/2)*cos(wt(x) + LS*(n + 1/2))
PAL(n+1) - PAL(n) = 2*(U(x)*sin(LS/2) + V(x)*cos(LS/2))*cos(wt(x) + LS*(n + 1/2))
PAL_BDGHIK(n+1) - PAL_BDGHIK(n) ~= (U(x) - V(x))*sqrt(2)*cos(wt(x) + LS*(n + 1/2))
PAL_MN(n+1) - PAL_MN(n) ~= (U(x) + V(x))*sqrt(2)*cos(wt(x) + LS*(n + 1/2))

PAL(n+2) - PAL(n+1) = U(x)*(sin(wt(x)+LS*(n+2)) - sin(wt(x)+LS*(n+1))) - V(x)*(cos(wt(x)+LS*(n+2)) + cos(wt(x)+LS*(n+1)))
PAL(n+2) - PAL(n+1) = U(x)*2*sin(LS/2)*cos(wt(x) + LS*(n + 3/2)) - V(x)*2*cos(LS/2)*cos(wt(x) + LS*(n + 3/2))
PAL(n+2) - PAL(n+1) = 2*(U(x)*sin(LS/2) - V(x)*cos(LS/2))*cos(wt(x) + LS*(n + 3/2))
PAL_BDGHIK(n+2) - PAL_BDGHIK(n+1) ~= (U(x) + V(x))*sqrt(2)*cos(wt(x) + LS*(n + 3/2))
PAL_MN(n+2) - PAL_MN(n+1) ~= (U(x) - V(x))*sqrt(2)*cos(wt(x) + LS*(n + 3/2))

PAL(n+1) + PAL(n) = 2*Y(x) + U(x)*(sin(wt(x)+LS*(n+1)) + sin(wt(x)+LS*n)) + V(x)*(cos(wt(x)+LS*(n+1)) - cos(wt(x)+LS*n))
PAL(n+1) + PAL(n) = 2*Y(x) + U(x)*2*cos(LS/2)*sin(wt(x) + LS*(n + 1/2)) - V(x)*2*sin(LS/2)*sin(wt(x) + LS*(n + 1/2))
PAL(n+1) + PAL(n) = 2*Y(x) + 2*(U(x)*cos(LS/2) - V(x)*sin(LS/2))*sin(wt(x) + LS*(n + 1/2))
PAL_BDGHIK(n+1) + PAL_BDGHIK(n) ~= 2*Y(x) - (U(x) + V(x))*sqrt(2)*sin(wt(x) + LS*(n + 1/2))
PAL_MN(n+1) + PAL_MN(n) ~= 2*Y(x) + (U(x) - V(x))*sqrt(2)*sin(wt(x) + LS*(n + 1/2))

PAL(n+2) + PAL(n+1) = 2*Y(x) + U(x)*(sin(wt(x)+LS*(n+2)) + sin(wt(x)+LS*(n+1))) - V(x)*(cos(wt(x)+LS*(n+2)) - cos(wt(x)+LS*(n+1)))
PAL(n+2) + PAL(n+1) = 2*Y(x) + U(x)*2*cos(LS/2)*sin(wt(x) + LS*(n + 3/2)) + V(x)*2*sin(LS/2)*sin(wt(x) + LS*(n + 3/2))
PAL(n+2) + PAL(n+1) = 2*Y(x) + 2*(U(x)*cos(LS/2) + V(x)*sin(LS/2))*sin(wt(x) + LS*(n + 3/2))
PAL_BDGHIK(n+2) + PAL_BDGHIK(n+1) ~= 2*Y(x) - (U(x) - V(x))*sqrt(2)*sin(wt(x) + LS*(n + 3/2))
PAL_MN(n+2) + PAL_MN(n+1) ~= 2*Y(x) + (U(x) + V(x))*sqrt(2)*sin(wt(x) + LS*(n + 3/2))

Exact calculations:
n, n+1:
{ U(x)*sin(LS/2) + V(x)*cos(LS/2) = DIFSIG
{ U(x)*cos(LS/2) - V(x)*sin(LS/2) = SUMSIG

{ U(x) = (DIFSIG + SUMSIG)/(2*sin(LS/2))
{ V(x) = (DIFSIG - SUMSIG)/(2*cos(LS/2))

n+1, n+2:
{ U(x)*sin(LS/2) - V(x)*cos(LS/2) = DIFSIG
{ U(x)*cos(LS/2) + V(x)*sin(LS/2) = SUMSIG

{ U(x) = (DIFSIG + SUMSIG)/(2*cos(LS/2))
{ V(x) = (DIFSIG - SUMSIG)/(-2*sin(LS/2))


NTSC
====

NTSC-IM:    LS = %pi
NTSC443:    LS = %pi*281197/180000

NTSC(n) = Y(x) + U(x)*sin(wt(x)+LS*n) + V(x)*cos(wt(x)+LS*n)

NTSC(n+1) - NTSC(n) = U(x)*(sin(wt(x)+LS*(n+1)) - sin(wt(x)+LS*n)) + V(x)*(cos(wt(x)+LS*(n+1)) - cos(wt(x)+LS*n))
NTSC(n+1) - NTSC(n) = U(x)*2*sin(LS/2)*cos(wt(x) + LS*(n + 1/2)) - V(x)*2*sin(LS/2)*sin(wt(x) + LS*(n + 1/2))
NTSC(n+1) - NTSC(n) = 2*sin(LS/2)*(U(x)*cos(wt(x) + LS*(n + 1/2)) - V(x)*sin(wt(x) + LS*(n + 1/2)))
NTSC_IM(n+1) - NTSC_IM(n) = 2*(U(x)*cos(wt(x) + LS*(n + 1/2)) - V(x)*sin(wt(x) + LS*(n + 1/2)))

NTSC(n+1) + NTSC(n) = 2*Y(x) + U(x)*(sin(wt(x)+LS*(n+1)) + sin(wt(x)+LS*n)) + V(x)*(cos(wt(x)+LS*(n+1)) + cos(wt(x)+LS*n))
NTSC(n+1) + NTSC(n) = 2*Y(x) + U(x)*2*cos(LS/2)*sin(wt(x) + LS*(n + 1/2)) + V(x)*2*cos(LS/2)*cos(wt(x) + LS*(n + 1/2))
NTSC(n+1) + NTSC(n) = 2*Y(x) + 2*cos(LS/2)*(U(x)*sin(wt(x) + LS*(n + 1/2)) + V(x)*cos(wt(x) + LS*(n + 1/2)))
NTSC_IM(n+1) + NTSC_IM(n) = 2*Y(x)
"""
