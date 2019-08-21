# -*- coding: utf-8 -*-

import collections

import numpy
import scipy.signal

from color_modem import utils

QamConfig = collections.namedtuple('QamConfig', ['fsc', 'bandwidth3db', 'bandwidth20db'])


class QamColorModem(object):
    def __init__(self, wc, wp, ws, gpass, gstop):
        self.carrier_phase_step = 0.5 * numpy.pi * wc
        self._chroma_precorrect_lowpass = utils.iirdesign(wp, ws, gpass, gstop)
        self._extract_chroma2x, self._remove_chroma2x = utils.iirsplitter(0.5 * wc, 0.5 * wp, 0.5 * ws, gpass, gstop)
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

    @property
    def extract_chroma_phase_shift(self):
        return self._extract_chroma2x.phase_shift

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


class AbstractQamColorModem(utils.ConstantFrequencyCarrier):
    def __init__(self, line_config, config):
        self.line_config = line_config
        self.config = config
        self.qam = QamColorModem(2.0 * config.fsc / line_config.fs, 2.0 * config.bandwidth3db / line_config.fs,
                                 2.0 * config.bandwidth20db / line_config.fs, 3.0, 20.0)

    def modulate(self, frame, line, r, g, b):
        return self.modulate_components(frame, line, *self.encode_components(r, g, b))

    def demodulate(self, frame, line, *args, **kwargs):
        return self.decode_components(*self.demodulate_components(frame, line, *args, **kwargs))
