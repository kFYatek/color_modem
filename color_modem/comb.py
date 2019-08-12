# -*- coding: utf-8 -*-

import numpy
import scipy.signal

from color_modem import utils


def minavg(val1, val2):
    sign = (1.0 - numpy.signbit(val1)) - numpy.signbit(val2)
    return sign * numpy.minimum(numpy.abs(val1), numpy.abs(val2))


def _notch(qam_modem, q):
    b, a = scipy.signal.iirnotch(2.0 * qam_modem.config.fsc / qam_modem.fs, q)
    return lambda x: scipy.signal.lfilter(b, a, x)


class AbstractCombModem(object):
    def __init__(self, backend, notch=0.0):
        self.backend = backend
        self._last_frame = -1
        self._last_line = -1
        self._last_composite = None
        self.notch = None
        if notch:
            self.notch = _notch(backend, notch)

    @property
    def config(self):
        return self.backend.config

    @property
    def fs(self):
        return self.backend.fs

    def modulate_yuv(self, frame, line, y, u, v):
        return self.backend.modulate_yuv(frame, line, y, u, v)

    def modulate(self, frame, line, r, g, b):
        return self.backend.modulate(frame, line, r, g, b)

    def demodulate_yuv(self, frame, line, composite, strip_chroma=True, *args, **kwargs):
        if frame != self._last_frame or line != self._last_line + 2 or self._last_composite is None:
            y, u, v = self.backend.demodulate_yuv(frame, line, composite, strip_chroma)
        else:
            y, u, v = self.demodulate_yuv_combed(frame, line, self._last_composite, composite, *args, **kwargs)
            if strip_chroma:
                y = y - self.backend.modulate_yuv(frame, line, numpy.zeros(len(composite)), u, v)
                if self.notch:
                    y = self.notch(y)
        self._last_frame = frame
        self._last_line = line
        self._last_composite = numpy.array(composite)
        return y, u, v

    def decode_yuv(self, y, u, v):
        return self.backend.decode_yuv(y, u, v)

    def demodulate(self, *args, **kwargs):
        return self.backend.decode_yuv(*self.demodulate_yuv(*args, **kwargs))

    def encode_composite_level(self, value):
        return self.backend.encode_composite_level(value)

    def decode_composite_level(self, value):
        return self.backend.decode_composite_level(value)


class SimpleCombModem(object):
    def __init__(self, backend, notch=0.0, avg=None, delay=False):
        self.backend = backend
        self._own_delay = 1 if delay else 0
        self.modulation_delay = getattr(backend, 'modulation_delay', 0)
        self.demodulation_delay = getattr(backend, 'demodulation_delay', 0) + self._own_delay
        self._last_frame = -1
        self._last_line = -1
        self._last_demodulated = None

        if avg is not None:
            self._avg = avg
        else:
            self._avg = lambda a, b: 0.5 * (a + b)

        self._notch = None
        if notch:
            self._notch = _notch(backend, notch)

    def modulate_yuv(self, frame, line, y, u, v):
        return self.backend.modulate_yuv(frame, line, y, u, v)

    def modulate(self, frame, line, r, g, b):
        return self.backend.modulate(frame, line, r, g, b)

    def demodulate_yuv(self, frame, line, composite, strip_chroma=True, *args, **kwargs):
        if frame != self._last_frame or line != self._last_line + 2:
            curr = self.backend.demodulate_yuv(frame, line, composite, strip_chroma=False, *args, **kwargs)
            y, u, v = curr
        else:
            curr = self.backend.demodulate_yuv(frame, line, composite, strip_chroma=False, *args, **kwargs)
            y = self._last_demodulated[0] if self._own_delay else curr[0]
            u = self._avg(self._last_demodulated[1], curr[1])
            v = self._avg(self._last_demodulated[2], curr[2])
            if strip_chroma:
                y = y - self.backend.modulate_yuv(frame, line - 2 * self._own_delay, numpy.zeros(len(composite)), u, v)
                if self._notch:
                    y = self._notch(y)
        self._last_frame = frame
        self._last_line = line
        self._last_demodulated = curr
        return y, u, v

    def decode_yuv(self, y, u, v):
        return self.backend.decode_yuv(y, u, v)

    def demodulate(self, *args, **kwargs):
        return self.backend.decode_yuv(*self.demodulate_yuv(*args, **kwargs))

    def encode_composite_level(self, value):
        return self.backend.encode_composite_level(value)

    def decode_composite_level(self, value):
        return self.backend.decode_composite_level(value)


class Simple3DCombModem(SimpleCombModem):
    def __init__(self, backend, notch=0.0, avg=None):
        super(Simple3DCombModem, self).__init__(backend, notch, avg, True)
