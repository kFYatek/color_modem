# -*- coding: utf-8 -*-
import numpy


class AbstractCombModem:
    BLANK_LINE = numpy.zeros(720)

    def __init__(self, backend):
        self.backend = backend
        self._last_frame = -1
        self._last_line = -1
        self._last_composite = numpy.zeros(720)

    def modulate_yuv(self, frame, line, y, u, v):
        return self.backend.modulate_yuv(frame, line, y, u, v)

    def modulate(self, frame, line, r, g, b):
        return self.backend.modulate(frame, line, r, g, b)

    def demodulate_yuv(self, frame, line, composite, strip_chroma=True, *args, **kwargs):
        if frame != self._last_frame or line != self._last_line + 2:
            y, u, v = self.backend.demodulate_yuv(frame, line, composite, strip_chroma)
        else:
            y, u, v = self.demodulate_yuv_combed(frame, line, self._last_composite, composite, *args, **kwargs)
            if strip_chroma:
                y = numpy.array(y)
                new_chroma = self.backend.modulate_yuv(frame, line, self.BLANK_LINE, u, v)
                for i in range(len(y)):
                    y[i] -= new_chroma[i]
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


class SimpleCombModem:
    BLANK_LINE = numpy.zeros(720)

    @staticmethod
    def _minavg(val1, val2):
        sign1 = 1.0 if val1 >= 0.0 else -1.0
        sign2 = 1.0 if val2 >= 0.0 else -1.0
        if sign1 != sign2:
            return 0.0
        else:
            return sign1 * min(abs(val1), abs(val2))

    def __init__(self, backend, avg=None, delay=False):
        self.backend = backend
        self.demodulation_delay = 1 if delay else 0
        self._last_frame = -1
        self._last_line = -1
        self._last_demodulated = None

        if avg is not None:
            self._avg = avg
        else:
            self._avg = self._minavg

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
            y = self._last_demodulated[0] if self.demodulation_delay else curr[0]
            u = numpy.zeros(len(y))
            v = numpy.zeros(len(y))
            for i in range(len(y)):
                u[i] = self._avg(self._last_demodulated[1][i], curr[1][i])
                v[i] = self._avg(self._last_demodulated[2][i], curr[2][i])
            if strip_chroma:
                y = numpy.array(y)
                new_chroma = self.backend.modulate_yuv(frame, line - 2 * self.demodulation_delay, self.BLANK_LINE, u, v)
                for i in range(len(y)):
                    y[i] -= new_chroma[i]
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
    def __init__(self, backend, avg=None):
        super().__init__(backend, avg, True)
