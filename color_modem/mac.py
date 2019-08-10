# -*- coding: utf-8 -*-

import collections
import fractions

import numpy
import scipy.signal

MacVariant = collections.namedtuple('MacVariant', ['width'])

MacVariant.D2MAC_12MHZ = MacVariant(1080)
MacVariant.D2MAC_7MHZ = MacVariant(720)


class MacModem(object):
    def __init__(self, variant_or_width=MacVariant.D2MAC_12MHZ):
        try:
            self._width = int(variant_or_width.width)
        except AttributeError:
            self._width = int(variant_or_width)
        self._last_frame = -1
        self._last_line = -1
        self._last_chroma = None

    @staticmethod
    def encode_mac_components(r, g, b):
        assert len(r) == len(g) == len(b)
        luma = 0.299 * r + 0.587 * g + 0.114 * b
        dr = 0.649827 * r - 0.544149 * g - 0.105678 * b
        db = -0.219167 * r - 0.430271 * g + 0.649438 * b
        return luma, dr, db

    @staticmethod
    def decode_mac_components(luma, dr, db):
        assert len(luma) == len(dr) == len(db)
        r = luma + 1.0787486515641855 * dr
        g = luma - 0.5494818514781797 * dr - 0.2649492993950324 * db
        b = luma + 1.364256480218281 * db
        return r, g, b

    @staticmethod
    def _is_alternate_line(frame, line):
        return ((line % 2) ^ ((line // 2) % 2)) != frame % 2

    def modulate_mac_components(self, frame, line, luma, dr, db):
        if not MacModem._is_alternate_line(frame, line):
            chroma = dr
        else:
            chroma = db

        luma_resample_fraction = fractions.Fraction(720, len(luma))
        chroma_resample_fraction = fractions.Fraction(360, len(chroma))
        if luma_resample_fraction.numerator != luma_resample_fraction.denominator:
            luma = scipy.signal.resample_poly(luma, up=luma_resample_fraction.numerator,
                                              down=luma_resample_fraction.denominator)
        if chroma_resample_fraction.numerator != chroma_resample_fraction.denominator:
            chroma = scipy.signal.resample_poly(chroma, up=chroma_resample_fraction.numerator,
                                                down=chroma_resample_fraction.denominator)

        chroma += 0.5
        output = 0.5 * numpy.ones(1080)
        # output[15:18] = attenuated_chroma
        output[15] = 0.4375 + 0.125 * chroma[2]
        output[16] = 0.25 + 0.5 * chroma[3]
        output[17] = 0.0625 + 0.875 * chroma[4]
        output[18:369] = chroma[5:356]
        output[369] = 0.875 * chroma[356] + 0.125 * luma[8]
        output[370] = 0.5 * chroma[357] + 0.5 * luma[9]
        output[371] = 0.125 * chroma[358] + 0.875 * luma[10]
        output[372:1071] = luma[11:710]
        output[1071] = 0.0625 + 0.875 * luma[710]
        output[1072] = 0.25 + 0.5 * luma[711]
        output[1073] = 0.4375 + 0.125 * luma[712]

        output_resample_fraction = fractions.Fraction(self._width, len(output))
        if output_resample_fraction.numerator != output_resample_fraction.denominator:
            output = scipy.signal.resample_poly(output, up=output_resample_fraction.numerator,
                                                down=output_resample_fraction.denominator)
        return output

    def demodulate(self, frame, line, composite):
        if frame != self._last_frame or line != self._last_line + 2 or self._last_chroma is None:
            self._last_chroma = numpy.zeros(720)

        composite_resample_fraction = fractions.Fraction(1080, len(composite))
        if composite_resample_fraction.numerator != composite_resample_fraction.denominator:
            composite = scipy.signal.resample_poly(composite, up=composite_resample_fraction.numerator,
                                                   down=composite_resample_fraction.denominator)

        luma = 0.5 * numpy.ones(720)
        chroma = 0.5 * numpy.ones(360)

        luma[8:710] = composite[369:1071]
        luma[710] = (composite[1071] - 0.0625) / 0.875
        luma[711] = 2.0 * composite[1072] - 0.5
        luma[712] = 8.0 * composite[1073] - 3.5

        chroma[2] = 8.0 * composite[15] - 3.5
        chroma[3] = 2.0 * composite[16] - 0.5
        chroma[4] = (composite[17] - 0.0625) / 0.875
        chroma[5:359] = composite[18:372]

        chroma = scipy.signal.resample_poly(chroma, up=2, down=1) - 0.5

        if not MacModem._is_alternate_line(frame, line):
            dr = chroma
            self._last_chroma, db = dr, self._last_chroma
        else:
            db = chroma
            self._last_chroma, dr = db, self._last_chroma

        self._last_frame = frame
        self._last_line = line
        return self.decode_mac_components(luma, dr, db)

    def modulate(self, frame, line, r, g, b):
        return self.modulate_mac_components(frame, line, *MacModem.encode_mac_components(r, g, b))

    @staticmethod
    def encode_composite_level(value):
        return value

    @staticmethod
    def decode_composite_level(value):
        return value


class AveragingMacModem(MacModem):
    def __init__(self, *args, **kwargs):
        super(AveragingMacModem, self).__init__(*args, **kwargs)
        self.modulation_delay = 1
        self._last_modulated_frame = -1
        self._last_modulated_line = -1
        self._last_y = None
        self._last_u = None
        self._last_v = None

    def modulate_mac_components(self, frame, line, y, u, v):
        if frame != self._last_modulated_frame or line != self._last_modulated_line + 2 \
                or self._last_u is None or self._last_v is None:
            self._last_y = y
            self._last_u = u
            self._last_v = v
        self._last_y, y = y, self._last_y
        self._last_u, u = u, 0.5 * (u + self._last_u)
        self._last_v, v = v, 0.5 * (v + self._last_v)
        self._last_modulated_frame = frame
        self._last_modulated_line = line
        return super(AveragingMacModem, self).modulate_mac_components(frame, line - 2, y, u, v)
