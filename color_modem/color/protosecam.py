# -*- coding: utf-8 -*-

import numpy
import scipy.signal

from color_modem import qam, utils


class ProtoSecamVariant(qam.QamConfig):
    pass


# References:
# - F. Schröter, "Fernsehtechnik: Technik des elektronischen Fernsehens", p. 498-501
#   https://books.google.pl/books?id=bEL8CAAAQBAJ&pg=PA498
#   https://books.google.pl/books?id=5hvLBgAAQBAJ&pg=PA502
# - Wireless World, September 1957, p. 426-429
#   https://www.americanradiohistory.com/Archive-Wireless-World/50s/Wireless-World-1957-09.pdf
#
# Both those sources mention 8.57 MHz as the subcarrier frequency, but also mention that it's an odd multiple of half
# the line frequency. The line frequency of the 819-line system is 20475 Hz, and odd multiples of 10237.5 Hz nearest to
# 8.37 MHz are 8364037.5 Hz (817 * 10237.5 Hz) and 8384512.5 Hz (819 * 10237.5 Hz). Neither can be sensibly rounded to
# 8.37 MHz. Weird. Going with 8384512.5 Hz, because 819 = 3 * 3 * 7 * 13, while 817 is a prime number, and large prime
# numbers are unlikely to be used for frequency multiplication in the 1950s.
ProtoSecamVariant.SECAM_1957 = ProtoSecamVariant(fsc=8384512.5, bandwidth3db=800000.0, bandwidth20db=2000000.0)


class ProtoSecamModem(utils.ConstantFrequencyCarrier):
    def __init__(self, line_config, variant=ProtoSecamVariant.SECAM_1957, premod_luma_filter=True):
        self.line_config = line_config
        self.config = variant
        self._premod_luma_filter = premod_luma_filter
        self._carrier_phase_step = numpy.pi * variant.fsc / line_config.fs
        self._chroma_precorrect_lowpass = utils.iirdesign(2.0 * variant.bandwidth3db / line_config.fs,
                                                          2.0 * variant.bandwidth20db / line_config.fs, 3.0, 20.0)
        self._demodulate_resample_factor = 3
        self._extract_chroma_up, self._remove_chroma_up = utils.iirsplitter(
            2.0 * variant.fsc / (self._demodulate_resample_factor * line_config.fs),
            2.0 * variant.bandwidth3db / (self._demodulate_resample_factor * line_config.fs),
            2.0 * variant.bandwidth20db / (self._demodulate_resample_factor * line_config.fs), 3.0, 20.0)

        if variant.fsc < variant.bandwidth20db:
            post_demod_bandwidth = variant.bandwidth3db
        else:
            post_demod_bandwidth = variant.bandwidth20db
        self._chroma_up_post_demod_filter = utils.iirdesign(
            2.0 * min(post_demod_bandwidth, variant.fsc - post_demod_bandwidth) / (
                    self._demodulate_resample_factor * line_config.fs),
            2.0 * max(post_demod_bandwidth, variant.fsc - post_demod_bandwidth) / (
                    self._demodulate_resample_factor * line_config.fs), 3.0, 20.0)
        self._last_frame = -1
        self._last_line = -1
        self._last_chroma = None

    @staticmethod
    def encode_components(r, g, b):
        assert len(r) == len(g) == len(b)
        luma = 0.3 * r + 0.59 * g + 0.11 * b
        dr = 1.001 * r - 0.8437 * g - 0.1573 * b
        db = -0.336 * r - 0.6608 * g + 0.9968 * b
        return luma, dr, db

    @staticmethod
    def decode_components(luma, dr, db):
        assert len(luma) == len(dr) == len(db)
        r = luma + 0.6993006993006993 * dr
        g = luma - 0.3555766267630674 * dr - 0.1664648910411622 * db
        b = luma + 0.8928571428571429 * db
        return r, g, b

    def modulate(self, frame, line, r, g, b):
        return self.modulate_components(frame, line, *self.encode_components(r, g, b))

    def modulate_components(self, frame, line, luma, dr, db):
        if not self.line_config.is_alternate_line(frame, line):
            chroma = dr
        else:
            chroma = db
        chroma = self._chroma_precorrect_lowpass(chroma)
        chroma = 0.125 * (1.0 + chroma)

        if self._premod_luma_filter:
            luma_up = scipy.signal.resample_poly(luma, up=self._demodulate_resample_factor, down=1)
            luma_up = self._remove_chroma_up(luma_up)
            luma = scipy.signal.resample_poly(luma_up, up=1, down=self._demodulate_resample_factor)

        start_phase = self.start_phase(frame, line)
        phase = numpy.linspace(start=start_phase, stop=start_phase + len(chroma) * 2.0 * self._carrier_phase_step,
                               num=len(chroma), endpoint=False) % (2.0 * numpy.pi)
        return luma + numpy.cos(phase) * chroma

    def demodulate(self, frame, line, composite):
        if frame != self._last_frame or line != self._last_line + 2 or self._last_chroma is None:
            self._last_chroma = numpy.zeros(len(composite))

        composite_up = scipy.signal.resample_poly(composite, up=self._demodulate_resample_factor, down=1)
        chroma_up = self._extract_chroma_up(composite_up)
        chroma_up = 0.5 * numpy.pi * numpy.abs(chroma_up)
        chroma_up = self._chroma_up_post_demod_filter(chroma_up)
        luma_up = self._remove_chroma_up(composite_up)
        luma = scipy.signal.resample_poly(luma_up, up=1, down=self._demodulate_resample_factor)
        chroma = scipy.signal.resample_poly(chroma_up, up=1, down=self._demodulate_resample_factor)
        chroma = 8.0 * chroma - 1.0

        if not self.line_config.is_alternate_line(frame, line):
            self._last_chroma, dr, db = chroma, chroma, self._last_chroma
        else:
            self._last_chroma, dr, db = chroma, self._last_chroma, chroma

        self._last_frame = frame
        self._last_line = line
        return self.decode_components(luma, dr, db)
