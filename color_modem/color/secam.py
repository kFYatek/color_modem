# -*- coding: utf-8 -*-

import collections

import numpy
import scipy.signal

from color_modem import utils

SecamVariant = collections.namedtuple('SecamVariant',
                                      ['fsc_dr', 'fsc_db', 'fdev_dr', 'fdev_db', 'flimit_minbell', 'flimit_maxbell',
                                       'm0', 'bell_f0', 'bell_kn', 'bell_kd', 'lf_precorrect_f1', 'lf_precorrect_k'])

# Possible parameters of SECAM I, based on Sven Boetcher & Echkard Matzel's chapter in "Mediengegenwart"
SecamVariant.SECAM_I = SecamVariant(fsc_dr=4437500.0,
                                    fsc_db=4437500.0,
                                    fdev_dr=250000.0,
                                    fdev_db=250000.0,
                                    flimit_minbell=-250000.0,
                                    flimit_maxbell=250000.0,
                                    m0=0.2,
                                    bell_f0=4437500.0,
                                    bell_kn=1.0,
                                    bell_kd=1.0,
                                    lf_precorrect_f1=0.0,
                                    lf_precorrect_k=1.0)

# Possible parameters of SECAM II, based on Sven Boetcher & Echkard Matzel's chapter in "Mediengegenwart"
SecamVariant.SECAM_II = SecamVariant(fsc_dr=4437500.0,
                                     fsc_db=4437500.0,
                                     fdev_dr=250000.0,
                                     fdev_db=250000.0,
                                     flimit_minbell=-250000.0,
                                     flimit_maxbell=250000.0,
                                     m0=0.1,
                                     bell_f0=4437500.0,
                                     bell_kn=16.0,
                                     bell_kd=1.26,
                                     lf_precorrect_f1=0.0,
                                     lf_precorrect_k=1.0)

# SECAM III as originally proposed
SecamVariant.SECAM_III = SecamVariant(fsc_dr=4437500.0,
                                      fsc_db=4437500.0,
                                      fdev_dr=230000.0,
                                      fdev_db=230000.0,
                                      flimit_minbell=-450000.0,
                                      flimit_maxbell=350000.0,
                                      m0=0.1,
                                      bell_f0=4437500.0,
                                      bell_kn=16.0,
                                      bell_kd=1.26,
                                      lf_precorrect_f1=70000.0,
                                      lf_precorrect_k=5.6)

# SECAM IIIb / III opt. - the actual broadcast variant
SecamVariant.SECAM = SecamVariant(fsc_dr=4406250.0,
                                  fsc_db=4250000.0,
                                  fdev_dr=280000.0,
                                  fdev_db=230000.0,
                                  flimit_minbell=-386000.0,
                                  flimit_maxbell=470250.0,
                                  m0=0.115,
                                  bell_f0=4286000.0,
                                  bell_kn=16.0,
                                  bell_kd=1.26,
                                  lf_precorrect_f1=85000.0,
                                  lf_precorrect_k=3.0)

# SECAM-A was allegedly tested by ITA in 1962 - that must have been based on SECAM I.
SecamVariant.SECAM_A = SecamVariant(fsc_dr=2660000.0,
                                    fsc_db=2660000.0,
                                    fdev_dr=250000.0,
                                    fdev_db=250000.0,
                                    flimit_minbell=-250000.0,
                                    flimit_maxbell=250000.0,
                                    m0=0.2,
                                    bell_f0=2660000.0,
                                    bell_kn=1.0,
                                    bell_kd=1.0,
                                    lf_precorrect_f1=0.0,
                                    lf_precorrect_k=1.0)

# Possible variant of FM SECAM I for 819 lines
SecamVariant.SECAM_E = SecamVariant(fsc_dr=8370000.0,
                                    fsc_db=8370000.0,
                                    fdev_dr=250000.0,
                                    fdev_db=250000.0,
                                    flimit_minbell=-250000.0,
                                    flimit_maxbell=250000.0,
                                    m0=0.2,
                                    bell_f0=8370000.0,
                                    bell_kn=1.0,
                                    bell_kd=1.0,
                                    lf_precorrect_f1=0.0,
                                    lf_precorrect_k=1.0)

# SECAM-M allegedly used to be broadcast in Cambodia and Vietnam. Assuming it was based on SECAM III.
SecamVariant.SECAM_M = SecamVariant(fsc_dr=227.5 * 15750.0 * 1000.0 / 1001.0,
                                    fsc_db=227.5 * 15750.0 * 1000.0 / 1001.0,
                                    fdev_dr=230000.0,
                                    fdev_db=230000.0,
                                    flimit_minbell=-500000.0,
                                    flimit_maxbell=500000.0,
                                    m0=0.1,
                                    bell_f0=227.5 * 15750.0 * 1000.0 / 1001.0,
                                    bell_kn=16.0,
                                    bell_kd=1.26,
                                    lf_precorrect_f1=70000.0,
                                    lf_precorrect_k=5.6)

# CCIR documents mention SECAM-N alongside SECAM-M. Most likely it was similar to SECAM-M.
SecamVariant.SECAM_N = SecamVariant(fsc_dr=3578125.0,
                                    fsc_db=3578125,
                                    fdev_dr=230000.0,
                                    fdev_db=230000.0,
                                    flimit_minbell=-500000.0,
                                    flimit_maxbell=500000.0,
                                    m0=0.1,
                                    bell_f0=3578125.0,
                                    bell_kn=16.0,
                                    bell_kd=1.26,
                                    lf_precorrect_f1=70000.0,
                                    lf_precorrect_k=5.6)


class FmDecoder(object):
    def __init__(self, fc, dev, resample_rate=2):
        self._fc = fc
        self._resample_rate = resample_rate
        self._lowpass = utils.iirfilter(6, (2.0 * fc - dev) / resample_rate, rs=48.0, btype='lowpass',
                                        ftype='cheby2')

    def __call__(self, data):
        # Assuming the signal is already filtered
        data_up = scipy.signal.resample_poly(data, up=self._resample_rate, down=1)
        phase = numpy.linspace(start=0.0, stop=(len(data_up) * numpy.pi * self._fc) / self._resample_rate,
                               num=len(data_up), endpoint=False)
        cosine_up = data_up * numpy.cos(phase)
        sine_up = data_up * numpy.sin(phase)
        cosine_up = self._lowpass(cosine_up)
        sine_up = self._lowpass(sine_up)
        data_up = cosine_up - 1.0j * sine_up
        # now we have analytic FM at baseband
        phases_up = numpy.angle(data_up)
        phases_up = numpy.unwrap(phases_up)
        phase_shift_up = numpy.diff(numpy.concatenate((phases_up[0:1], phases_up)))
        frequencies_up = self._fc + self._resample_rate * phase_shift_up / numpy.pi
        return scipy.signal.resample_poly(frequencies_up, up=1, down=self._resample_rate)


class SecamModem(object):
    def __init__(self, line_config, variant=SecamVariant.SECAM, alternate_phases=False):
        self._line_config = line_config
        self._variant = variant
        self._fsc_dr = 2.0 * variant.fsc_dr / line_config.fs
        self._fsc_db = 2.0 * variant.fsc_db / line_config.fs
        self._fdev_dr = 2.0 * variant.fdev_dr / line_config.fs
        self._fdev_db = 2.0 * variant.fdev_db / line_config.fs
        self._flimit_min = 2.0 * (variant.bell_f0 + variant.flimit_minbell) / line_config.fs
        self._flimit_max = 2.0 * (variant.bell_f0 + variant.flimit_maxbell) / line_config.fs
        self._bell_f0 = 2.0 * variant.bell_f0 / line_config.fs
        if not alternate_phases:
            self._start_phase_inversions = [False, False, True, False, False, True]
        else:
            self._start_phase_inversions = [False, False, False, True, True, True]
        self._chroma_demod_bell = None
        if variant.bell_kn != variant.bell_kd:
            self._chroma_demod_bell = self._chroma_demod_bell_design(self._bell_f0, self._flimit_max,
                                                                     variant.bell_kn, variant.bell_kd)
        self._chroma_precorrect_lowpass = utils.iirdesign(wp=2.0 * 1300000.0 / line_config.fs,
                                                          ws=2.0 * 3500000.0 / line_config.fs, gpass=3.0, gstop=30.0)
        self._chroma_precorrect = None
        self._reverse_chroma_precorrect = None
        if variant.lf_precorrect_k != 1.0:
            self._chroma_precorrect, self._reverse_chroma_precorrect = SecamModem._chroma_precorrect_design(
                2.0 * variant.lf_precorrect_f1 / line_config.fs, variant.lf_precorrect_k)

        center = 0.5 * (self._flimit_min + self._flimit_max)
        dev = 0.5 * (self._flimit_max - self._flimit_min)

        self._chroma_demod_filter_order = 3
        self._chroma_demod_chroma_filter = utils.iirfilter(3, [center - dev, center + dev],
                                                           rp=0.1, btype='bandpass', ftype='cheby1')
        self._chroma_demod_luma_filter = utils.iirfilter(3, [center - dev * numpy.e, center + dev * numpy.e],
                                                         btype='bandstop', ftype='bessel')
        self._chroma_demod = FmDecoder(center, dev)
        self._last_frame = -1
        self._last_line = -1
        self._last_chroma = None

    @staticmethod
    def encode_components(r, g, b):
        assert len(r) == len(g) == len(b)
        luma = 0.299 * r + 0.587 * g + 0.114 * b
        dr = -1.333302 * r + 1.116474 * g + 0.216828 * b
        db = -0.449995 * r - 0.883435 * g + 1.33343 * b
        # dr in [-1.333302, -1.333302]
        # db in [-1.33343, 1.33343]
        return luma, dr, db

    @staticmethod
    def decode_components(luma, dr, db):
        assert len(luma) == len(dr) == len(db)
        r = luma - 0.5257623554153522 * dr
        g = luma + 0.2678074007993021 * dr - 0.1290417517983779 * db
        b = luma + 0.6644518272425249 * db
        return r, g, b

    @staticmethod
    def _chroma_precorrect_design(wc, k):
        assert k != 1.0
        forward_b, forward_a = scipy.signal.iirfilter(1, k * wc, btype='highpass', ftype='butter')
        assert forward_a[0] == 1.0
        forward_b[0] = (k - 1.0) * forward_b[0] + 1.0
        forward_b[1] = (k - 1.0) * forward_b[1] + forward_a[1]
        backward_b = numpy.array([1.0, forward_a[1]]) / forward_b[0]
        backward_a = numpy.array([1.0, forward_b[1] / forward_b[0]])
        forward = utils.FilterFunction(forward_b, forward_a, k * wc, btype='highpass', shift=False)
        backward = utils.FilterFunction(backward_b, backward_a, k * wc, btype='lowpass', shift=False)
        return forward, backward

    @staticmethod
    def _chroma_demod_bell_design(f0, f_max, kn, kd):
        def gain(f):
            return numpy.sqrt(
                (kd * kd * f0 * f0 * f0 * f0 + (1 - 2 * kd * kd) * f * f * f0 * f0 + kd * kd * f * f * f * f) / (
                        kn * kn * f0 * f0 * f0 * f0 + (1 - 2 * kn * kn) * f * f * f0 * f0 + kn * kn * f * f * f * f))

        def gain_db(f):
            return 10.0 * numpy.log10(gain(f))

        assert kn != kd
        wp2 = f0 + 1 / 256.0
        wp1 = f0 * f0 / wp2
        ws2 = f_max
        ws1 = f0 * f0 / ws2
        return utils.iirdesign([wp1, wp2], [ws1, ws2], -gain_db(wp2), -gain_db(ws2), shift=False)

    def _modulate_chroma(self, start_phase, frequencies):
        bigF = frequencies / self._bell_f0 - self._bell_f0 / frequencies
        bigG = self._variant.m0 * (1.0 + 1.0j * self._variant.bell_kn * bigF) / (
                1.0 + 1.0j * self._variant.bell_kd * bigF)
        phase_shift = numpy.pi * frequencies
        phase = (start_phase - phase_shift[0] - numpy.angle(bigG[0]) + numpy.cumsum(phase_shift)) % (2.0 * numpy.pi)
        return numpy.real(bigG) * numpy.cos(phase) - numpy.imag(bigG) * numpy.sin(phase)

    def _start_phase_inverted(self, frame, line):
        assert len(self._start_phase_inversions) == 6
        frame %= 6
        if line % 2 == 0:
            line_in_field = 23 + (line // 2)
        else:
            line_in_field = 336 + (line // 2)
        line_in_sequence = (frame * 625 + line_in_field) % 6
        return self._start_phase_inversions[line_in_sequence] ^ (frame % 2 == 1)

    def modulate(self, frame, line, r, g, b):
        return self.modulate_components(frame, line, *self.encode_components(r, g, b))

    def modulate_components(self, frame, line, luma, dr, db):
        if not self._line_config.is_alternate_line(frame, line):
            dr = self._chroma_precorrect_lowpass(dr)
            if self._chroma_precorrect is not None:
                dr = self._chroma_precorrect(dr)
            chroma_frequencies = self._fsc_dr + self._fdev_dr * dr
        else:
            db = self._chroma_precorrect_lowpass(db)
            if self._chroma_precorrect is not None:
                db = self._chroma_precorrect(db)
            chroma_frequencies = self._fsc_db + self._fdev_db * db
        chroma_frequencies = numpy.minimum(numpy.maximum(chroma_frequencies, self._flimit_min), self._flimit_max)
        start_phase = numpy.pi if self._start_phase_inverted(frame, line) else 0.0
        chroma = self._modulate_chroma(start_phase, chroma_frequencies)

        return luma + chroma

    def demodulate(self, frame, line, composite):
        if frame != self._last_frame or line != self._last_line + 2 or self._last_chroma is None:
            self._last_chroma = numpy.zeros(len(composite))

        luma = self._chroma_demod_luma_filter(composite)
        chroma_rest = numpy.flip(composite[1:len(composite) // 40])
        chroma_comp = numpy.concatenate((chroma_rest, composite))
        chroma = self._chroma_demod_chroma_filter(chroma_comp)
        if self._chroma_demod_bell:
            chroma = self._chroma_demod_bell(chroma)

        frequencies = self._chroma_demod(chroma)[-len(composite):]
        frequencies = numpy.minimum(numpy.maximum(frequencies, self._flimit_min), self._flimit_max)
        if not self._line_config.is_alternate_line(frame, line):
            chroma = (frequencies - self._fsc_dr) / self._fdev_dr
        else:
            chroma = (frequencies - self._fsc_db) / self._fdev_db
        if self._reverse_chroma_precorrect is not None:
            chroma = self._reverse_chroma_precorrect(chroma)
        if not self._line_config.is_alternate_line(frame, line):
            self._last_chroma, dr, db = chroma, chroma, self._last_chroma
        else:
            self._last_chroma, dr, db = chroma, self._last_chroma, chroma

        self._last_frame = frame
        self._last_line = line
        return self.decode_components(luma, dr, db)
