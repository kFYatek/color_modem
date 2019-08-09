# -*- coding: utf-8 -*-

import numpy
import scipy.signal

from color_modem import utils


class SecamModem:
    DR_FC = 4406250.0
    DB_FC = 4250000.0
    DR_DEV = 280000.0
    DB_DEV = 230000.0
    MIN_FREQ = 3900000.0
    MAX_FREQ = 4756250.0

    def __init__(self, alternate_phases=False):
        if not alternate_phases:
            self._start_phase_inversions = [False, False, True, False, False, True]
        else:
            self._start_phase_inversions = [False, False, False, True, True, True]
        self._chroma_demod_comb = self._feedback_comb(0.8275, 13500000.0 / 4286000.0, 3)

        self._chroma_precorrect_lowpass = utils.iirdesign(wp=2.0 * 1300000.0 / 13500000.0,
                                                          ws=2.0 * 3500000.0 / 13500000.0, gpass=3.0, gstop=30.0)
        self._chroma_precorrect, self._reverse_chroma_precorrect = SecamModem._chroma_precorrect_design(
            3.0 * 85000.0 / 13500000.0)

        center = (SecamModem.MIN_FREQ + SecamModem.MAX_FREQ) / 13500000.0
        dev = (SecamModem.MAX_FREQ - SecamModem.MIN_FREQ) / 13500000.0

        self._chroma_demod_chroma_filter = utils.iirfilter(3, [center - dev, center + dev],
                                                           rp=0.1, btype='bandpass', ftype='cheby1')
        self._chroma_demod_luma_filter = utils.iirfilter(3, [center - dev * numpy.e, center + dev * numpy.e],
                                                         btype='bandstop', ftype='bessel')
        self._chroma_demod_dr = SecamModem._fm_decoder(self.DR_FC, self.DR_DEV)
        self._chroma_demod_db = SecamModem._fm_decoder(self.DB_FC, self.DB_DEV)
        self._last_frame = -1
        self._last_line = -1
        self._last_dr = None
        self._last_db = None

    @staticmethod
    def _encode_secam_components(r, g, b):
        assert len(r) == len(g) == len(b)
        r = numpy.array(r, copy=False)
        g = numpy.array(g, copy=False)
        b = numpy.array(b, copy=False)
        luma = 0.299 * r + 0.587 * g + 0.114 * b
        dr = -1.333302 * r + 1.116474 * g + 0.216828 * b
        db = -0.449995 * r - 0.883435 * g + 1.33343 * b
        # dr in [-1.333302, -1.333302]
        # db in [-1.33343, 1.33343]
        return luma, dr, db

    @staticmethod
    def _decode_secam_components(luma, dr, db):
        assert len(luma) == len(dr) == len(db)
        luma = numpy.array(luma, copy=False)
        dr = numpy.array(dr, copy=False)
        db = numpy.array(db, copy=False)
        r = luma - 0.5257623554153522 * dr
        g = luma + 0.2678074007993021 * dr - 0.1290417517983779 * db
        b = luma + 0.6644518272425249 * db
        return r, g, b

    @staticmethod
    def _highpass(data, fc_by_fs):
        alpha = 1.0 / (2.0 * numpy.pi * fc_by_fs + 1.0)
        return scipy.signal.lfilter([alpha, -alpha], [1.0, -alpha], data)

    @staticmethod
    def _chroma_precorrect_design(fc_by_fs):
        alpha = 1.0 / (2.0 * numpy.pi * fc_by_fs + 1.0)
        forward_b = numpy.array([(2.0 * alpha + 1.0) / 255.0, (-3.0 * alpha) / 255.0])
        forward_a = numpy.array([1.0, -alpha])
        backward_b = numpy.array([1.0, forward_a[1]]) / forward_b[0]
        backward_a = numpy.array([1.0, forward_b[1] / forward_b[0]])
        forward = lambda x: scipy.signal.lfilter(forward_b, forward_a, x)
        backward = lambda x: scipy.signal.lfilter(backward_b, backward_a, x)
        return forward, backward

    @staticmethod
    def _lanczos_kernel(a, offset=0.0, scale=1.0):
        assert (a == int(a) and a > 0)
        a = int(a)
        scaled_a = int(a / scale)
        x = (numpy.arange(2 * scaled_a + 1) - offset - scaled_a) * scale
        pix = numpy.pi * x

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = scale * a * numpy.sin(pix) * numpy.sin(pix / a) / (pix * pix)

        if offset == 0.0:
            result[scaled_a] = scale
        return result

    @staticmethod
    def _feedback_comb(alpha, k, lanczos_a):
        k_frac = -k % 1.0
        k_int = int(numpy.floor(k))
        kernel = alpha * SecamModem._lanczos_kernel(lanczos_a, k_frac)[1:]

        shift = max(len(kernel) - k_int - lanczos_a, 0)
        filter_b = numpy.array(shift * [0.0] + [1.0])
        filter_a = numpy.zeros(shift + lanczos_a + k_int + 1)
        filter_a[0] = 1.0
        for i in range(len(kernel)):
            filter_a[shift + lanczos_a + k_int - i] = -kernel[i]

        # These numbers don't seem to make much sense, but give good results
        frequencies = -0.5 * (SecamModem.MIN_FREQ + SecamModem.MAX_FREQ) * numpy.ones(len(filter_a) - 1)
        phase_shift = 2.0 * numpy.pi * frequencies / 13500000.0
        phase = numpy.cumsum(phase_shift)
        phase = phase - phase[0]
        initial_data = -175.0 * numpy.cos(phase)
        calculated_zi = scipy.signal.lfiltic(filter_b, filter_a, initial_data)

        def filter(data, start_phase_inverted):
            if start_phase_inverted:
                zi = -calculated_zi
            else:
                zi = calculated_zi
            if shift:
                return scipy.signal.lfilter(filter_b, filter_a,
                                            numpy.concatenate((data, numpy.zeros(shift))), zi=zi)[0][shift:]
            else:
                return scipy.signal.lfilter(filter_b, filter_a, data, zi=zi)[0]

        return filter

    def _demodulate_dr(self, frequencies):
        min_freq = SecamModem.DR_FC - (SecamModem.MAX_FREQ - SecamModem.DR_FC)
        return self._reverse_chroma_precorrect(
            (numpy.minimum(numpy.maximum(frequencies, min_freq),
                           SecamModem.MAX_FREQ) - SecamModem.DR_FC) / SecamModem.DR_DEV)

    def _demodulate_db(self, frequencies):
        max_freq = SecamModem.DB_FC + (SecamModem.DB_FC - SecamModem.MIN_FREQ)
        return self._reverse_chroma_precorrect((numpy.minimum(numpy.maximum(frequencies, SecamModem.MIN_FREQ),
                                                              max_freq) - SecamModem.DB_FC) / SecamModem.DB_DEV)

    @staticmethod
    def _modulate_chroma(start_phase, frequencies):
        phase_shift = 2.0 * numpy.pi * frequencies / 13500000.0
        phase = (start_phase + numpy.cumsum(phase_shift)) % (2.0 * numpy.pi)
        bigF = frequencies / 4286000.0 - 4286000.0 / frequencies
        bigG = 255.0 * 0.115 * (1.0 + 16.0j * bigF) / (1.0 + 1.26j * bigF)
        return numpy.real(bigG) * numpy.cos(phase) - numpy.imag(bigG) * numpy.sin(phase)

    @staticmethod
    def _is_alternate_line(frame, line):
        # For 525-line system:
        #
        # For 625-line system:
        #
        # digital line 0 ~ analog line 23 (even fields 1 & 3)
        # digital line 1 ~ analog line 336 (odd fields 2 & 4)
        # digital line 2 - analog line 24 (even fields 1 & 3)
        # digital line 3 - analog line 337 (odd fields 2 & 4)
        return ((line % 2) ^ ((line // 2) % 2)) != frame % 2

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
        luma, dr, db = SecamModem._encode_secam_components(r, g, b)
        if not SecamModem._is_alternate_line(frame, line):
            chroma_frequencies = self.DR_FC + self.DR_DEV * self._chroma_precorrect(self._chroma_precorrect_lowpass(dr))
        else:
            chroma_frequencies = self.DB_FC + self.DB_DEV * self._chroma_precorrect(self._chroma_precorrect_lowpass(db))
        chroma_frequencies = numpy.minimum(numpy.maximum(chroma_frequencies, self.MIN_FREQ), self.MAX_FREQ)
        start_phase = numpy.pi if self._start_phase_inverted(frame, line) else 0.0
        chroma = SecamModem._modulate_chroma(start_phase, chroma_frequencies)

        return luma + chroma

    @staticmethod
    def _fm_decoder(fc, dev):
        lowpass = utils.iirfilter(6, (2.0 * fc - dev) / 13500000.0, rs=48.0, btype='lowpass', ftype='cheby2')

        def decode(data):
            # Assuming the signal is already filtered
            data2x = scipy.signal.resample_poly(data, up=2, down=1)
            phase = numpy.linspace(start=0.0, stop=len(data2x) * numpy.pi * fc / 13500000.0, num=len(data2x),
                                   endpoint=False)
            cosine2x = lowpass(data2x * numpy.cos(phase))
            sine2x = lowpass(data2x * numpy.sin(phase))
            data = scipy.signal.resample_poly(cosine2x, up=1, down=2) - 1.0j * scipy.signal.resample_poly(sine2x, up=1,
                                                                                                          down=2)
            # now we have analytic FM at baseband
            phases = numpy.angle(data)
            phase_shift = numpy.diff(numpy.concatenate((phases[0:1], phases)))
            phase_shift = (phase_shift + numpy.pi) % (2.0 * numpy.pi) - numpy.pi
            return fc + (0.5 * 13500000.0 / numpy.pi) * phase_shift

        return decode

    def demodulate(self, frame, line, composite):
        composite = numpy.array(composite, copy=False)

        if frame != self._last_frame or line != self._last_line + 2 or self._last_dr is None or self._last_db is None:
            self._last_dr = numpy.zeros(len(composite))
            self._last_db = numpy.zeros(len(composite))

        if not SecamModem._is_alternate_line(frame, line):
            chroma_demod = self._chroma_demod_dr
        else:
            chroma_demod = self._chroma_demod_db

        luma = self._chroma_demod_luma_filter(composite)
        chroma = self._chroma_demod_chroma_filter(composite)
        chroma = self._chroma_demod_comb(chroma, self._start_phase_inverted(frame, line))
        chroma = SecamModem._highpass(chroma, 0.5 * 4286000.0 / 13500000.0)

        frequencies = chroma_demod(chroma)
        if not SecamModem._is_alternate_line(frame, line):
            dr = self._demodulate_dr(frequencies)
            self._last_dr = dr
            db = self._last_db
        else:
            dr = self._last_dr
            db = self._demodulate_db(frequencies)
            self._last_db = db

        self._last_frame = frame
        self._last_line = line
        return self._decode_secam_components(luma, dr, db)

    @staticmethod
    def encode_composite_level(value):
        # max excursion: 933/700
        # white level: 1
        # black level: 0
        # min excursion: -233/700
        adjusted = (value * 700.0 + 59415.0) / 1166.0
        clamped = numpy.maximum(numpy.minimum(adjusted, 255.0), 0.0)
        return numpy.uint8(numpy.rint(clamped))

    @staticmethod
    def decode_composite_level(value):
        return (value * 1166.0 - 59415.0) / 700.0
