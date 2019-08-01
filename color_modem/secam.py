# -*- coding: utf-8 -*-

import numpy
import scipy.signal

from color_modem import qam


class SecamModem:
    BLANK_LINE = numpy.zeros(720)
    DR_FC = 4406250.0
    DB_FC = 4250000.0
    DR_DEV = 280000.0
    DB_DEV = 230000.0

    def __init__(self, alternate_phases=False):
        if not alternate_phases:
            self._start_phase_inversions = [False, False, True, False, False, True]
        else:
            self._start_phase_inversions = [False, False, False, True, True, True]
        self._chroma_demod_comb = self._feedback_comb(0.8275, 13.5 / 4.286, 3)
        self._chroma_precorrect_lowpass = self._lanczos_lowpass(1, 1.7 / 13.5)
        self._chroma_demod_dr_low_filter = SecamModem._lanczos_lowpass(4, (self.DR_FC - self.DR_DEV) / 13500000.0)
        self._chroma_demod_dr_high_filter = SecamModem._lanczos_lowpass(4, (self.DR_FC + self.DR_DEV) / 13500000.0)
        self._chroma_demod_db_low_filter = SecamModem._lanczos_lowpass(5, (self.DB_FC - self.DB_DEV) / 13500000.0)
        self._chroma_demod_db_high_filter = SecamModem._lanczos_lowpass(5, (self.DB_FC + self.DB_DEV) / 13500000.0)
        self._last_frame = -1
        self._last_line = -1
        self._last_dr = SecamModem.BLANK_LINE
        self._last_db = SecamModem.BLANK_LINE

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
    def _lanczos_lowpass(a, fc_by_fs):
        kernel = SecamModem._lanczos_kernel(a, scale=2.0 * fc_by_fs)
        return lambda data: scipy.signal.convolve(data, kernel, mode='same')

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

        return lambda data: scipy.signal.lfilter(filter_b, filter_a, data)

    def _chroma_precorrect(self, chroma):
        assert len(chroma) == 720

        chroma = self._chroma_precorrect_lowpass(chroma)
        chromahp = SecamModem._highpass(chroma, 3.0 * 85.0 / 13500.0)

        chroma += 2.0 * chromahp
        chroma /= 255.0
        return chroma

    @staticmethod
    def _reverse_chroma_precorrect(chroma):
        return scipy.signal.lfilter([91.46940073902381, -81.76529963048812], [1.0, -0.9619447015351542], chroma)

    def _dr_frequencies(self, dr):
        # maximum possible deviation: (13999671*sqrt(135778))/(25*sqrt(70201))
        # min ~= 3627469.087
        # max ~= 5185050.913
        return self.DR_FC + self.DR_DEV * self._chroma_precorrect(dr)

    @staticmethod
    def _demodulate_dr(frequencies):
        return SecamModem._reverse_chroma_precorrect((numpy.minimum(numpy.maximum(frequencies, 3627469.087),
                                                                    5185050.913) - SecamModem.DR_FC) / SecamModem.DR_DEV)

    def _db_frequencies(self, db):
        # maximum possible deviation: (9200667*sqrt(67889))/(10*sqrt(140402))
        # min ~= 3610217.479
        # max ~= 4889782.521
        return self.DB_FC + self.DB_DEV * self._chroma_precorrect(db)

    @staticmethod
    def _demodulate_db(frequencies):
        return SecamModem._reverse_chroma_precorrect((numpy.minimum(numpy.maximum(frequencies, 3610217.479),
                                                                    4889782.521) - SecamModem.DB_FC) / SecamModem.DB_DEV)

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

    def _calculate_start_phase(self, frame, line):
        assert len(self._start_phase_inversions) == 6
        frame %= 6
        if line % 2 == 0:
            line_in_field = 23 + (line // 2)
        else:
            line_in_field = 336 + (line // 2)
        line_in_sequence = (frame * 625 + line_in_field) % 6
        invert = self._start_phase_inversions[line_in_sequence] ^ (frame % 2 == 1)
        return -0.5 * numpy.pi if invert else 0.5 * numpy.pi

    def modulate(self, frame, line, r, g, b):
        luma, dr, db = SecamModem._encode_secam_components(r, g, b)
        if not SecamModem._is_alternate_line(frame, line):
            chroma_frequencies = self._dr_frequencies(dr)
        else:
            chroma_frequencies = self._db_frequencies(db)
        start_phase = self._calculate_start_phase(frame, line)
        chroma = SecamModem._modulate_chroma(start_phase, chroma_frequencies)

        return luma + chroma

    @staticmethod
    def _decode_fm(data, fc, dev):
        # Assuming the signal is already filtered
        data2x = numpy.fft.irfft(qam.QamColorModem._fft_expand2x(numpy.fft.rfft(data)))
        phase = numpy.linspace(start=0.0, stop=len(data2x) * numpy.pi * fc / 13500000.0, num=len(data2x),
                               endpoint=False)
        cosine2x = data2x * numpy.cos(phase)
        sine2x = data2x * numpy.sin(phase)
        # filter out the high frequency components
        cosine_fft = numpy.fft.rfft(cosine2x)
        sine_fft = numpy.fft.rfft(sine2x)
        cutoff = int(numpy.ceil(fc * (len(cosine_fft) - 1) / 13500000.0))
        cosine_fft[cutoff:] = 0.0
        sine_fft[cutoff:] = 0.0
        cosine_fft = qam.QamColorModem._fft_limit2x(cosine_fft)
        sine_fft = qam.QamColorModem._fft_limit2x(sine_fft)
        data = numpy.fft.irfft(cosine_fft) - 1.0j * numpy.fft.irfft(sine_fft)
        # now we have analytic FM at baseband
        phases = numpy.angle(data)
        phase_shift = numpy.diff(phases, prepend=phases[0])
        phase_shift = (phase_shift + numpy.pi) % (2.0 * numpy.pi) - numpy.pi
        return fc + (0.5 * 13500000.0 / numpy.pi) * phase_shift

    def demodulate(self, frame, line, composite):
        assert len(composite) == 720
        composite = numpy.array(composite, copy=False)

        if frame != self._last_frame or line != self._last_line + 2:
            self._last_dr = SecamModem.BLANK_LINE
            self._last_db = SecamModem.BLANK_LINE

        if not SecamModem._is_alternate_line(frame, line):
            chroma_low_filter = self._chroma_demod_dr_low_filter
            chroma_high_filter = self._chroma_demod_dr_high_filter
            chroma_fc = self.DR_FC
            chroma_dev = self.DR_DEV
            chroma_divider = 0.419
        else:
            chroma_low_filter = self._chroma_demod_db_low_filter
            chroma_high_filter = self._chroma_demod_db_high_filter
            chroma_fc = self.DB_FC
            chroma_dev = self.DB_DEV
            chroma_divider = 0.432

        # TODO: Better luma filtering
        low_luma = chroma_low_filter(composite)
        chroma_and_high_luma = composite - low_luma
        chroma = chroma_high_filter(chroma_and_high_luma)
        luma = composite - chroma / chroma_divider

        chroma = self._chroma_demod_comb(chroma)
        chroma = SecamModem._highpass(chroma, 0.5 * 4.286 / 13.5)

        frequencies = SecamModem._decode_fm(chroma, chroma_fc, chroma_dev)
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
