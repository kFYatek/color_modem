# -*- coding: utf-8 -*-

import numpy

from color_modem import qam


class SecamModem:
    BLANK_LINE = numpy.zeros(720)

    def __init__(self, alternate_phases=False):
        if not alternate_phases:
            self._start_phase_inversions = [False, False, True, False, False, True]
        else:
            self._start_phase_inversions = [False, False, False, True, True, True]
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
        result = numpy.array(data)
        for i in range(len(data)):
            if i > 0:
                result[i] = alpha * (result[i - 1] + data[i] - data[i - 1])
        return result

    @staticmethod
    def _lanczos_kernel(x, a):
        if x == 0.0:
            return 1.0
        elif -a <= x < a:
            pix = numpy.pi * x
            return a * numpy.sin(pix) * numpy.sin(pix / a) / (pix * pix)
        else:
            return 0.0

    @staticmethod
    def _lanczos_access(data, x, a):
        assert (a == int(a) and a > 0)
        a = int(a)
        result = 0.0
        for i in range(int(x) - a + 1, int(x) + a + 1):
            result += data[min(max(i, 0), len(data) - 1)] * SecamModem._lanczos_kernel(x - i, a)
        return result

    @staticmethod
    def _lanczos_lowpass(data, a, fc_by_fs):
        low_bound = int(-a / (2.0 * fc_by_fs))
        high_bound = int(a / (2.0 * fc_by_fs) + 1)
        result = numpy.zeros(len(data))
        for i in range(len(data)):
            val = 0.0
            for j in range(low_bound, high_bound):
                val += 2.0 * fc_by_fs * data[min(max(i + j, 0), len(data) - 1)] * SecamModem._lanczos_kernel(
                    j * 2.0 * fc_by_fs, a)
            result[i] = val
        return result

    @staticmethod
    def _chroma_precorrect(chroma):
        assert len(chroma) == 720

        chroma = SecamModem._lanczos_lowpass(chroma, 1, 1.7 / 13.5)
        chromahp = SecamModem._highpass(chroma, 3.0 * 85.0 / 13500.0)

        chroma += 2.0 * chromahp
        chroma /= 255.0
        return chroma

    @staticmethod
    def _reverse_chroma_precorrect(chroma):
        chroma = numpy.array(chroma)
        chroma *= 255.0
        result = numpy.array(chroma)
        for i in range(1, len(chroma)):
            result[i] = -0.3206482338450514*chroma[i - 1] + 0.3587035323098973*chroma[i] + 0.9619447015351542*result[i - 1]
        return result

    @staticmethod
    def _dr_frequencies(dr):
        corrected = SecamModem._chroma_precorrect(dr)
        for i in range(len(corrected)):
            corrected[i] = 4406250.0 + 280000.0 * corrected[i]
        # maximum possible deviation: (13999671*sqrt(135778))/(25*sqrt(70201))
        # min ~= 3627469.087
        # max ~= 5185050.913
        return corrected

    @staticmethod
    def _demodulate_dr(frequencies):
        result = numpy.zeros(len(frequencies))
        for i in range(len(frequencies)):
            result[i] = (min(max(frequencies[i], 3627469.087), 5185050.913) - 4406250.0) / 280000.0
        return SecamModem._reverse_chroma_precorrect(result)

    @staticmethod
    def _db_frequencies(db):
        corrected = SecamModem._chroma_precorrect(db)
        for i in range(len(corrected)):
            corrected[i] = 4250000.0 + 230000.0 * corrected[i]
        # maximum possible deviation: (9200667*sqrt(67889))/(10*sqrt(140402))
        # min ~= 3610217.479
        # max ~= 4889782.521
        return corrected

    @staticmethod
    def _demodulate_db(frequencies):
        result = numpy.zeros(len(frequencies))
        for i in range(len(frequencies)):
            result[i] = (min(max(frequencies[i], 3610217.479), 4889782.521) - 4250000.0) / 230000.0
        return SecamModem._reverse_chroma_precorrect(result)

    @staticmethod
    def _modulate_chroma(start_phase, frequencies):
        result = numpy.zeros(len(frequencies))
        phase = start_phase
        for i in range(len(frequencies)):
            phase_shift = 2.0 * numpy.pi * frequencies[i] / 13500000.0
            phase += phase_shift
            while phase >= 2.0 * numpy.pi:
                phase -= 2.0 * numpy.pi
            bigF = frequencies[i] / 4286000.0 - 4286000.0 / frequencies[i]
            bigG = 255.0 * 0.115 * (1.0 + 16.0j * bigF) / (1.0 + 1.26j * bigF)
            result[i] = bigG.real * numpy.cos(phase) - bigG.imag * numpy.sin(phase)
        return result

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
            chroma_frequencies = SecamModem._dr_frequencies(dr)
        else:
            chroma_frequencies = SecamModem._db_frequencies(db)
        start_phase = self._calculate_start_phase(frame, line)
        chroma = SecamModem._modulate_chroma(start_phase, chroma_frequencies)

        return luma + chroma

    @staticmethod
    def _decode_fm(data, fc, dev):
        # Assuming the signal is already filtered
        data2x = numpy.fft.irfft(qam.QamColorModem._fft_expand2x(numpy.fft.rfft(data)))
        cosine2x = numpy.array(data2x)
        sine2x = numpy.array(data2x)
        phase = 0.0
        for i in range(len(data2x)):
            cosine2x[i] *= numpy.cos(phase)
            sine2x[i] *= numpy.sin(phase)
            phase += numpy.pi * fc / 13500000.0
        # filter out the high frequency components
        cosine_fft = numpy.fft.rfft(cosine2x)
        sine_fft = numpy.fft.rfft(sine2x)
        for i in range(len(cosine_fft)):
            freq = 13500000.0 * (i / (len(cosine_fft) - 1))
            if freq >= fc:
                cosine_fft[i] = 0.0
                sine_fft[i] = 0.0
        cosine_fft = qam.QamColorModem._fft_limit2x(cosine_fft)
        sine_fft = qam.QamColorModem._fft_limit2x(sine_fft)
        data = numpy.fft.irfft(cosine_fft) - 1.0j * numpy.fft.irfft(sine_fft)
        # now we have analytic FM at baseband
        phases = numpy.angle(data)
        freqs = numpy.zeros(len(phases))
        freqs[0] = fc
        for i in range(1, len(freqs)):
            phase_shift = (phases[i] - phases[i-1]) % (2.0 * numpy.pi)
            if phase_shift >= numpy.pi:
                phase_shift -= 2.0 * numpy.pi
            freqs[i] = fc + (0.5 * 13500000.0 * phase_shift) / numpy.pi
        return freqs

    def demodulate(self, frame, line, composite):
        def feedback_comb(data, alpha, k, lanczos_a):
            output = numpy.zeros(len(data))
            for i in range(len(output)):
                output[i] = data[i] + alpha * SecamModem._lanczos_access(output, i - k, lanczos_a)
            return output

        assert len(composite) == 720
        composite = numpy.array(composite, copy=False)

        if frame != self._last_frame or line != self._last_line + 2:
            self._last_dr = SecamModem.BLANK_LINE
            self._last_db = SecamModem.BLANK_LINE

        if not SecamModem._is_alternate_line(frame, line):
            chroma_fc = 4406250.0
            chroma_dev = 280000.0
            chroma_filter_a = 4
            chroma_divider = 0.419
        else:
            chroma_fc = 4250000.0
            chroma_dev = 230000.0
            chroma_filter_a = 5
            chroma_divider = 0.432

        # TODO: Better luma filtering
        low_luma = SecamModem._lanczos_lowpass(composite, chroma_filter_a, (chroma_fc - chroma_dev) / 13500000.0)
        chroma_and_high_luma = composite - low_luma
        chroma = SecamModem._lanczos_lowpass(chroma_and_high_luma, chroma_filter_a, (chroma_fc + chroma_dev) / 13500000.0)
        luma = composite - chroma / chroma_divider

        chroma = feedback_comb(chroma, 0.8275, 13.5 / 4.286, 3)
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
        clamped = max(min(adjusted, 255.0), 0.0)
        return int(clamped + 0.5)

    @staticmethod
    def decode_composite_level(value):
        return (value * 1166.0 - 59415.0) / 700.0
