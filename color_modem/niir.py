# -*- coding: utf-8 -*-

import numpy
import scipy.signal

from color_modem import utils


class NiirModem:
    def __init__(self):
        self._carrier_phase_step = numpy.pi * 4433618.75 / 13500000.0

        line_shift_by_pi = (2.0 * 864 * 4433618.75 / 13500000.0) % 2.0
        self.line_shift = numpy.pi * line_shift_by_pi
        odd_numbered_digital_line_shift_by_pi = (line_shift_by_pi * 313) % 2.0
        self._odd_numbered_digital_line_shift = numpy.pi * odd_numbered_digital_line_shift_by_pi
        frame_shift_by_pi = (line_shift_by_pi * 625) % 2.0
        self._frame_shift = numpy.pi * frame_shift_by_pi

        self._chroma_precorrect_lowpass = utils.chroma_precorrect_lowpass(2.0 * 1300000.0 / 13500000.0,
                                                                          2.0 * 4000000.0 / 13500000.0, 3.0, 20.0)
        self._demodulate_resample_factor = 8
        self._demodulate_upsampled_baseband_filter, self._demodulate_upsampled_filter = NiirModem._demodulate_am_design(
            2.0 * 4433618.75 / 13500000.0, 2.0 * 1300000.0 / 13500000.0, 2.0 * 4000000.0 / 13500000.0, 3.0, 20.0,
            self._demodulate_resample_factor)

        def _carrier_notch():
            b, a = scipy.signal.iirfilter(2, [4433618.75 - 9375.0, 4433618.75 + 9375.0], ftype='bessel',
                                          fs=13500000.0 * self._demodulate_resample_factor)
            return lambda x: scipy.signal.lfilter(b, a, x)

        self._carrier_up_notch = _carrier_notch()

        self._ep_line_up = 25.5 * numpy.ones(self._demodulate_resample_factor * 720)

        self._last_frame = -1
        self._last_line = -1
        self._last_modulated_up = None
        self._last_saturation_plus_ep_up = None

    @staticmethod
    def _encode_niir_components(r, g, b):
        assert len(r) == len(g) == len(b)
        r = numpy.array(r, copy=False)
        g = numpy.array(g, copy=False)
        b = numpy.array(b, copy=False)
        luma = 0.299 * r + 0.587 * g + 0.114 * b
        dr = 0.6149122807017545 * r - 0.5149122807017544 * g - 0.1 * b
        db = -0.1472906403940887 * r - 0.2891625615763547 * g + 0.4364532019704434 * b
        return luma, dr, db

    @staticmethod
    def _decode_niir_components(luma, dr, db):
        assert len(luma) == len(dr) == len(db)
        luma = numpy.array(luma, copy=False)
        dr = numpy.array(dr, copy=False)
        db = numpy.array(db, copy=False)
        r = luma + 1.14 * dr
        g = luma - 0.5806814310051107 * dr - 0.3942419080068143 * db
        b = luma + 2.03 * db
        return r, g, b

    def _calculate_start_phase(self, frame, line):
        frame %= 4
        frame_shift = (frame * self._frame_shift) % (2.0 * numpy.pi)
        field_shift = (line % 2) * self._odd_numbered_digital_line_shift
        line_shift = ((line // 2) * self.line_shift) % (2.0 * numpy.pi)
        return (frame_shift + field_shift + line_shift) % (2.0 * numpy.pi)

    def _is_alternate_line(self, frame, line):
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
        return ((line % 2) ^ ((line // 2) % 2)) != frame % 2

    def _modulate_precorrected_chroma(self, frame, line, updr, updb):
        start_phase = self._calculate_start_phase(frame, line)
        phase = numpy.linspace(start=start_phase, stop=start_phase + len(updb) * 2.0 * self._carrier_phase_step,
                               num=len(updb), endpoint=False) % (2.0 * numpy.pi)
        if not self._is_alternate_line(frame, line):
            return updr * numpy.cos(phase) - updb * numpy.sin(phase)
        else:
            return -numpy.sqrt(updr * updr + updb * updb) * numpy.sin(phase)

    def _modulate_chroma(self, frame, line, dr, db):
        assert len(db) == len(dr)
        saturation = numpy.sqrt(dr * dr + db * db) + 25.5
        hue = numpy.arctan2(db + (numpy.random.random_sample(len(db)) - 0.5) / 65536.0,
                            dr + (numpy.random.random_sample(len(dr)) - 0.5) / 65536.0)
        drep = saturation * numpy.cos(hue)
        dbep = saturation * numpy.sin(hue)
        drep = self._chroma_precorrect_lowpass(drep)
        dbep = self._chroma_precorrect_lowpass(dbep)
        return self._modulate_precorrected_chroma(frame, line, drep, dbep)

    def modulate(self, frame, line, r, g, b):
        # TODO: Take the next line's hue into consideration when encoding
        luma, dr, db = NiirModem._encode_niir_components(r, g, b)
        chroma = self._modulate_chroma(frame, line, dr, db)
        return luma + chroma

    @staticmethod
    def _demodulate_am_design(wc, wp, ws, gpass, gstop, resample_factor):
        def _baseband_filter(wp, ws, gpass, gstop):
            b, a = scipy.signal.iirdesign(wp, ws, gpass, gstop, ftype='butter')
            shift = int(numpy.round(scipy.signal.group_delay((b, a), [0.0])[1]))
            return lambda x: scipy.signal.lfilter(b, a, numpy.concatenate((x, numpy.zeros(shift))))[shift:]

        def _modulated_filter(wc, wp, ws, gpass, gstop):
            b, a = scipy.signal.iirdesign([wc - wp, wc + wp], [wc - ws, wc + ws], gpass, gstop, ftype='butter')
            shift = int(numpy.round(scipy.signal.group_delay((b, a), [wc], fs=2.0)[1]))
            return lambda x: scipy.signal.lfilter(b, a, numpy.concatenate((x, numpy.zeros(shift))))[shift:]

        return _baseband_filter(wp / resample_factor, ws / resample_factor, 0.5 * gpass,
                                0.5 * gstop), _modulated_filter(wc / resample_factor, wp / resample_factor,
                                                                ws / resample_factor, 0.5 * gpass, 0.5 * gstop)

    def demodulate(self, frame, line, composite):
        if frame != self._last_frame or line != self._last_line + 2 \
                or self._last_modulated_up is None or self._last_saturation_plus_ep_up is None:
            last_modulated = self._modulate_precorrected_chroma(frame, line - 2, 0.0, 25.5 * numpy.ones(len(composite)))
            self._last_modulated_up = self._demodulate_upsampled_filter(
                scipy.signal.resample_poly(last_modulated, up=self._demodulate_resample_factor, down=1))
            self._last_saturation_plus_ep_up = self._ep_line_up

        upsampled = scipy.signal.resample_poly(composite, up=self._demodulate_resample_factor, down=1)
        modulated_up = self._demodulate_upsampled_filter(upsampled)
        demod_up = 0.5 * numpy.pi * numpy.abs(modulated_up)
        saturation_plus_ep_up = self._demodulate_upsampled_baseband_filter(demod_up)

        if not self._is_alternate_line(frame, line):
            carrier_up = self._carrier_up_notch(self._last_modulated_up)
            huemod_up = modulated_up
            huemod_amplitude = saturation_plus_ep_up
        else:
            carrier_up = self._carrier_up_notch(modulated_up)
            huemod_up = self._last_modulated_up
            huemod_amplitude = self._last_saturation_plus_ep_up

        if not self._is_alternate_line(frame, line):
            carrier_up = -carrier_up

        shifted_carrier_up = carrier_up[0:-1] + carrier_up[1:]
        altcarrier_up = numpy.concatenate((numpy.zeros(1), numpy.diff(shifted_carrier_up), numpy.zeros(1)))

        carrier_up /= numpy.max(numpy.abs(carrier_up))
        cosphi_up = huemod_up * carrier_up / huemod_amplitude
        cosphi_up = numpy.minimum(numpy.maximum(cosphi_up, -1.0), 1.0)

        altcarrier_up /= numpy.max(numpy.abs(altcarrier_up))
        sinphi_up = huemod_up * altcarrier_up / huemod_amplitude
        sinphi_up = numpy.minimum(numpy.maximum(sinphi_up, -1.0), 1.0)

        cosphi = scipy.signal.resample_poly(cosphi_up, up=1, down=self._demodulate_resample_factor)
        sinphi = scipy.signal.resample_poly(sinphi_up, up=1, down=self._demodulate_resample_factor)
        phi = numpy.arctan2(sinphi, cosphi)

        saturation_plus_ep = scipy.signal.resample_poly(saturation_plus_ep_up, up=1,
                                                        down=self._demodulate_resample_factor)
        saturation = numpy.maximum(saturation_plus_ep - 25.5, 0.0)
        dr = saturation * numpy.cos(phi)
        db = saturation * numpy.sin(phi)
        updr = saturation_plus_ep * numpy.cos(phi)
        updb = saturation_plus_ep * numpy.sin(phi)
        luma = composite - self._modulate_precorrected_chroma(frame, line, updr, updb)

        self._last_modulated_up = modulated_up
        self._last_saturation_plus_ep_up = saturation_plus_ep_up
        self._last_frame = frame
        self._last_line = line
        return self._decode_niir_components(luma, dr, db)

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
