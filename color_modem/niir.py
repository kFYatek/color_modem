# -*- coding: utf-8 -*-

import numpy
import scipy.signal

from color_modem import utils, pal


class NiirModem(object):
    def __init__(self, config=pal.PalVariant.PAL, fs=13500000.0, noise_level=0.0):
        self._carrier_phase_step = 2.0 * numpy.pi * config.fsc / fs
        self._noise_level = noise_level
        self.config = config
        self.fs = fs

        self._chroma_precorrect_lowpass = utils.iirdesign(2.0 * config.bandwidth3db / fs,
                                                          2.0 * config.bandwidth20db / fs, 3.0, 20.0)
        self._demodulate_resample_factor = 3
        self._demodulate_upsampled_baseband_filter, self._demodulate_upsampled_filter = NiirModem._demodulate_am_design(
            2.0 * config.fsc / fs, 2.0 * config.bandwidth3db / fs, 2.0 * config.bandwidth20db / fs, 3.0, 20.0,
            self._demodulate_resample_factor)

        self._last_frame = -1
        self._last_line = -1
        self._last_phasemod_up = None

    @staticmethod
    def _encode_niir_components(r, g, b):
        assert len(r) == len(g) == len(b)
        r = numpy.array(r, copy=False)
        g = numpy.array(g, copy=False)
        b = numpy.array(b, copy=False)
        luma = 0.299 * r + 0.587 * g + 0.114 * b
        db = 0.1472906403940887 * r + 0.2891625615763547 * g - 0.4364532019704434 * b
        dr = 0.6149122807017545 * r - 0.5149122807017544 * g - 0.1 * b
        return luma, db, dr

    def encode_yuv(self, r, g, b):
        luma, db, dr = NiirModem._encode_niir_components(r, g, b)
        saturation = numpy.sqrt(db * db + dr * dr) + 0.1
        if self._noise_level != 0.0:
            db += (numpy.random.random_sample(len(db)) - 0.5) * self._noise_level
            dr += (numpy.random.random_sample(len(dr)) - 0.5) * self._noise_level
        hue = numpy.arctan2(db, dr)
        return luma, saturation * numpy.sin(hue), saturation * numpy.cos(hue)

    @staticmethod
    def _decode_niir_components(luma, db, dr):
        assert len(luma) == len(db) == len(dr)
        luma = numpy.array(luma, copy=False)
        db = numpy.array(db, copy=False)
        dr = numpy.array(dr, copy=False)
        r = luma + 1.14 * dr
        g = luma + 0.3942419080068143 * db - 0.5806814310051107 * dr
        b = luma - 2.03 * db
        return r, g, b

    @staticmethod
    def decode_yuv(luma, db, dr):
        saturation = numpy.maximum(numpy.sqrt(db * db + dr * dr) - 0.1, 0.0)
        hue = numpy.arctan2(db, dr)
        return NiirModem._decode_niir_components(luma, saturation * numpy.sin(hue), saturation * numpy.cos(hue))

    def _modulate_precorrected_chroma(self, frame, line, updb, updr):
        start_phase = self.config.start_phase(frame, line)
        phase = numpy.linspace(start=start_phase, stop=start_phase + len(updb) * self._carrier_phase_step,
                               num=len(updb), endpoint=False) % (2.0 * numpy.pi)
        if not self.config.is_alternate_line(frame, line):
            return updb * numpy.sin(phase) + updr * numpy.cos(phase)
        else:
            return -numpy.sqrt(updb * updb + updr * updr) * numpy.sin(phase)

    def modulate(self, frame, line, r, g, b):
        return self.modulate_yuv(frame, line, *self.encode_yuv(r, g, b))

    def modulate_yuv(self, frame, line, luma, db, dr):
        # TODO: Take the next line's hue into consideration when encoding
        db = self._chroma_precorrect_lowpass(db)
        dr = self._chroma_precorrect_lowpass(dr)
        chroma = self._modulate_precorrected_chroma(frame, line, db, dr)
        return luma + chroma

    @staticmethod
    def _demodulate_am_design(wc, wp, ws, gpass, gstop, resample_factor):
        return utils.iirdesign(wp / resample_factor, ws / resample_factor, gpass, gstop), utils.iirdesign_wc(
            wc / resample_factor, wp / resample_factor, ws / resample_factor, gpass, gstop)

    def demodulate(self, frame, line, *args, **kwargs):
        return self.decode_yuv(*self.demodulate_yuv(frame, line, *args, **kwargs))

    def demodulate_yuv(self, frame, line, composite, strip_chroma=True):
        if frame != self._last_frame or line != self._last_line + 2 or self._last_phasemod_up is None:
            last_modulated = self._modulate_precorrected_chroma(frame, line - 2, numpy.ones(len(composite)), 0.0)
            self._last_phasemod_up = self._demodulate_upsampled_filter(
                scipy.signal.resample_poly(last_modulated, up=self._demodulate_resample_factor, down=1))

        upsampled = scipy.signal.resample_poly(composite, up=self._demodulate_resample_factor, down=1)
        modulated_up = self._demodulate_upsampled_filter(upsampled)
        demod_up = 0.5 * numpy.pi * numpy.abs(modulated_up)
        saturation_up = self._demodulate_upsampled_baseband_filter(demod_up)
        phasemod_up = modulated_up / saturation_up

        if not self.config.is_alternate_line(frame, line):
            carrier_up = self._last_phasemod_up
            huemod_up = phasemod_up
            shift = self.config.line_shift
        else:
            carrier_up = phasemod_up
            huemod_up = self._last_phasemod_up
            shift = -self.config.line_shift

        shifted_carrier_up = 0.5 * (carrier_up[0:-1] + carrier_up[1:])
        altcarrier_up = numpy.concatenate((numpy.zeros(1), numpy.diff(shifted_carrier_up), numpy.zeros(
            1))) * self._demodulate_resample_factor / self._carrier_phase_step

        sinphi_up = huemod_up * carrier_up
        cosphi_up = huemod_up * altcarrier_up

        sinphi = scipy.signal.resample_poly(sinphi_up, up=1, down=self._demodulate_resample_factor)
        cosphi = scipy.signal.resample_poly(cosphi_up, up=1, down=self._demodulate_resample_factor)
        normalizer = numpy.sqrt(cosphi * cosphi + sinphi * sinphi)
        cosphi /= normalizer
        sinphi /= normalizer

        sinphi, cosphi = -cosphi * numpy.sin(shift) - sinphi * numpy.cos(shift), \
                         sinphi * numpy.sin(shift) - cosphi * numpy.cos(shift)

        saturation = scipy.signal.resample_poly(saturation_up, up=1, down=self._demodulate_resample_factor)
        db = saturation * sinphi
        dr = saturation * cosphi
        luma = composite
        if strip_chroma:
            luma = luma - self._modulate_precorrected_chroma(frame, line, db, dr)

        self._last_phasemod_up = phasemod_up
        self._last_frame = frame
        self._last_line = line
        return luma, db, dr

    @staticmethod
    def encode_composite_level(value):
        # max excursion: 933/700
        # white level: 1
        # black level: 0
        # min excursion: -233/700
        return (value * 700.0 + 233.0) / 1166.0

    @staticmethod
    def decode_composite_level(value):
        return (value * 1166.0 - 233.0) / 700.0


class HueCorrectingNiirModem(NiirModem):
    def __init__(self, *args, **kwargs):
        super(HueCorrectingNiirModem, self).__init__(*args, **kwargs)
        self.modulation_delay = 1
        self._last_modulated_frame = -1
        self._last_modulated_line = -1
        self._last_luma = None
        self._last_db = None
        self._last_dr = None

    def modulate(self, frame, line, r, g, b):
        luma, db, dr = NiirModem._encode_niir_components(r, g, b)
        if frame != self._last_modulated_frame or line != self._last_modulated_line + 2 \
                or self._last_db is None or self._last_dr is None:
            self._last_luma = luma
            self._last_db = db
            self._last_dr = dr
        self._last_luma, luma = luma, self._last_luma
        last_saturation = numpy.sqrt(self._last_db * self._last_db + self._last_dr * self._last_dr)
        saturation = numpy.sqrt(db * db + dr * dr)
        divisor = last_saturation + saturation
        divisor[numpy.equal(divisor, 0.0)] = 1.0
        avgdb = (self._last_db * last_saturation + db * saturation) / divisor
        avgdr = (self._last_dr * last_saturation + dr * saturation) / divisor
        if self._noise_level != 0.0:
            avgdb += (numpy.random.random_sample(len(db)) - 0.5) * self._noise_level
            avgdr += (numpy.random.random_sample(len(dr)) - 0.5) * self._noise_level
        last_saturation_plus_ep = last_saturation + 0.1
        hue = numpy.arctan2(avgdb, avgdr)
        dbep, drep = last_saturation_plus_ep * numpy.sin(hue), last_saturation_plus_ep * numpy.cos(hue)
        self._last_db = db
        self._last_dr = dr
        self._last_modulated_frame = frame
        self._last_modulated_line = line
        return self.modulate_yuv(frame, line - 2, luma, dbep, drep)
