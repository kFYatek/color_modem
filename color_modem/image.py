# -*- coding: utf-8 -*-

import numpy
from PIL import Image


def _as_bytes(array):
    return numpy.uint8(numpy.rint(255.0 * numpy.maximum(numpy.minimum(array, 1.0), 0.0)))


class ImageModem(object):
    def __init__(self, modem):
        self._modem = modem

    @staticmethod
    def encode_composite_level(value):
        # max excursion: 933/700
        # white level: 1
        # black level: 0
        # min excursion: -233/700
        return 0.6 * value + 0.2

    @staticmethod
    def decode_composite_level(value):
        return (5.0 * value - 1.0) / 3.0

    def modulate(self, img, frame=0):
        if img.mode != 'RGB':
            img = img.convert('RGB')
        modulation_delay = getattr(self._modem, 'modulation_delay', 0)
        r = bytearray(img.getdata(0))
        g = bytearray(img.getdata(1))
        b = bytearray(img.getdata(2))

        def get_lines(y):
            return numpy.array(r[img.width * y:img.width * (y + 1)]) / 255.0, \
                   numpy.array(g[img.width * y:img.width * (y + 1)]) / 255.0, \
                   numpy.array(b[img.width * y:img.width * (y + 1)]) / 255.0

        def output(y, bytes):
            if output.data is None:
                output.data = numpy.zeros(len(bytes) * img.height, dtype=numpy.uint8)
            output.data[len(bytes) * y:len(bytes) * (y + 1)] = bytes

        output.data = None

        for field in range(2):
            for y in range(field, 2 * modulation_delay, 2):
                self._modem.modulate(frame, y, *get_lines(y))
            for y in range(field, img.height, 2):
                input_y = y + 2 * modulation_delay
                while input_y >= img.height:
                    input_y -= 2
                output(y, _as_bytes(self.encode_composite_level(
                    self._modem.modulate(frame, y + 2 * modulation_delay, *get_lines(input_y)))))
        return Image.frombytes('L', (len(output.data) // img.height, img.height), bytes(bytearray(output.data)))

    def demodulate(self, img, frame=0):

        if img.mode != 'L':
            img = img.convert('L')
        composite = self.decode_composite_level(numpy.array(img.getdata()) / 255.0)
        demodulation_delay = getattr(self._modem, 'demodulation_delay', 0)

        def output(y, r_bytes, g_bytes, b_bytes):
            assert len(r_bytes) == len(g_bytes) == len(b_bytes)
            if output.data is None:
                output.data = numpy.zeros(len(r_bytes) * img.height * 3, dtype=numpy.uint8)
            output.data[3 * len(r_bytes) * y:3 * len(r_bytes) * (y + 1):3] = r_bytes
            output.data[3 * len(r_bytes) * y + 1:3 * len(r_bytes) * (y + 1) + 1:3] = g_bytes
            output.data[3 * len(r_bytes) * y + 2:3 * len(r_bytes) * (y + 1) + 2:3] = b_bytes

        output.data = None

        for field in range(2):
            for y in range(field, 2 * demodulation_delay, 2):
                self._modem.demodulate(frame, y, composite[img.width * y:img.width * (y + 1)])
            for y in range(field, img.height, 2):
                input_y = y + 2 * demodulation_delay
                while input_y >= img.height:
                    input_y -= 2
                output(y, *map(_as_bytes, self._modem.demodulate(
                    frame, y + 2 * demodulation_delay, composite[img.width * input_y:img.width * (input_y + 1)])))
        return Image.frombytes('RGB', (len(output.data) // (3 * img.height), img.height), bytes(bytearray(output.data)))
