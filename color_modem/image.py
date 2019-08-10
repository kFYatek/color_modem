# -*- coding: utf-8 -*-

import itertools

import numpy
from PIL import Image


def _as_bytes(array):
    return numpy.uint8(numpy.rint(255.0 * numpy.maximum(numpy.minimum(array, 1.0), 0.0)))


class ImageModem(object):
    def __init__(self, modem):
        self._modem = modem

    def modulate(self, img, frame=0):
        if img.mode != 'RGB':
            img = img.convert('RGB')
        r = bytearray(img.getdata(0))
        g = bytearray(img.getdata(1))
        b = bytearray(img.getdata(2))
        output = numpy.zeros(img.width * img.height, dtype=numpy.uint8)
        for y in itertools.chain(range(0, img.height, 2), range(1, img.height, 2)):
            r_line = numpy.array(r[img.width * y:img.width * (y + 1)]) / 255.0
            g_line = numpy.array(g[img.width * y:img.width * (y + 1)]) / 255.0
            b_line = numpy.array(b[img.width * y:img.width * (y + 1)]) / 255.0
            output[img.width * y:img.width * (y + 1)] = _as_bytes(
                self._modem.encode_composite_level(self._modem.modulate(frame, y, r_line, g_line, b_line)))
        return Image.frombytes('L', img.size, bytes(bytearray(output)))

    def demodulate(self, img, frame=0):

        if img.mode != 'L':
            img = img.convert('L')
        composite = self._modem.decode_composite_level(numpy.array(img.getdata()) / 255.0)
        output = numpy.zeros(img.width * img.height * 3, dtype=numpy.uint8)
        demodulation_delay = getattr(self._modem, 'demodulation_delay', 0)
        for field in range(2):
            for y in range(field, 2 * demodulation_delay, 2):
                self._modem.demodulate(frame, y, composite[img.width * y:img.width * (y + 1)])
            for y in range(field, img.height, 2):
                input_y = y + 2 * demodulation_delay
                while input_y >= img.height:
                    input_y -= 2
                r_line, g_line, b_line = self._modem.demodulate(
                    frame, y + 2 * demodulation_delay, composite[img.width * input_y:img.width * (input_y + 1)])
                output[3 * img.width * y:3 * img.width * (y + 1):3] = _as_bytes(r_line)
                output[3 * img.width * y + 1:3 * img.width * (y + 1) + 1:3] = _as_bytes(g_line)
                output[3 * img.width * y + 2:3 * img.width * (y + 1) + 2:3] = _as_bytes(b_line)
        return Image.frombytes('RGB', img.size, bytes(bytearray(output)))
