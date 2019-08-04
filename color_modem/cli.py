# -*- coding: utf-8 -*-
import itertools
import sys

import numpy
from PIL import Image

from color_modem.comb import Simple3DCombModem
from color_modem.ntsc import NtscCombModem, NtscVariant, NtscModem
from color_modem.pal import Pal3DModem, PalVariant, PalDModem, PalSModem
from color_modem.secam import SecamModem


def modulate(modem, img, frame):
    assert img.width == 720
    if img.mode != 'RGB':
        img = img.convert('RGB')
    r = bytearray(img.getdata(0))
    g = bytearray(img.getdata(1))
    b = bytearray(img.getdata(2))
    output = numpy.zeros(720 * img.height, dtype=numpy.uint8)
    for y in itertools.chain(range(0, img.height, 2), range(1, img.height, 2)):
        output[720 * y:720 * (y + 1)] = modem.encode_composite_level(
            modem.modulate(frame, y, r[720 * y:720 * (y + 1)], g[720 * y:720 * (y + 1)], b[720 * y:720 * (y + 1)]))

    return Image.frombytes('L', img.size, bytes(bytearray(output)))


def demodulate(modem, img, frame):
    def as_bytes(array):
        return numpy.uint8(numpy.rint(numpy.maximum(numpy.minimum(array, 255.0), 0.0)))

    assert img.width == 720
    if img.mode != 'L':
        img = img.convert('L')
    composite = modem.decode_composite_level(numpy.array(img.getdata()))
    output = numpy.zeros(720 * img.height * 3, dtype=numpy.uint8)
    demodulation_delay = getattr(modem, 'demodulation_delay', 0)
    for field in range(2):
        for y in range(field, 2 * demodulation_delay, 2):
            modem.demodulate(frame, y, composite[720 * y:720 * (y + 1)])
        for y in range(field, img.height, 2):
            input_y = y + 2 * demodulation_delay
            while input_y >= img.height:
                input_y -= 2
            rLine, gLine, bLine = modem.demodulate(frame, y + 2 * demodulation_delay,
                                                   composite[720 * input_y:720 * (input_y + 1)])
            output[3 * 720 * y:3 * 720 * (y + 1):3] = as_bytes(rLine)
            output[3 * 720 * y + 1:3 * 720 * (y + 1) + 1:3] = as_bytes(gLine)
            output[3 * 720 * y + 2:3 * 720 * (y + 1) + 2:3] = as_bytes(bLine)
    return Image.frombytes('RGB', img.size, bytes(bytearray(output)))


def main():
    #### NTSC
    # best quality 3D comb filter
    # modem = Simple3DCombModem(NtscCombModem(NtscVariant.NTSC))
    # simple 2D comb filter
    # modem = NtscCombModem(NtscVariant.NTSC)
    # simple bandpass filtering
    # modem = NtscModem(NtscVariant.NTSC)

    #### PAL
    # best quality 3D comb filter
    # modem = Pal3DModem(PalVariant.PAL)
    # standard PAL-D (2D comb filter)
    # modem = PalDModem(PalVariant.PAL)
    # simple PAL-S (bandpass filtering)
    # modem = PalSModem(PalVariant.PAL)

    #### SECAM
    modem = SecamModem()

    img = Image.open(sys.argv[1])
    img = modulate(modem, img, 0)
    img.save(sys.argv[2])
    img = demodulate(modem, img, 0)
    img.save(sys.argv[3])
