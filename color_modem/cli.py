# -*- coding: utf-8 -*-

import sys

from PIL import Image

from color_modem.comb import Simple3DCombModem, SimpleCombModem
from color_modem.image import ImageModem
from color_modem.mac import MacModem, MacVariant, AveragingMacModem
from color_modem.niir import NiirModem, HueCorrectingNiirModem
from color_modem.ntsc import NtscCombModem, NtscVariant, NtscModem
from color_modem.pal import Pal3DModem, PalVariant, PalDModem, PalSModem
from color_modem.secam import SecamModem, AveragingSecamModem


def main():
    #### NTSC
    # best quality 3D comb filter
    # modem = Simple3DCombModem(NtscCombModem())
    # simple 2D comb filter
    # modem = NtscCombModem()
    # simple bandpass filtering
    # modem = NtscModem()

    #### PAL
    # best quality 3D comb filter
    # modem = Pal3DModem()
    # standard PAL-D (2D comb filter)
    # modem = PalDModem()
    # simple PAL-S (bandpass filtering)
    # modem = PalSModem()

    #### SECAM
    # better quality modulation - filers out unrepresentable color patterns
    modem = AveragingSecamModem()
    # basic
    # modem = SecamModem()

    #### NIIR (SECAM IV)
    # better quality modulation - hue correction
    # modem = HueCorrectingNiirModem()
    # standard
    # modem = HueCorrectingNiirModem()
    # comb filter - turned out to be a bad idea
    # modem = SimpleCombModem(NiirModem(), avg=lambda a, b: 0.5 * (a + b))

    #### D2-MAC
    # better quality modulation - filers out unrepresentable color patterns
    # modem = AveragingMacModem()
    # basic
    # modem = MacModem()

    img_modem = ImageModem(modem)
    img = Image.open(sys.argv[1])
    img = img_modem.modulate(img, 0)
    img.save(sys.argv[2])
    img = img_modem.demodulate(img, 0)
    img.save(sys.argv[3])
