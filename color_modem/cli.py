# -*- coding: utf-8 -*-

import sys

from PIL import Image

from color_modem.color.mac import MacModem, MacVariant
from color_modem.color.niir import NiirModem, HueCorrectingNiirModem
from color_modem.color.ntsc import NtscCombModem, NtscVariant, NtscModem
from color_modem.color.pal import Pal3DModem, PalVariant, PalDModem, PalSModem
from color_modem.color.protosecam import ProtoSecamModem, ProtoSecamVariant
from color_modem.color.secam import SecamModem, SecamVariant
from color_modem.comb import Simple3DCombModem, SimpleCombModem, ColorAveragingModem
from color_modem.image import ImageModem
from color_modem.line import LineStandard, LineConfig


def main():
    img = Image.open(sys.argv[1])
    line_config = LineConfig(img.size)

    #### NTSC
    # best quality 3D comb filter
    # modem = Simple3DCombModem(NtscCombModem(line_config))
    # simple 2D comb filter
    # modem = NtscCombModem(line_config)
    # simple bandpass filtering
    # modem = NtscModem(line_config)

    #### PAL
    # best quality 3D comb filter
    # modem = Pal3DModem(line_config)
    # standard PAL-D (2D comb filter)
    # modem = PalDModem(line_config)
    # simple PAL-S (bandpass filtering)
    # modem = PalSModem(line_config)

    #### SECAM
    # better quality modulation - filers out unrepresentable color patterns
    modem = ColorAveragingModem(SecamModem(line_config))
    # basic
    # modem = SecamModem(line_config)
    # prototype 1957 AM 819-line variant
    # modem = ColorAveragingModem(ProtoSecamModem(line_config))

    #### NIIR (SECAM IV)
    # better quality modulation - hue correction
    # modem = HueCorrectingNiirModem(line_config)
    # standard
    # modem = NiirModem(line_config)
    # comb filter - turned out to be a bad idea
    # modem = SimpleCombModem(HueCorrectingNiirModem(line_config))

    #### D2-MAC
    # better quality modulation - filers out unrepresentable color patterns
    # modem = ColorAveragingModem(MacModem(line_config))
    # basic
    # modem = MacModem(line_config)

    img_modem = ImageModem(modem)
    img = img_modem.modulate(img, 0)
    img.save(sys.argv[2])
    img = img_modem.demodulate(img, 0)
    img.save(sys.argv[3])
