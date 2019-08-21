# Color Modem

A little project that replicates effects of analog TV color standards in
software. Written mostly to educate myself on how these standards work, and on
modulations and filters.

The code is messy, unoptimal, and bad in general. There is no real CLI, either.
You'll most likely need to make ad-hoc modifications to the `color_modem/cli.py`
file to make this code do anything useful. Or call the functions directly from a
Python REPL.

## Supported formats

* **NTSC**-style simple QAM
  * with presets for:
    * Standard NTSC
    * NTSC-A, as reportedly tested by BBC in the 1950s
    * NTSC-I, as reportedly tested by BBC in the 1960s; Soviet OSKM system, also
      reportedly tested in the 1960s, was apparently identical
    * NTSC 4.43, as output by some PAL VCRs when playing NTSC tapes
    * NTSC-N - it has apparently been proposed to CCIR in the 1960s; the
      original document is nowhere to be found on the web, so I guessed the
      subcarrier frequency myself based on typical practices
    * NTSC 3.61, which is a non-standard format that is apparently output by the
      Raspberry Pi when nominally configured for PAL-M
  * decoder variants:
    * simple band splitter
    * 2D comb filter
    * 3D comb filter
* **PAL**
  * with presets for:
    * Standard 625-line PAL, as used on Systems B, D, G, H, I and K
    * PAL-M, as used in Brazil
    * PAL-N, as used in Argentina, Paraguay and Uruguay
  * decoder variants:
    * simple band splitter (PAL-S)
    * PAL-D
    * 3D comb filter
* **SECAM**, with presets for:
  * SECAM IIIb / SECAM III optimized / the final broadcast SECAM, as used on
    systems B, D, G, H, K and L
  * SECAM III, as originally proposed to CCIR in the 1960s
  * possible recreations of SECAM I and SECAM II, based on decriptions by Sven
    Boetcher and Eckhard Matzel in
    ["Medienwissenschaft"](https://books.google.pl/books?id=j8HB9_W_llEC&pg=PA2180)
  * possible recreations of SECAM-A, SECAM-E, SECAM-M and SECAM-N; SECAM-M
    reportedly used to be broadcast in Cambodia and Vietnam, the rest were only
    used for test purposes
  * SECAM (and MAC) encoder can optionally average neighboring lines' color
    information, to reduce artifacts when decoding
* **Initial AM-based variant of SECAM**, as tested in late 1950s and early 1960s
  on the French 819-line system
* **NIIR / SECAM-IV**, as proposed by the Soviet Scientific Research Institute
  for Radio in the 1960s, and considered by CCIR/ITU into the 1990s
  * NIIR encoder can optionally average neighboring lines' hue, to reduce
    artifacts when decoding
* **MAC**-style time multiplexing, based on the analog portion of D2-MAC
  * Bandwidth limiting is supported, with a preset emulating D2-MAC in a 7 MHz
    VSB channel

Note that only the visible portion of the signal is processed. All
synchronization signals are implicit. This also means that attempting to decode
real-life signals might yield false colors due to misaligned color burst.

## Copyright notice

This is free and unencumbered software released into the public domain using The
Unlicense. Please see the `LICENSE` file or http://unlicense.org for details.
