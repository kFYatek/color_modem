# -*- coding: utf-8 -*-

import collections


class LineStandard(collections.namedtuple('LineStandard',
                                          ['frame_rate',
                                           'total_lines',
                                           'odd_field_first_active_line',
                                           'odd_field_last_active_line',
                                           'even_field_first_active_line',
                                           'even_field_last_active_line',
                                           'total_width_factor'])):
    def __new__(cls, *args, **kwargs):
        self = LineStandard.__base__.__new__(cls, *args, **kwargs)
        assert self.odd_field_last_active_line >= self.odd_field_first_active_line
        assert self.even_field_last_active_line >= self.even_field_first_active_line
        assert (self.odd_field_last_active_line - self.odd_field_first_active_line) \
               == (self.even_field_last_active_line - self.even_field_first_active_line)
        assert self.active_lines <= self.total_lines
        return self

    @property
    def active_lines(self):
        return (self.odd_field_last_active_line - self.odd_field_first_active_line
                + self.even_field_last_active_line - self.even_field_first_active_line + 2)

    @classmethod
    def detect(cls, active_lines):
        standards = (std for std in cls.__dict__.values() if isinstance(std, cls))
        standards = list(sorted(standards, key=lambda std: std.active_lines, reverse=True))
        standard = None
        for std in standards:
            if std.active_lines < active_lines:
                break
            standard = std
        if standard is None:
            raise IndexError('No supported line standard supports %d lines' % (active_lines,))
        return standard


LineStandard.BAIRD_405 = LineStandard(25.0, 405, 16, 203, 218, 405, 1.2)
LineStandard.NTSC_525 = LineStandard(30000.0 / 1001.0, 525, 21, 263, 283, 525, 858.0 / 720.0)
LineStandard.GERBER_625 = LineStandard(25.0, 625, 336, 623, 23, 310, 1.2)
LineStandard.FRENCH_819 = LineStandard(25.0, 819, 39, 407, 448, 816, 1.2)
LineStandard.BELGIAN_819 = LineStandard(25.0, 819, 437, 816, 27, 406, 1.2)


class LineConfig(object):
    def __init__(self, size, line_standard=None):
        if line_standard is None:
            line_standard = LineStandard.detect(size[1])
        self.fs = line_standard.frame_rate * line_standard.total_lines * size[0] * line_standard.total_width_factor
        self.line_standard = line_standard
        self._line_shift = (line_standard.active_lines - size[1]) // 2

    def analog_line(self, digital_line):
        adjusted = digital_line + self._line_shift
        if adjusted % 2 == 0:
            return self.line_standard.even_field_first_active_line + adjusted // 2
        else:
            return self.line_standard.odd_field_first_active_line + adjusted // 2

    def is_alternate_line(self, frame, line):
        return self.analog_line(line) % 2 == frame % 2

