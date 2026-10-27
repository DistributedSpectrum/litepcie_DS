#
# This file is part of LitePCIe.
#
# SPDX-License-Identifier: BSD-2-Clause

import unittest

from migen import *
from migen.sim import run_simulation, passive

from litex.gen import *

from litepcie.phy.s7pciephy import S7RX128BByteEnables
from litepcie.tlp.depacketizer import LitePCIeTLPHeaderExtracter128b


def eof(dw):
    return 0b10000 | (4*dw + 3)


def sof(dw):
    return 0b10000 | (4*dw)


class TestS7RX128BByteEnables(unittest.TestCase):
    def check(self, tkeep, is_sof, is_eof, expected):
        dut = S7RX128BByteEnables()
        result = []

        def generator():
            yield dut.tkeep.eq(tkeep)
            yield dut.is_sof.eq(is_sof)
            yield dut.is_eof.eq(is_eof)
            yield
            result.append((yield dut.be))

        run_simulation(dut, generator())
        self.assertEqual(result[0], expected)

    def test_no_eof_keeps_tkeep(self):
        self.check(0xffff, 0, 0, 0xffff)
        self.check(0x0fff, 0, 0, 0x0fff)

    def test_eof_masks_trailing_lanes(self):
        # The core reports tkeep = 0xffff on the last beat; only the lanes up to the EOF DW are data.
        for dw, expected in enumerate([0x000f, 0x00ff, 0x0fff, 0xffff]):
            self.check(0xffff, 0, eof(dw), expected)

    def test_straddle_keeps_next_packet_lanes(self):
        # Packet ends on DW0/DW1 and the next one starts on DW2 in the same beat.
        self.check(0xffff, sof(2), eof(0), 0xff0f)
        self.check(0xffff, sof(2), eof(1), 0xffff)

    def test_sof_before_eof_is_ignored(self):
        # A packet starting on DW0 and ending in the same beat: only its own lanes.
        self.check(0xffff, sof(0), eof(2), 0x0fff)


class TestS7RX128BCompletionBeats(unittest.TestCase):
    """A 3DW-header completion as the 7-Series core delivers it (tkeep all ones) must leave the
    128-bit header extracter as exactly length/4 payload beats, with no flushed tail beat."""
    def run_completion(self, length, with_byte_enables):
        class DUT(LiteXModule):
            def __init__(self):
                self.byte_enables = S7RX128BByteEnables()
                self.extracter    = LitePCIeTLPHeaderExtracter128b()

        dut   = DUT()
        dws   = [0x4a000000 | length, 0x00000000, 0x00000000] + [0xd0000000 + i for i in range(length)]
        beats = [dws[i:i + 4] for i in range(0, len(dws), 4)]
        out   = []

        def source():
            for n, beat in enumerate(beats):
                beat = beat + [0]*(4 - len(beat))
                last = n == len(beats) - 1
                yield dut.extracter.sink.dat.eq(sum(w << (32*i) for i, w in enumerate(beat)))
                yield dut.byte_enables.tkeep.eq(0xffff)
                yield dut.byte_enables.is_sof.eq(sof(0) if n == 0 else 0)
                yield dut.byte_enables.is_eof.eq(eof((len(dws) - 1) % 4) if last else 0)
                yield
                yield dut.extracter.sink.be.eq((yield dut.byte_enables.be) if with_byte_enables else 0xffff)
                yield dut.extracter.sink.first.eq(n == 0)
                yield dut.extracter.sink.last.eq(last)
                yield dut.extracter.sink.valid.eq(1)
                yield
                while not (yield dut.extracter.sink.ready):
                    yield
                yield dut.extracter.sink.valid.eq(0)
            for _ in range(16):
                yield

        @passive
        def sink():
            yield dut.extracter.source.ready.eq(1)
            while True:
                if (yield dut.extracter.source.valid):
                    out.append(((yield dut.extracter.source.dat), (yield dut.extracter.source.be),
                                (yield dut.extracter.source.last)))
                yield

        run_simulation(dut, [source(), sink()])
        return out

    def test_completion_beats(self):
        for length in [32, 64, 1, 2, 3, 4, 5]:
            out = self.run_completion(length, with_byte_enables=True)
            payload = [(dat >> (32*i)) & 0xffffffff for dat, be, _ in out for i in range(4)
                       if (be >> (4*i)) & 0xf]
            self.assertEqual(payload, [0xd0000000 + i for i in range(length)], f"length={length}")
            self.assertEqual(len(out), (length + 3)//4, f"length={length}")
            self.assertEqual([last for *_, last in out], [0]*(len(out) - 1) + [1])

    def test_tkeep_alone_adds_a_tail_beat(self):
        # Without the EOF-derived byte enables the last beat looks full and a junk tail beat is
        # emitted: a 32-DW completion comes out as 9 beats (the DMA reader stall on Jetson hosts).
        out = self.run_completion(32, with_byte_enables=False)
        self.assertEqual(len(out), 9)


if __name__ == "__main__":
    unittest.main()
