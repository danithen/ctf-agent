"""Deterministic triage brief: zero-info seed facts (inventory, ELF arch, needle strings)."""
from pathlib import Path

from ctf_agent.tools.triage import _elf_desc, triage_brief


def _elf64(machine: int = 0x3E, entry: int = 0x1050) -> bytes:
    h = bytearray(64)
    h[:4] = b"\x7fELF"
    h[4] = 2                                   # 64-bit
    h[5] = 1                                   # little-endian
    h[18:20] = machine.to_bytes(2, "little")
    h[24:32] = entry.to_bytes(8, "little")
    return bytes(h)


def test_elf64_x86_64_sniffed():
    assert _elf_desc(_elf64()) == "ELF64 x86-64, entry 0x1050"


def test_unknown_machine_falls_back_to_number():
    # 0x12 = SPARC: header sniff must never crash on an unusual arch byte.
    assert "machine 0x12" in _elf_desc(_elf64(machine=0x12))


def test_i386_32bit_sniffed():
    h = bytearray(_elf64())
    h[4] = 1
    h[18:20] = (3).to_bytes(2, "little")
    h[24:28] = (0x11E0).to_bytes(4, "little")
    assert _elf_desc(bytes(h)) == "ELF32 i386, entry 0x11e0"


def test_non_elf_returns_data():
    assert _elf_desc(b"#!/bin/sh\nexit 0\n") is None


def test_needle_strings_with_offsets(tmp_path: Path):
    data = b"\x00" * 100 + b"Please enter the flag:\x00" + b"nothing" + b"key" + b"Correct!" + b"\x00"
    p = tmp_path / "prog.bin"
    p.write_bytes(data)
    brief = triage_brief(tmp_path)
    assert "- prog.bin (" + str(len(data)) + " bytes, data)" in brief
    assert '0x64: "Please enter the flag:"' in brief
    assert "Correct!" in brief


def test_hits_capped_per_file(tmp_path: Path):
    p = tmp_path / "x"
    p.write_bytes(b"\x00".join(b"flag_part%d" % i for i in range(10)))
    brief = triage_brief(tmp_path)
    assert brief.count("    0x") == 6 and "_part9" not in brief


def test_no_files_returns_empty(tmp_path: Path):
    assert triage_brief(tmp_path) == ""


def test_broken_file_never_breaks(tmp_path: Path):
    (tmp_path / "d").mkdir()   # dir in workspace: iterdir filters is_file, no crash
    (tmp_path / "f").write_bytes(_elf64())
    assert "f (" in triage_brief(tmp_path)
