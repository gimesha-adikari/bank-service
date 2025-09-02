# app/features/ocr/fields/nic.py
from __future__ import annotations

import re
from typing import Dict

# Sri Lanka NIC formats:
#   Old:  9 digits + [VvXx]  e.g., 912345678V
#   New:  12 digits          e.g., 199912345678
NIC_OLD_RE = re.compile(r"\b(\d{9})\s*([VvXx])\b")
NIC_NEW_RE = re.compile(r"\b(\d{12})\b")

# Anchored per-line to avoid bleeding into next line
NAME_RE = re.compile(r"(?mi)^\s*(?:name|full\s*name)\s*[:\-]\s*([^\r\n]+?)\s*$")
DOB_RE  = re.compile(r"(?mi)^\s*(?:dob|date\s*of\s*birth)\s*[:\-]\s*([0-9]{4}[-/.][0-9]{1,2}[-/.][0-9]{1,2})\s*$")


def parse_nic_text(text: str) -> Dict[str, str]:
    """
    Best-effort parser for LK NIC text blocks. Returns any fields it can find.
    Keys: nic, nic_format, name, dob
    """
    out: Dict[str, str] = {}

    m_new = NIC_NEW_RE.search(text)
    m_old = NIC_OLD_RE.search(text)

    if m_new:
        out["nic"] = m_new.group(1)
        out["nic_format"] = "new_12"
    elif m_old:
        out["nic"] = f"{m_old.group(1)}{m_old.group(2).upper()}"
        out["nic_format"] = "old_9X"

    m_name = NAME_RE.search(text)
    if m_name:
        out["name"] = m_name.group(1).strip()

    m_dob = DOB_RE.search(text)
    if m_dob:
        # Normalize to YYYY-MM-DD
        dob = m_dob.group(1).replace("/", "-").replace(".", "-")
        y, m, d = dob.split("-")
        if len(m) == 1:
            m = m.zfill(2)
        if len(d) == 1:
            d = d.zfill(2)
        out["dob"] = "-".join([y, m, d])

    return out
