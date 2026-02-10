from __future__ import annotations

from pathlib import Path


# Byte-level replacements so the script is safe to run from shells with awkward encodings.
REPLACEMENTS: list[tuple[bytes, bytes]] = [
    # Common punctuation mojibake
    (bytes.fromhex("C3A2E282ACE2809D"), bytes.fromhex("E28094")),  # â€” -> —
    (bytes.fromhex("C3A2E280A2"), bytes.fromhex("E280A2")),        # â€¢ -> •
    (bytes.fromhex("C3A2E280A6"), bytes.fromhex("E280A6")),        # â€¦ -> …
    (bytes.fromhex("C3A2E280A0E28099"), bytes.fromhex("E28692")),  # â†’ -> →
    (bytes.fromhex("C3A2E282ACE284A2"), bytes.fromhex("27")),      # â€™ -> '

    # Emoji / symbols mojibake seen in this repo
    (bytes.fromhex("C3A2C593E280A6"), bytes.fromhex("E29C85")),    # âœ… -> ✅
    (bytes.fromhex("C3B0C5B8E2809CCB86"), bytes.fromhex("F09F9388")),  # ðŸ“ˆ -> 📈
    (bytes.fromhex("C3B0C5B8C5A1E282AC"), bytes.fromhex("F09F9A80")),  # ðŸš€ -> 🚀
    (bytes.fromhex("C3B0C5B8E2809CC5A0"), bytes.fromhex("F09F938A")),  # ðŸ“Š -> 📊
    (bytes.fromhex("C3B0C5B8C2A7C2AA"), bytes.fromhex("F09FA7AA")),    # ðŸ§ª -> 🧪
    (bytes.fromhex("C3B0C5B8E28094C5BEC3AFC2B8C28F"), bytes.fromhex("F09F979EEFB88F")),  # ðŸ—žï¸ -> 🗞️
    (bytes.fromhex("C3B0C5B8E2809DE2809E"), bytes.fromhex("F09F9484")),  # ðŸ”„ -> 🔄
    (bytes.fromhex("C3A2C2ACE280A1C3AFC2B8C28F"), bytes.fromhex("E2AC87EFB88F")),  # â¬‡ï¸ -> ⬇️

    # Currency artifacts
    (bytes.fromhex("C382C2A3"), bytes.fromhex("C2A3")),  # Â£ -> £
    (bytes.fromhex("C38220"), bytes.fromhex("20")),      # Â  -> space
]


def normalize_file(path: Path) -> None:
    raw = path.read_bytes()
    # Normalize line endings to LF to avoid mixed CRLF/LF in patches.
    raw = raw.replace(b"\r\n", b"\n")
    for a, b in REPLACEMENTS:
        raw = raw.replace(a, b)
    path.write_bytes(raw)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    targets = [root / "app.py"]
    for t in targets:
        if t.exists():
            normalize_file(t)


if __name__ == "__main__":
    main()

