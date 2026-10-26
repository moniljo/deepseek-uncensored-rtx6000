# Public-source benchmark corpus notices

This is the exact frozen text used by the retained-context benchmark, not a stock Python distribution or an importable module bundle. It has 208 labelled sections, five synthetic audit comments, an instruction preamble and a truncated last section. Its exact original Python/distribution version has not been established. `_distutils_system_mod.py` explicitly describes Debian changes. The `_pydecimal.py`-labelled section also contains generated build configuration. The captured-section inventory records bytes; it does not claim that every labelled section is a clean upstream module file.

Python source is distributed under the Python Software Foundation License Version 2, with incorporated-code notices retained. The unchanged generic CPython license and acknowledgements are in `PYTHON-LICENSE.txt`. Original copyright and SPDX notices remain in the compressed text and inventory. They include Python Software Foundation, CWI/CNRI, Google, Armin Ronacher, Brian Quinlan and MagicStack notices. The asyncio files carrying `PSF-2.0 AND (MIT OR Apache-2.0)` are redistributed under the PSF and Apache option; the Apache terms are included in the accompanying license file.

Primary license and provenance references:

- [Python history and license](https://docs.python.org/3/license.html)
- [Debian Python copyright](https://sources.debian.org/src/python3.11/3.11.2-6%2Bdeb12u8/debian/copyright/)
- [Debian distutils customization](https://sources.debian.org/src/python3.13/3.13.5-2%2Bdeb13u4/debian/patches/distutils-install-layout.diff/)

The gzip stream uses an empty filename, timestamp 0, compression level 9 and OS byte 255. Decompress it as UTF-8 and require the manifest SHA-256 before tokenization. It does not depend on the reader's platform Python source installation. Legacy fact-position metadata is preserved separately from measured byte/Unicode offsets; the frozen bytes and hashes are authoritative.

No million-ID array, HTTP request, client prompt, model weight, credential or local personal path is bundled. Public Python source naturally contains example Unix paths, project URLs and original author contact notices. Those public-source strings have not been redacted, because doing so would change the benchmark input. The two instruction lines and five audit comments are synthetic benchmark framing, not client data.
