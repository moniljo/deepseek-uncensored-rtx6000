# Exact FP4 indexer attribution

The experimental ds4_index_native_impl.cuh adapts the exact numerical guard, query packing, native21 tensor-core mapping, and ordered head accumulation from the local Xolotl source25 overlay in lightning-indexer.cu and main22.cuh. Its minimal native MMA/ldmatrix register mapping comes from ggml mma.cuh. This is a local isolated experiment, not an upstream release. Strata cache layout, causal/candidate mapping, fallback, lifetime and diagnostic fixture are adapted here.

Source snapshot SHA-256:
- ggml\src\ggml-cuda\lightning-indexer.cu: 9c55f79a47128ef0902d9baf50fe376e03b4b803d30832f7723e08e6702828e7
- ggml\src\ggml-cuda\mma.cuh: 26d47789b8ca405c7ae3d596d01df26dd51466a0d1cedba7b0d2cbba968394fe
- ggml\src\ggml-cuda\main22.cuh: 72d8e6d0b93634063a31b1efb3b82276cebdf2ec5aaa5e2eb4ad5aec7ce488d4


The copied material is distributed under the ggml MIT license below.

MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
