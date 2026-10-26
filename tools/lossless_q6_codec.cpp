// Offline pack codec/auditor only. No CUDA, model inference, or GGUF mutation.
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <omp.h>

static constexpr int code[16] = {0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
static uint16_t scale_half(int e) {
    return e < 114 ? uint16_t(1u << (e - 104)) : uint16_t((e - 113) << 10);
}
static float half_float(uint16_t h) {
    const int e = (h >> 10) & 31;
    const int q = h & 1023;
    return e ? std::ldexp(float(1024 + q), e - 25) : std::ldexp(float(q), -24);
}
static uint32_t bits(float v) { uint32_t r; std::memcpy(&r,&v,4); return r; }

extern "C" __declspec(dllexport)
int convert_mx_q6(const uint8_t* src, uint8_t* dst, size_t n, unsigned long long* audited,
                  char* error, size_t error_cap) {
    if (!src || !dst || !audited) return 2;
    std::atomic<size_t> first_bad{n};
    #pragma omp parallel for num_threads(12) schedule(static)
    for (long long group = 0; group < (long long)n; ++group) {
        const uint8_t* m = src + (size_t)group * 136;
        uint8_t* out = dst + (size_t)group * 210;
        int mine = 255, maxe = 0;
        for (int b=0;b<8;++b) { int e=m[17*b]; if(e<mine)mine=e;if(e>maxe)maxe=e; }
        if (mine<104 || mine>143 || maxe-mine>6) { first_bad.store((size_t)group); continue; }
        uint8_t q[256];
        for (int b=0;b<8;++b) {
            const uint8_t* mx=m+b*17;
            const int scale=1 << (mx[0]-mine);
            out[192+2*b]=out[193+2*b]=(uint8_t)scale;
            for(int j=0;j<16;++j) {
                q[32*b+j]=(uint8_t)(code[mx[1+j]&15]+32);
                q[32*b+16+j]=(uint8_t)(code[mx[1+j]>>4]+32);
            }
        }
        for(int h=0;h<2;++h) for(int l=0;l<32;++l) {
            int i=128*h+l;
            out[64*h+l]=(q[i]&15)|((q[i+64]&15)<<4);
            out[64*h+32+l]=(q[i+32]&15)|((q[i+96]&15)<<4);
            out[128+32*h+l]=(q[i]>>4)|((q[i+32]>>4)<<2)|((q[i+64]>>4)<<4)|((q[i+96]>>4)<<6);
        }
        const uint16_t dh=scale_half(mine);
        out[208]=(uint8_t)dh;out[209]=(uint8_t)(dh>>8);
        // Independent Q6 reference indexing, compared to original MX scale/code values.
        // These are separate decoders; no q[] encoder intermediates are used here.
        const float d=half_float((uint16_t)(out[208]|(uint16_t(out[209])<<8)));
        for(int i=0;i<256;++i) {
            int h=i/128, quarter=(i%128)/32, lane=i%32;
            int ql=(out[64*h+32*(quarter&1)+lane]>>(4*(quarter/2)))&15;
            int qh=(out[128+32*h+lane]>>(2*quarter))&3;
            int qi=(ql|(qh<<4))-32;
            float value=(d*float((int8_t)out[192+i/16]))*float(qi);
            int b=i/32,j=i%32;
            uint8_t packed=m[17*b+1+j%16];
            int original_code=code[j<16?packed&15:packed>>4];
            float original=std::ldexp(1.0f,int(m[17*b])-128)*float(original_code);
            if(bits(value)!=bits(original)) { first_bad.store((size_t)group); break; }
        }
    }
    if(first_bad.load()!=n) {
        if(error&&error_cap) std::snprintf(error,error_cap,"nonrepresentable or mismatched block at group %zu",first_bad.load());
        *audited=0;return 1;
    }
    *audited=(unsigned long long)n*256;return 0;
}
