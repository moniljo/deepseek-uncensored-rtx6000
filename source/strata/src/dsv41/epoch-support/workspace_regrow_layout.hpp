#pragma once
// Exact raw actual shrink47 formula block, reused by model and pure C++ test.
#include "workspace_regrow_transaction.hpp"
namespace strata::ds4::chunkregrow {
template<class ChunkLike,class Geometry,class Pack,class Add>
bool append_layout(ChunkLike& c,const Geometry& g,const Pack& pk,size_t oldC,size_t newC,
                   int oldA,int newA,int oldAI,int newAI,size_t topkWidth,size_t candWidth,Add&& add) {
    if(sizeof(size_t)<8||oldC!=1024||newC!=65280||oldA!=1024||newA!=4096||oldAI!=128||newAI!=128||
       g.dim!=5120||g.hc!=4||g.hd!=512||g.ff!=2304||g.q_lora!=1280||g.heads!=64||
       g.o_groups!=8||g.o_lora!=1024||g.win!=128||topkWidth!=512||candWidth!=2048||
       pk.eg_ngram<2||pk.eg_ngram>25||pk.eg_heads<1||pk.eg_heads>24||
       (size_t)(pk.eg_ngram-1)*pk.eg_heads!=24||pk.eg_hd!=256||pk.eg_layers!=2)return false;
    const size_t D=(size_t)g.dim,H=(size_t)g.hc,HD=(size_t)g.hd,FF=(size_t)g.ff;
    constexpr size_t kPicks=8; // actual padded physical routing stride, selected experts stay6
    const auto head_members=[](int){return 1;};
    const auto topk_stride=[&]{return topkWidth;};
    const auto cand_stride=[&]{return candWidth;};
        // Proven C-scaled allocations, matching alloc_chunks's element counts.
        add("h", c.h, oldC*H*D, newC*H*D); add("x", c.x, oldC*D, newC*D);
        add("acc", c.acc, oldC*D, newC*D); add("x16", c.x16, oldC*D, newC*D);
        add("qr", c.qr, oldC*g.q_lora, newC*g.q_lora); add("qr16", c.qr16, oldC*g.q_lora, newC*g.q_lora);
        add("kvr", c.kvr, oldC*HD, newC*HD); add("ckr", c.ckr, oldC*HD, newC*HD); add("cgr", c.cgr, oldC*HD, newC*HD);
        add("latent", c.latent, (oldC+2)*HD, (newC+2)*HD); add("lat16", c.lat16, (oldC+2)*HD, (newC+2)*HD);
        add("ikr", c.ikr, (oldC+2)*g.ix_hd, (newC+2)*g.ix_hd);
        add("ext", c.ext, (oldC+g.win)*HD, (newC+g.win)*HD);
        add("iw", c.iw, oldC*64, newC*64); add("topk", c.topk, oldC*topk_stride(), newC*topk_stride());
        add("cand", c.cand, oldC*cand_stride(), newC*cand_stride()); add("mix", c.mix, oldC*32, newC*32);
        add("a_pre", c.a_pre, oldC*8, newC*8); add("a_post", c.a_post, oldC*8, newC*8);
        add("a_comb", c.a_comb, oldC*64, newC*64); add("f_pre", c.f_pre, oldC*8, newC*8);
        add("f_post", c.f_post, oldC*8, newC*8); add("f_comb", c.f_comb, oldC*64, newC*64);
        add("pre", c.pre, oldC*8, newC*8); add("glog", c.glog, oldC*1024, newC*1024);
        add("ridx", c.ridx, oldC*kPicks, newC*kPicks); add("rw", c.rw, oldC*kPicks, newC*kPicks);
        add("row_tok", c.row_tok, oldC*kPicks, newC*kPicks); add("pos", c.pos, oldC*kPicks, newC*kPicks);
        add("sg", c.sg, oldC*FF, newC*FF); add("su", c.su, oldC*FF, newC*FF);
        add("sh16", c.sh16, oldC*FF, newC*FF); add("h16", c.h16, oldC*FF, newC*FF);
        add("y", c.y, oldC*D, newC*D); add("emb", c.emb, oldC*D, newC*D);
        const size_t cols = (size_t)(pk.eg_ngram-1)*pk.eg_heads, per = cols*(pk.eg_hd+pk.eg_hd/32);
        add("eg_rows", c.eg_rows, oldC*per, newC*per);
        const size_t NHL = (size_t)g.heads / head_members(1), HQ = NHL*HD, OG = (size_t)g.o_groups / head_members(1);
        add("q", c.q, (size_t)oldA*HQ, (size_t)newA*HQ); add("o", c.o, (size_t)oldA*HQ, (size_t)newA*HQ);
        add("o16", c.o16, (size_t)oldA*HQ, (size_t)newA*HQ);
        add("oa", c.oa, (size_t)oldA*OG*g.o_lora, (size_t)newA*OG*g.o_lora);
        add("oa16", c.oa16, (size_t)oldA*OG*g.o_lora, (size_t)newA*OG*g.o_lora);
        auto hand_bytes = [&](size_t n) { return n*(H*D*4 + 8*4 + topk_stride()*4 + cand_stride()*4); };
        for (int slot = 0; slot < 2; ++slot) {
            add("emb_host", c.emb_host[slot], oldC*D, newC*D, shrink::Kind::pinned);
            add("eg_host", c.eg_host[slot], oldC*pk.eg_layers*per, newC*pk.eg_layers*per, shrink::Kind::pinned);
            add("hand", c.hand[slot], hand_bytes(oldC), hand_bytes(newC), shrink::Kind::mapped);
        }
    return true;
}
} // namespace strata::ds4::chunkregrow
