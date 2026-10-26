        // No mailbox was reserved for this window. The completed route event
        // precedes plan reuse; prepare queued d.st's stage-ready wait before use.
        bool gpu_misses_used=false;
        if(!gpu_misses_prepare(l,d,T,h_idx,gpu_misses_used,err))return false;
        if(gpu_misses_used) {
            uint32_t* p=prof.data()+(size_t)l.il*g.n_exp;
            for(int t=0;t<T;++t)for(int i=0;i<l.k_exp;++i) {
                const int e=h_idx[t*kPicks+i];++p[e];l.heat[(size_t)e]+=1.0f;
                if(l.slot[(size_t)e]<0)l.cpu_hits.push_back(e);
            }
            const int before_errors=gg::launch_errors();
            if(T==1) {
                x3::MoeArgs ma;
                ma.plan_ptr=d.plan_ptr;ma.plan_w=d.rw;ma.cpu_part=nullptr;ma.cpu_flag=nullptr;
                ma.lay=l.lay;ma.k=l.k_exp;ma.n_embd=D;ma.n_ff=FF;ma.limit=g.lim;
                gg::moe_gate_up(ma,d.x,d.xg,s);
                gg::moe_down(ma,d.xg,d.sh_out,d.moe_out,s);
            } else {
                gf::MoeWin mw;
                mw.plan_ptr=d.plan_ptr;mw.T=T;mw.k=l.k_exp;mw.grp_ptr=d.grp_ptr;
                mw.grp_start=d.grp_start;mw.grp_ent=d.grp_ent;mw.n_groups=d.n_groups;
                gf::moe_win_group(mw,s);
                x3::WinArgs wa;
                wa.grp_ptr=d.grp_ptr;wa.grp_start=d.grp_start;wa.grp_ent=d.grp_ent;wa.n_groups=d.n_groups;
                wa.plan_ptr=d.plan_ptr;wa.plan_w=d.rw;wa.cpu_part=nullptr;wa.cpu_flag=nullptr;wa.wait_ptr=nullptr;
                wa.x=d.x;wa.x_stride=(size_t)D;wa.h=d.xg;wa.part=d.ed;wa.sh_out=d.sh_out;
                wa.out=d.moe_out;wa.out_stride=(size_t)D;wa.lay=l.lay;wa.T=T;wa.k=l.k_exp;
                wa.n_embd=D;wa.n_ff=FF;wa.limit=g.lim;
                if(!gg::win_experts(wa,s))return cache_epoch_terminal("GPU miss window kernels refused",err);
            }
            if(gg::launch_errors()!=before_errors||cudaPeekAtLastError()!=cudaSuccess)
                return cache_epoch_terminal("GPU miss expert kernel launch failed",err);
            ++st.gpu_miss_kernel_windows;st.experts+=(int64_t)T*l.k_exp;
            hc_post(d.moe_out,d.h,d.f_post,d.f_comb,D,H,T,d.h2,s);
            std::swap(d.h,d.h2);
            if(cudaMemcpyAsync(d.pre,d.f_pre,(size_t)T*8*4,cudaMemcpyDeviceToDevice,s)!=cudaSuccess)
                return cache_epoch_terminal("GPU miss post-mix state copy failed",err);
            return true;
        }
