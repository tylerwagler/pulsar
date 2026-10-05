/* exl3_gemv_gate, L251 section: Qwen3.8-Flash-Next's routed experts -- 512
 * experts x top-10, in 2560, mid 640, gate_up FUSED (one [2560 -> 1280] slice
 * per expert, gate rows then up rows, as the container carries it) and down
 * 640 -> 2560, rates K = 4 and 5.  The down's K = 640 = 5 x 128 is the arm's
 * K % 256 == 128 tail: the last staging pass of every row is one Hadamard
 * block.  Graded like the V4.1 sections (host authority + double), plus the
 * row-block identity on the tail, a mutation and the refusals.  Included by
 * tests/exl3_gemv_gate.cu and run AFTER the V4.1 sections, so their random
 * draws -- and the byte hashes they print -- do not move. */

/* one projection per assignment: z = W_hat^T H(suh x) of a [K -> M] slice */
static void grade_fused(const char *what, const routing &r, const acts &a, int K, int M, int k2,
                        const std::vector<expert_blob> &ex, const std::vector<float> &z_dev, int n_check) {
    double worst = 0, scale = 0;
    std::vector<double> w;
    int checked = 0;
    for (int col = 0; col < r.n_assign && checked < n_check; col++, checked++) {
        const int pair = r.ids_dst[col], e = r.selected[pair], t = r.ids_tok[col];
        dequant_expert(ex[e], K, M, k2, w);
        std::vector<double> z;
        double s;
        ref_gemv(w, K, M, a.x.data() + (size_t)t * K, ex[e].suh(), true, z);
        worst = fmax(worst, max_rel(z_dev, z, (size_t)pair * M, M, &s));
        scale = fmax(scale, s);
    }
    CHECK(worst < 2e-5, "Qwen %s K=%g: max rel %.3e vs the host authority (limit 2e-5)", what, k2 / 2.0, worst);
    printf("  Qwen %s K=%g: %d of %d assignments x %d outputs, max rel %.2e (|z| up to %.3g)\n",
           what, k2 / 2.0, checked, r.n_assign, M, worst, scale);
}

template <typename T> static T *to_dev(const std::vector<T> &v) {
    T *d = nullptr;
    if (cudaMalloc((void **)&d, v.size() * sizeof(T)) != cudaSuccess) return nullptr;
    cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}
static std::vector<float> from_dev(const float *d, size_t n) {
    std::vector<float> v(n);
    cudaDeviceSynchronize();
    cudaMemcpy(v.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost);
    return v;
}

/* the fold's mid (E4M3 + its scale bytes) re-staged as the down GEMV's input */
static acts restage_mid(const routing &r, const std::vector<uint8_t> &mq, const std::vector<uint8_t> &msf,
                        int MID, int kbp) {
    acts am;
    const int nbm = MID / 128;
    am.resize((size_t)nbm * r.n_assign);
    am.x.assign((size_t)r.n_assign * MID, 0.0);
    for (int col = 0; col < r.n_assign; col++) {
        const int pair = r.ids_dst[col];
        for (int b = 0; b < nbm; b++) {
            block_mx_act_mmq &blk = am.blk((size_t)b * r.n_assign + col);
            for (int g = 0; g < 4; g++) {
                const uint8_t sb = msf[pulsar_mx_sfoff(pair, b * 4 + g, kbp)];
                blk.d4[g] = (float)sb;
                for (int j = 0; j < 32; j++) {
                    const uint8_t q = mq[(size_t)pair * MID + b * 128 + g * 32 + j];
                    blk.qs[g * 32 + j] = (int8_t)q;
                    am.x[(size_t)pair * MID + b * 128 + g * 32 + j] = (double)e4m3_to_f32(q) * exp2((double)sb - 127.0);
                }
            }
        }
    }
    return am;
}

static int qwen_section(void) {
    const int K = 2560, MID = 640, GU = 2 * MID, E = 16, n_tok = 3, n_slot = 10;
    printf("exl3-gemv-gate (L251 Qwen): fused gate_up %d -> %d, down %d -> %d, top-%d of %d experts, K = 4 and 5\n",
           K, GU, MID, K, n_slot, E);
    const routing r = make_routing(n_tok, n_slot, E);
    const acts a = make_acts(r, K);
    int32_t *d_ids = to_dev(r.ids_dst), *d_bounds = to_dev(r.bounds), *d_sel = to_dev(r.selected);
    float *d_w = to_dev(r.weights);
    const block_mx_act_mmq *d_act = (const block_mx_act_mmq *)to_dev(a.raw);
    float *d_zgu = nullptr, *d_zd = nullptr, *d_out = nullptr;
    uint32_t *d_nf = nullptr;
    CUDA_OK(cudaMalloc((void **)&d_zgu, (size_t)r.n_assign * GU * sizeof(float)));
    CUDA_OK(cudaMalloc((void **)&d_zd, (size_t)r.n_assign * K * sizeof(float)));
    CUDA_OK(cudaMalloc((void **)&d_out, (size_t)n_tok * K * sizeof(float)));
    CUDA_OK(cudaMalloc((void **)&d_nf, sizeof(uint32_t)));
    const int kbp = pulsar_mx_kbp(MID);
    const size_t sf_bytes = pulsar_mx_sf_slab_bytes(r.n_assign, kbp);
    uint8_t *d_mq = nullptr, *d_msf = nullptr;
    CUDA_OK(cudaMalloc((void **)&d_mq, (size_t)r.n_assign * MID));
    CUDA_OK(cudaMalloc((void **)&d_msf, sf_bytes));

    for (int k2 : {8, 10}) {
        std::vector<expert_blob> gu, down;
        for (int e = 0; e < E; e++) { gu.push_back(make_expert(K, GU, k2)); down.push_back(make_expert(MID, K, k2)); }
        dev_stack sgu, sd;
        if (upload_stack(gu, sgu) || upload_stack(down, sd)) return 1;

        /* fused gate_up */
        int rc = ds4_exl3_moe_gemv_fused_launch(sgu.table, k2, d_act, d_ids, d_bounds, d_zgu, GU, K, r.n_assign, E, 0);
        CHECK(rc == 0, "Qwen fused launch k2=%d rc=%d", k2, rc);
        const std::vector<float> zgu = from_dev(d_zgu, (size_t)r.n_assign * GU);
        grade_fused("fused gate_up GEMV", r, a, K, GU, k2, gu, zgu, r.n_assign);

        /* the fused fold: gate = z[0, 640), up = z[640, 1280), svh from the one plane;
         * no clamp (Qwen's SwiGLU is plain silu(g) * u), the router weight folded in,
         * the down input rotation, the E4M3 emit */
        CUDA_OK(cudaMemset(d_msf, 0, sf_bytes));
        rc = ds4_exl3_moe_fold_fused_launch(d_zgu, d_sel, d_w, sgu.table, sd.table, K, MID, r.n_assign,
                                            0.0f, d_mq, d_msf, kbp, 0);
        CHECK(rc == 0, "Qwen fold rc=%d", rc);
        CUDA_OK(cudaDeviceSynchronize());
        std::vector<uint8_t> mq((size_t)r.n_assign * MID), msf(sf_bytes);
        CUDA_OK(cudaMemcpy(mq.data(), d_mq, mq.size(), cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(msf.data(), d_msf, sf_bytes, cudaMemcpyDeviceToHost));
        size_t n_bad = 0;
        for (int pair = 0; pair < r.n_assign; pair++) {
            const int e = r.selected[pair];
            std::vector<double> g(MID), u(MID), t(MID);
            for (int n = 0; n < MID; n++) { g[n] = zgu[(size_t)pair * GU + n]; u[n] = zgu[(size_t)pair * GU + MID + n]; }
            had_blocks(g);
            had_blocks(u);
            for (int n = 0; n < MID; n++) {
                const float yg = (float)(g[n] * exl3_f16_to_f32(gu[e].svh(K)[n]));
                const float yu = (float)(u[n] * exl3_f16_to_f32(gu[e].svh(K)[MID + n]));
                t[n] = (double)swiglu_host(yg, yu, r.weights[pair], 0.0f) * exl3_f16_to_f32(down[e].suh()[n]);
            }
            had_blocks(t);
            for (int grp = 0; grp < MID / 32; grp++) {
                const int se = (int)msf[pulsar_mx_sfoff(pair, grp, kbp)] - 127;
                const double ulp = exp2((double)se) * 0.125 * 2.0;
                for (int j = 0; j < 32; j++) {
                    const double got = (double)e4m3_to_f32(mq[(size_t)pair * MID + grp * 32 + j]) * exp2((double)se);
                    const double lim = fmax(ulp, fabs(t[grp * 32 + j]) * (1.0 / 8.0)) + 1e-30;
                    if (fabs(got - t[grp * 32 + j]) > lim) n_bad++;
                }
            }
        }
        CHECK(n_bad == 0, "Qwen fused fold K=%g: %zu mid values outside one E4M3 ulp", k2 / 2.0, n_bad);
        printf("  Qwen fused fold K=%g: %d pairs x %d, %zu outside one E4M3 ulp of the host chain\n",
               k2 / 2.0, r.n_assign, MID, n_bad);

        /* down 640 -> 2560: the K % 256 == 128 tail, on the fold's own mid */
        const acts am = restage_mid(r, mq, msf, MID, kbp);
        const block_mx_act_mmq *d_am = (const block_mx_act_mmq *)to_dev(am.raw);
        rc = ds4_exl3_moe_gemv_single_launch(sd.table, k2, d_am, d_ids, d_bounds, d_zd, K, MID, r.n_assign, E, 0);
        CHECK(rc == 0, "Qwen down launch k2=%d rc=%d", k2, rc);
        const std::vector<float> zd = from_dev(d_zd, (size_t)r.n_assign * K);
        double dworst = 0, dscale = 0;
        std::vector<double> wd;
        for (int col = 0; col < r.n_assign; col++) {
            const int pair = r.ids_dst[col], e = r.selected[pair];
            dequant_expert(down[e], MID, K, k2, wd);
            std::vector<double> z;
            ref_gemv(wd, MID, K, am.x.data() + (size_t)pair * MID, nullptr, false, z);
            double sc;
            dworst = fmax(dworst, max_rel(zd, z, (size_t)pair * K, K, &sc));
            dscale = fmax(dscale, sc);
        }
        CHECK(dworst < 2e-5, "Qwen down GEMV K=%g: max rel %.3e (limit 2e-5)", k2 / 2.0, dworst);
        printf("  Qwen down GEMV K=%g (K %% 256 == 128 tail): %d assignments x %d, max rel %.2e (|z| up to %.3g)\n",
               k2 / 2.0, r.n_assign, K, dworst, dscale);

        /* the sum, slot order, on the device's z_d */
        CUDA_OK(cudaMemset(d_nf, 0, sizeof(uint32_t)));
        rc = ds4_exl3_moe_sum_launch(d_out, d_zd, d_sel, sd.table, MID, K, n_slot, n_tok, d_nf, 0x501u, 0);
        CHECK(rc == 0, "Qwen sum rc=%d", rc);
        const std::vector<float> out = from_dev(d_out, (size_t)n_tok * K);
        double sworst = 0;
        for (int t = 0; t < n_tok; t++) {
            std::vector<double> acc(K, 0.0);
            for (int sl = 0; sl < n_slot; sl++) {
                const int pair = t * n_slot + sl, e = r.selected[pair];
                std::vector<double> z(zd.begin() + (size_t)pair * K, zd.begin() + (size_t)(pair + 1) * K);
                had_blocks(z);
                for (int o = 0; o < K; o++) acc[o] += z[o] * exl3_f16_to_f32(down[e].svh(MID)[o]);
            }
            double sc;
            sworst = fmax(sworst, max_rel(out, acc, (size_t)t * K, K, &sc));
        }
        CHECK(sworst < 1e-5, "Qwen sum K=%g: max rel %.3e (limit 1e-5)", k2 / 2.0, sworst);
        printf("  Qwen sum K=%g: %d tokens x %d over %d slots, max rel %.2e\n", k2 / 2.0, n_tok, K, n_slot, sworst);

        /* the row block on the tail: a prefill-like routing (40 tokens x 10 of 16
         * experts = 400 assignments, ~25 rows per expert) through a 640-in fused
         * slice (in = the down's K, so every row ends in the one-block pass): R = 1,
         * 4 and 16 bit-identical, 24 assignments graded against the host; the fused
         * launch at 2560-in with R = 4 as well. */
        {
            const routing rp = make_routing(40, 10, E);
            const acts ap = make_acts(rp, MID);
            std::vector<expert_blob> tg;
            for (int e = 0; e < E; e++) tg.push_back(make_expert(MID, GU, k2));
            dev_stack stg;
            if (upload_stack(tg, stg)) return 1;
            const block_mx_act_mmq *d_ap = (const block_mx_act_mmq *)to_dev(ap.raw);
            int32_t *d_pids = to_dev(rp.ids_dst), *d_pb = to_dev(rp.bounds);
            float *d_pz = nullptr;
            CUDA_OK(cudaMalloc((void **)&d_pz, (size_t)rp.n_assign * GU * sizeof(float)));
            std::vector<float> ref_z;
            size_t diff = 0;
            for (int Rb : {1, 4, 16}) {
                rc = ds4_exl3_moe_gemv_fused_launch_rows(stg.table, k2, d_ap, d_pids, d_pb, d_pz, GU, MID, rp.n_assign, E, Rb, 0);
                CHECK(rc == 0, "Qwen tail rows R=%d rc=%d", Rb, rc);
                const std::vector<float> z = from_dev(d_pz, (size_t)rp.n_assign * GU);
                if (Rb == 1) { ref_z = z; continue; }
                for (size_t i = 0; i < z.size(); i++) diff += z[i] != ref_z[i];
            }
            CHECK(diff == 0, "Qwen tail: R=4/16 differ from R=1 in %zu outputs", diff);
            grade_fused("640-in fused slice on the tail (R=1)", rp, ap, MID, GU, k2, tg, ref_z, 24);
            printf("  Qwen tail row block R=1/4/16 on %d assignments: %s\n", rp.n_assign, diff ? "DIFFER" : "bit-identical");
            /* the 2560-in fused launch at 400 assignments takes R = 4 */
            const acts ak = make_acts(rp, K);
            const block_mx_act_mmq *d_ak = (const block_mx_act_mmq *)to_dev(ak.raw);
            float *d_kz = nullptr;
            CUDA_OK(cudaMalloc((void **)&d_kz, (size_t)rp.n_assign * GU * sizeof(float)));
            rc = ds4_exl3_moe_gemv_fused_launch(sgu.table, k2, d_ak, d_pids, d_pb, d_kz, GU, K, rp.n_assign, E, 0);
            CHECK(rc == 0, "Qwen fused at 400 assignments rc=%d", rc);
            grade_fused("fused gate_up at 400 assignments (R=4 rows)", rp, ak, K, GU, k2, gu, from_dev(d_kz, (size_t)rp.n_assign * GU), 24);
            /* the down at 400 assignments takes R = 4 on the tail */
            const acts apd = make_acts(rp, MID);
            const block_mx_act_mmq *d_apd = (const block_mx_act_mmq *)to_dev(apd.raw);
            float *d_pd = nullptr;
            CUDA_OK(cudaMalloc((void **)&d_pd, (size_t)rp.n_assign * K * sizeof(float)));
            rc = ds4_exl3_moe_gemv_single_launch(sd.table, k2, d_apd, d_pids, d_pb, d_pd, K, MID, rp.n_assign, E, 0);
            CHECK(rc == 0, "Qwen down R=4 rc=%d", rc);
            const std::vector<float> pd = from_dev(d_pd, (size_t)rp.n_assign * K);
            double w4 = 0;
            for (int col = 0; col < 24; col++) {
                const int pair = rp.ids_dst[col], e = rp.selected[pair];
                dequant_expert(down[e], MID, K, k2, wd);
                std::vector<double> z;
                ref_gemv(wd, MID, K, apd.x.data() + (size_t)rp.ids_tok[col] * MID, nullptr, false, z);   /* make_acts rows are per TOKEN */
                double sc;
                w4 = fmax(w4, max_rel(pd, z, (size_t)pair * K, K, &sc));
            }
            CHECK(w4 < 2e-5, "Qwen down at 400 assignments (R=4): max rel %.3e", w4);
            printf("  Qwen down GEMV K=%g at %d assignments (R=4 rows, tail): max rel %.2e on 24\n", k2 / 2.0, rp.n_assign, w4);
            cudaFree((void *)d_ap); cudaFree((void *)d_ak); cudaFree((void *)d_apd); cudaFree(d_pids); cudaFree(d_pb);
            cudaFree(d_pz); cudaFree(d_kz); cudaFree(d_pd);
            cudaFree(stg.arena); cudaFree((void *)stg.table);
        }

        if (k2 == 10) {
            /* mutation: one flipped trellis bit of expert 0's down must move z_d */
            const bool expert0_used = r.bounds[1] > r.bounds[0];
            CHECK(expert0_used, "Qwen mutation fixture: expert 0 unrouted, the mutation would prove nothing");
            uint8_t byte = 0;
            CUDA_OK(cudaMemcpy(&byte, sd.arena + 777, 1, cudaMemcpyDeviceToHost));
            byte ^= 0x04u;
            CUDA_OK(cudaMemcpy(sd.arena + 777, &byte, 1, cudaMemcpyHostToDevice));
            rc = ds4_exl3_moe_gemv_single_launch(sd.table, k2, d_am, d_ids, d_bounds, d_zd, K, MID, r.n_assign, E, 0);
            const std::vector<float> zd2 = from_dev(d_zd, (size_t)r.n_assign * K);
            size_t moved = 0;
            for (size_t i = 0; i < zd.size(); i++) moved += zd[i] != zd2[i];
            CHECK(moved > 0, "Qwen mutation: a flipped bit in expert 0's down did not move z_d");
            printf("  Qwen mutation: one flipped down bit moved %zu outputs\n", moved);
            /* refusals: a K that is not a 128 multiple; a rate a kind does not instantiate */
            CHECK(ds4_exl3_moe_gemv_single_launch(sd.table, k2, d_am, d_ids, d_bounds, d_zd, K, 704, r.n_assign, E, 0) != 0,
                  "K=704 was accepted");
            CHECK(ds4_exl3_moe_gemv_single_launch(sd.table, 7, d_am, d_ids, d_bounds, d_zd, K, MID, r.n_assign, E, 0) != 0,
                  "k2=7 was accepted");
            CHECK(ds4_exl3_moe_gemv_fused_launch(sgu.table, 6, d_act, d_ids, d_bounds, d_zgu, GU, K, r.n_assign, E, 0) != 0,
                  "fused at K=3 was accepted (no container carries it)");
            CHECK(ds4_exl3_gemv_rate_supported(EXL3_ARM_GATE_UP_FUSED, 8) && ds4_exl3_gemv_rate_supported(EXL3_ARM_DOWN, 10) &&
                  ds4_exl3_gemv_rate_supported(EXL3_ARM_PAIR, 8) && !ds4_exl3_gemv_rate_supported(EXL3_ARM_GATE_UP_FUSED, 6) &&
                  ds4_exl3_gemv_rate_supported(EXL3_ARM_PAIR, 6) && !ds4_exl3_gemv_rate_supported(EXL3_ARM_DOWN, 7) &&
                  ds4_exl3_gemv_rate_supported(EXL3_ARM_PAIR, 12) && !ds4_exl3_gemv_rate_supported(EXL3_ARM_GATE_UP_FUSED, 12) &&
                  !ds4_exl3_gemv_rate_supported(EXL3_ARM_DENSE, 8),
                  "the per-kind rate table");
            printf("  refusals: K=704, k2=7, fused at K=3 -- refused; the per-kind rate table holds\n");
        }
        cudaFree((void *)d_am);
        cudaFree(sgu.arena); cudaFree((void *)sgu.table); cudaFree(sd.arena); cudaFree((void *)sd.table);
    }
    cudaFree(d_ids); cudaFree(d_bounds); cudaFree(d_sel); cudaFree(d_w); cudaFree((void *)d_act);
    cudaFree(d_zgu); cudaFree(d_zd); cudaFree(d_out); cudaFree(d_nf); cudaFree(d_mq); cudaFree(d_msf);
    return 0;
}
