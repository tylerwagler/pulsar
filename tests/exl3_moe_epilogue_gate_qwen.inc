/* exl3_moe_epilogue_gate, L251 section: Qwen3.8-Flash-Next's routed experts -- 512
 * experts x top-10, in 2560, mid 640, gate_up FUSED (one [2560 -> 1280] slice
 * per expert, gate rows then up rows, as the container carries it) and down
 * 640 -> 2560, rates K = 4 and 5.  The down's K = 640 = 5 x 128 is the arm's
 * K % 256 == 128 tail: the last staging pass of every row is one Hadamard
 * block.  Graded like the V4.1 sections (host authority + double), plus a
 * 400-assignment routing on the tail, a mutation and the refusals.  Included by
 * tests/exl3_moe_epilogue_gate.cu and run AFTER the V4.1 sections, so their random
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
    CHECK(worst < 1e-3, "Qwen %s K=%g: max rel %.3e vs the host authority (limit 1e-3)", what, k2 / 2.0, worst);
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

static int qwen_section(void) {
    const int K = 2560, MID = 640, GU = 2 * MID, E = 16, n_tok = 3, n_slot = 10;
    printf("exl3-moe-epilogue-gate (L251 Qwen): fused gate_up %d -> %d, down %d -> %d, top-%d of %d experts, K = 4 and 5\n",
           K, GU, MID, K, n_slot, E);
    const routing r = make_routing(n_tok, n_slot, E);
    const acts a = make_acts(r, K);
    int32_t *d_ids = to_dev(r.ids_dst), *d_bounds = to_dev(r.bounds), *d_sel = to_dev(r.selected);
    int32_t *d_src = to_dev(r.ids_tok);
    float *d_w = to_dev(r.weights);
    const exl3_moe_act d_act = upload_act(a);
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
        int rc = gemv_one(EXL3_ARM_GATE_UP_FUSED, sgu.table, k2, d_act, d_ids, d_src, d_bounds, d_zgu, GU, K, r.n_assign, E);
        CHECK(rc == 0, "Qwen fused launch k2=%d rc=%d", k2, rc);
        const std::vector<float> zgu = from_dev(d_zgu, (size_t)r.n_assign * GU);
        grade_fused("fused gate_up", r, a, K, GU, k2, gu, zgu, r.n_assign);

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

        /* down 640 -> 2560: the K % 256 == 128 tail, on the fold's own mid, read in its slot (row = pair) */
        const acts am = decode_slot(mq, msf, r.n_assign, MID, kbp);
        const exl3_moe_act d_am{EXL3_MOE_ACT_E4M3_SLOT, d_mq, d_msf, kbp};
        rc = gemv_one(EXL3_ARM_DOWN, sd.table, k2, d_am, d_ids, d_ids, d_bounds, d_zd, K, MID, r.n_assign, E);
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
        CHECK(dworst < 2e-5, "Qwen down K=%g: max rel %.3e (limit 2e-5)", k2 / 2.0, dworst);
        printf("  Qwen down K=%g (K %% 256 == 128 tail): %d assignments x %d, max rel %.2e (|z| up to %.3g)\n",
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

        /* a wide decode routing on the tail: 40 tokens x 10 of 16 experts = 400 assignments (~25 rows per
         * expert: two m16 tiles each) through a 640-in fused slice (in = the down's K) and the 2560-in fused
         * slice, 24 assignments graded against the host; and the down at 400 assignments. */
        {
            const routing rp = make_routing(40, 10, E);
            const acts ap = make_acts(rp, MID);
            std::vector<expert_blob> tg;
            for (int e = 0; e < E; e++) tg.push_back(make_expert(MID, GU, k2));
            dev_stack stg;
            if (upload_stack(tg, stg)) return 1;
            const exl3_moe_act d_ap = upload_act(ap);
            int32_t *d_pids = to_dev(rp.ids_dst), *d_pb = to_dev(rp.bounds), *d_ptok = to_dev(rp.ids_tok);
            float *d_pz = nullptr;
            CUDA_OK(cudaMalloc((void **)&d_pz, (size_t)rp.n_assign * GU * sizeof(float)));
            rc = gemv_one(EXL3_ARM_GATE_UP_FUSED, stg.table, k2, d_ap, d_pids, d_ptok, d_pb, d_pz, GU, MID, rp.n_assign, E);
            CHECK(rc == 0, "Qwen tail fused at 400 assignments rc=%d", rc);
            grade_fused("640-in fused slice on the tail, 400 assignments", rp, ap, MID, GU, k2, tg,
                        from_dev(d_pz, (size_t)rp.n_assign * GU), 24);
            const acts ak = make_acts(rp, K);
            const exl3_moe_act d_ak = upload_act(ak);
            float *d_kz = nullptr;
            CUDA_OK(cudaMalloc((void **)&d_kz, (size_t)rp.n_assign * GU * sizeof(float)));
            rc = gemv_one(EXL3_ARM_GATE_UP_FUSED, sgu.table, k2, d_ak, d_pids, d_ptok, d_pb, d_kz, GU, K, rp.n_assign, E);
            CHECK(rc == 0, "Qwen fused at 400 assignments rc=%d", rc);
            grade_fused("fused gate_up at 400 assignments", rp, ak, K, GU, k2, gu, from_dev(d_kz, (size_t)rp.n_assign * GU), 24);
            const acts apd = make_acts(rp, MID);
            const exl3_moe_act d_apd = upload_act(apd);
            float *d_pd = nullptr;
            CUDA_OK(cudaMalloc((void **)&d_pd, (size_t)rp.n_assign * K * sizeof(float)));
            rc = gemv_one(EXL3_ARM_DOWN, sd.table, k2, d_apd, d_pids, d_ptok, d_pb, d_pd, K, MID, rp.n_assign, E);
            CHECK(rc == 0, "Qwen down at 400 assignments rc=%d", rc);
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
            CHECK(w4 < 2e-5, "Qwen down at 400 assignments: max rel %.3e", w4);
            printf("  Qwen down K=%g at %d assignments (tail): max rel %.2e on 24\n", k2 / 2.0, rp.n_assign, w4);
            free_act(d_ap); free_act(d_ak); free_act(d_apd); cudaFree(d_pids); cudaFree(d_pb); cudaFree(d_ptok);
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
            rc = gemv_one(EXL3_ARM_DOWN, sd.table, k2, d_am, d_ids, d_ids, d_bounds, d_zd, K, MID, r.n_assign, E);
            const std::vector<float> zd2 = from_dev(d_zd, (size_t)r.n_assign * K);
            size_t moved = 0;
            for (size_t i = 0; i < zd.size(); i++) moved += zd[i] != zd2[i];
            CHECK(moved > 0, "Qwen mutation: a flipped bit in expert 0's down did not move z_d");
            printf("  Qwen mutation: one flipped down bit moved %zu outputs\n", moved);
            /* refusals: a K that is not a 128 multiple; a rate a kind does not instantiate */
            CHECK(gemv_one(EXL3_ARM_DOWN, sd.table, k2, d_am, d_ids, d_ids, d_bounds, d_zd, K, 704, r.n_assign, E) != 0,
                  "K=704 was accepted");
            CHECK(gemv_one(EXL3_ARM_DOWN, sd.table, 7, d_am, d_ids, d_ids, d_bounds, d_zd, K, MID, r.n_assign, E) != 0,
                  "k2=7 was accepted");
            CHECK(gemv_one(EXL3_ARM_GATE_UP_FUSED, sgu.table, 6, d_act, d_ids, d_src, d_bounds, d_zgu, GU, K, r.n_assign, E) != 0,
                  "fused at K=3 was accepted (no container carries it)");
            CHECK(exl3_arm_has_rate(EXL3_ARM_GATE_UP_FUSED, 8) && exl3_arm_has_rate(EXL3_ARM_DOWN, 10) &&
                  exl3_arm_has_rate(EXL3_ARM_PAIR, 8) && !exl3_arm_has_rate(EXL3_ARM_GATE_UP_FUSED, 6) &&
                  exl3_arm_has_rate(EXL3_ARM_PAIR, 6) && !exl3_arm_has_rate(EXL3_ARM_DOWN, 7) &&
                  exl3_arm_has_rate(EXL3_ARM_PAIR, 12) && !exl3_arm_has_rate(EXL3_ARM_GATE_UP_FUSED, 12),
                  "the per-kind rate table");
            printf("  refusals: K=704, k2=7, fused at K=3 -- refused; the per-kind rate table holds\n");
        }
        cudaFree(sgu.arena); cudaFree((void *)sgu.table); cudaFree(sd.arena); cudaFree((void *)sd.table);
    }
    cudaFree(d_ids); cudaFree(d_bounds); cudaFree(d_sel); cudaFree(d_w); free_act(d_act); cudaFree(d_src);
    cudaFree(d_zgu); cudaFree(d_zd); cudaFree(d_out); cudaFree(d_nf); cudaFree(d_mq); cudaFree(d_msf);
    return 0;
}
