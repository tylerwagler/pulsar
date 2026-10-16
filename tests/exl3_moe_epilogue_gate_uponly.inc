/* exl3_moe_epilogue_gate, L296 section: UP-ONLY (ungated) relu^2 experts -- Nemotron-H's latent MoE shape, one up
 * slice [1024 -> 2688] and one down slice [2688 -> 1024] per expert, no gate, rates K = 4 and 5.  Graded like the
 * other sections against the host authority (exl3_trellis.h + double):
 *
 *   U1. the up projection, launched as ONE rotated slice (the front door's up-only form): z vs the host, and z ==
 *       the up half of a gate + up PAIR launch over the same up table, bit for bit (the same kernel arithmetic);
 *   U2. the up-only fold (ds4_exl3_moe_fold_up_launch): relu(svh_u * H(z))^2 * w, the down input rotation, emitted
 *       bf16 (the rows lane's mid) and E4M3 (the slot) -- vs the host chain within one bf16 / E4M3 ulp; and fed the
 *       decode arm's split partials == fed the summed z, bit for bit;
 *   U3. the down on the fold's own bf16 mid and the slot-ordered sum vs the host;
 *   U4. a mutation (a flipped svh_u sign must move the mid) and the refusals (mid % 128, no up table).
 * Included by tests/exl3_moe_epilogue_gate.cu and run LAST, so the earlier sections' draws do not move. */

static double bf16_to_double(uint16_t h) {
    const uint32_t u = (uint32_t)h << 16;
    float f;
    memcpy(&f, &u, 4);
    return (double)f;
}
static float relu2_host(float u, float wv) {
    const float r = u <= 0.0f ? 0.0f : u;
    return r * r * wv;
}

static int uponly_section(void) {
    const int K = 1024, MID = 2688, E = 12, n_tok = 3, n_slot = 6;
    printf("exl3-moe-epilogue-gate (L296 up-only relu^2): up %d -> %d, down %d -> %d, top-%d of %d experts, K = 4 and 5\n",
           K, MID, MID, K, n_slot, E);
    const routing r = make_routing(n_tok, n_slot, E);
    const acts a = make_acts(r, K);
    int32_t *d_ids = upload_ids(r.ids_dst), *d_bounds = upload_ids(r.bounds), *d_sel = upload_ids(r.selected);
    int32_t *d_src = upload_ids(r.ids_tok);
    float *d_w = to_dev(r.weights);
    const exl3_moe_act d_act = upload_act(a);
    float *d_zu = nullptr, *d_zg = nullptr, *d_zp = nullptr, *d_zd = nullptr, *d_out = nullptr;
    uint32_t *d_nf = nullptr;
    uint16_t *d_mb = nullptr;
    CUDA_OK(cudaMalloc((void **)&d_zu, (size_t)r.n_assign * MID * sizeof(float)));
    CUDA_OK(cudaMalloc((void **)&d_zg, (size_t)r.n_assign * MID * sizeof(float)));
    CUDA_OK(cudaMalloc((void **)&d_zp, (size_t)r.n_assign * MID * sizeof(float)));
    CUDA_OK(cudaMalloc((void **)&d_zd, (size_t)r.n_assign * K * sizeof(float)));
    CUDA_OK(cudaMalloc((void **)&d_out, (size_t)n_tok * K * sizeof(float)));
    CUDA_OK(cudaMalloc((void **)&d_nf, sizeof(uint32_t)));
    CUDA_OK(cudaMalloc((void **)&d_mb, (size_t)r.n_assign * MID * 2));
    const int kbp = pulsar_mx_kbp(MID);
    const size_t sf_bytes = pulsar_mx_sf_slab_bytes(r.n_assign, kbp);
    uint8_t *d_mq = nullptr, *d_msf = nullptr;
    CUDA_OK(cudaMalloc((void **)&d_mq, (size_t)r.n_assign * MID));
    CUDA_OK(cudaMalloc((void **)&d_msf, sf_bytes));

    for (int k2 : {8, 10}) {
        std::vector<expert_blob> up, gate, down;
        for (int e = 0; e < E; e++) {
            up.push_back(make_expert(K, MID, k2));
            gate.push_back(make_expert(K, MID, k2));
            down.push_back(make_expert(MID, K, k2));
        }
        dev_stack su, sg, sd;
        if (upload_stack(up, su) || upload_stack(gate, sg) || upload_stack(down, sd)) return 1;

        /* U1: the up as one rotated slice (the front door projects an up-only stack as a fused slice is) */
        exl3_moe_partials keep{};
        int rc = gemv_one(EXL3_ARM_GATE_UP_FUSED, su.table, k2, d_act, d_ids, d_src, d_bounds, d_zu, MID, K, r.n_assign,
                          E, &keep);
        CHECK(rc == 0, "up-only up launch k2=%d rc=%d", k2, rc);
        const std::vector<float> zu = from_dev(d_zu, (size_t)r.n_assign * MID);
        grade_fused("up-only up", r, a, K, MID, k2, up, zu, r.n_assign);
        rc = gemv_pair(sg.table, su.table, k2, d_act, d_ids, d_src, d_bounds, d_zg, d_zp, MID, K, r.n_assign, E);
        CHECK(rc == 0, "pair launch k2=%d rc=%d", k2, rc);
        const std::vector<float> zpair = from_dev(d_zp, (size_t)r.n_assign * MID);
        CHECK(memcmp(zu.data(), zpair.data(), zu.size() * 4) == 0,
              "K=%g: the up-only up is not the pair's up bit for bit", k2 / 2.0);
        printf("  up-only up K=%g == a gate + up pair's up over the same table, bit for bit\n", k2 / 2.0);

        /* U2: the up-only fold, bf16 mid (the rows lane) and the E4M3 slot */
        rc = ds4_exl3_moe_fold_up_launch(d_zu, d_sel, d_w, su.table, sd.table, K, MID, r.n_assign, nullptr, nullptr, 0, 0,
                                         d_mb);
        CUDA_OK(cudaMemset(d_msf, 0, sf_bytes));
        rc |= ds4_exl3_moe_fold_up_launch(d_zu, d_sel, d_w, su.table, sd.table, K, MID, r.n_assign, d_mq, d_msf, kbp, 0);
        CHECK(rc == 0, "up-only fold rc=%d", rc);
        CUDA_OK(cudaDeviceSynchronize());
        std::vector<uint16_t> mb((size_t)r.n_assign * MID);
        std::vector<uint8_t> mq((size_t)r.n_assign * MID), msf(sf_bytes);
        CUDA_OK(cudaMemcpy(mb.data(), d_mb, mb.size() * 2, cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(mq.data(), d_mq, mq.size(), cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(msf.data(), d_msf, sf_bytes, cudaMemcpyDeviceToHost));
        size_t bad16 = 0, bad8 = 0, nonzero = 0;
        double worst16 = 0;
        for (int pair = 0; pair < r.n_assign; pair++) {
            const int e = r.selected[pair];
            std::vector<double> u(MID), t(MID);
            for (int n = 0; n < MID; n++) u[n] = zu[(size_t)pair * MID + n];
            had_blocks(u);
            for (int n = 0; n < MID; n++) {
                const float yu = (float)(u[n] * exl3_f16_to_f32(up[e].svh(K)[n]));
                t[n] = (double)relu2_host(yu, r.weights[pair]) * exl3_f16_to_f32(down[e].suh()[n]);
            }
            had_blocks(t);
            double scale = 0;
            for (int n = 0; n < MID; n++) scale = fmax(scale, fabs(t[n]));
            for (int n = 0; n < MID; n++) {
                const double got = bf16_to_double(mb[(size_t)pair * MID + n]), want = t[n];
                nonzero += got != 0.0;
                /* one bf16 ulp of the value, plus the f32 Hadamard's rounding at the row's scale */
                const double lim = fabs(want) * (1.0 / 128.0) + scale * 1e-6;
                worst16 = fmax(worst16, fabs(got - want) / (scale > 0 ? scale : 1.0));
                if (fabs(got - want) > lim) bad16++;
            }
            for (int grp = 0; grp < MID / 32; grp++) {
                const int se = (int)msf[pulsar_mx_sfoff(pair, grp, kbp)] - 127;
                const double ulp = exp2((double)se) * 0.125 * 2.0;
                for (int j = 0; j < 32; j++) {
                    const double got = (double)e4m3_to_f32(mq[(size_t)pair * MID + grp * 32 + j]) * exp2((double)se);
                    const double lim = fmax(ulp, fabs(t[grp * 32 + j]) * (1.0 / 8.0)) + 1e-30;
                    if (fabs(got - t[grp * 32 + j]) > lim) bad8++;
                }
            }
        }
        CHECK(bad16 == 0 && bad8 == 0 && nonzero > (size_t)r.n_assign * MID / 2,
              "up-only fold K=%g: %zu bf16 / %zu E4M3 values outside one ulp of the host chain, %zu nonzero", k2 / 2.0,
              bad16, bad8, nonzero);
        printf("  up-only fold K=%g: %d pairs x %d, %zu bf16 / %zu E4M3 outside one ulp of the host relu^2 chain (max "
               "rel %.2e)\n", k2 / 2.0, r.n_assign, MID, bad16, bad8, worst16);
        /* the split partials: fed the decode arm's partials == fed their sum, bit for bit */
        {
            uint16_t *d_mb2 = nullptr;
            CUDA_OK(cudaMalloc((void **)&d_mb2, mb.size() * 2));
            rc = gemv_one(EXL3_ARM_GATE_UP_FUSED, su.table, k2, d_act, d_ids, d_src, d_bounds, d_zu, MID, K, r.n_assign,
                          E, &keep);
            rc |= ds4_exl3_moe_fold_up_launch(keep.z[0], d_sel, d_w, su.table, sd.table, K, MID, r.n_assign, nullptr,
                                              nullptr, 0, 0, d_mb2, keep.splits, keep.split_stride);
            CUDA_OK(cudaDeviceSynchronize());
            std::vector<uint16_t> mb2(mb.size());
            CUDA_OK(cudaMemcpy(mb2.data(), d_mb2, mb2.size() * 2, cudaMemcpyDeviceToHost));
            /* the decode arm's split count is a function of K alone: at the latent's K = 1024 it is one split, so
             * this proves the fold reads the arm's own plane exactly (the multi-split sum is the V4.1 section's) */
            CHECK(rc == 0 && keep.splits >= 1 && mb == mb2,
                  "up-only fold K=%g: the decode arm's partials (%d) and the summed z emit different mids (rc=%d)",
                  k2 / 2.0, keep.splits, rc);
            printf("  up-only fold K=%g: fed the decode arm's partials (%d split at in %d) == fed the summed z, bit for "
                   "bit\n", k2 / 2.0, keep.splits, K);
            cudaFree(d_mb2);
        }

        /* U3: the down on the fold's own bf16 mid (row = pair), then the slot-ordered sum */
        const exl3_moe_act d_am{EXL3_MOE_ACT_BF16_ROWS, d_mb, nullptr, 0};
        rc = gemv_one(EXL3_ARM_DOWN, sd.table, k2, d_am, d_ids, d_ids, d_bounds, d_zd, K, MID, r.n_assign, E);
        CHECK(rc == 0, "up-only down launch k2=%d rc=%d", k2, rc);
        const std::vector<float> zd = from_dev(d_zd, (size_t)r.n_assign * K);
        double dworst = 0;
        std::vector<double> wd;
        for (int col = 0; col < r.n_assign; col++) {
            const int pair = r.ids_dst[col], e = r.selected[pair];
            dequant_expert(down[e], MID, K, k2, wd);
            std::vector<double> x(MID), z;
            for (int n = 0; n < MID; n++) x[n] = bf16_to_double(mb[(size_t)pair * MID + n]);
            ref_gemv(wd, MID, K, x.data(), nullptr, false, z);
            double sc;
            dworst = fmax(dworst, max_rel(zd, z, (size_t)pair * K, K, &sc));
        }
        /* the down's fp16 operand plane rounds a bf16 mid to 11 significant bits per (row, 128-block): 1e-3 */
        CHECK(dworst < 1e-3, "up-only down K=%g: max rel %.3e (limit 1e-3)", k2 / 2.0, dworst);
        CUDA_OK(cudaMemset(d_nf, 0, sizeof(uint32_t)));
        rc = ds4_exl3_moe_sum_launch(d_out, d_zd, d_sel, sd.table, MID, K, n_slot, n_tok, d_nf, 0x601u, 0);
        CHECK(rc == 0, "up-only sum rc=%d", rc);
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
        CHECK(sworst < 1e-5, "up-only sum K=%g: max rel %.3e (limit 1e-5)", k2 / 2.0, sworst);
        printf("  up-only down K=%g on the bf16 mid: max rel %.2e; sum over %d slots: max rel %.2e\n", k2 / 2.0, dworst,
               n_slot, sworst);

        if (k2 == 8) {
            /* U4: a mutation -- expert e0's svh_u sign flipped on its first 128 block moves that block's mid */
            const int e0 = r.selected[0];
            std::vector<uint8_t> blob = up[e0].bytes;
            uint16_t *sv = (uint16_t *)(blob.data() + up[e0].trellis) + K;
            for (int n = 0; n < 128; n++) sv[n] ^= 0x8000u;
            CUDA_OK(cudaMemcpy(su.arena + (size_t)e0 * su.stride, blob.data(), su.stride, cudaMemcpyHostToDevice));
            rc = gemv_one(EXL3_ARM_GATE_UP_FUSED, su.table, k2, d_act, d_ids, d_src, d_bounds, d_zu, MID, K, r.n_assign, E);
            rc |= ds4_exl3_moe_fold_up_launch(d_zu, d_sel, d_w, su.table, sd.table, K, MID, r.n_assign, nullptr, nullptr,
                                              0, 0, d_mb);
            CUDA_OK(cudaDeviceSynchronize());
            std::vector<uint16_t> mm(mb.size());
            CUDA_OK(cudaMemcpy(mm.data(), d_mb, mm.size() * 2, cudaMemcpyDeviceToHost));
            size_t moved = 0;
            for (size_t i = 0; i < mm.size(); i++) moved += mm[i] != mb[i];
            CHECK(rc == 0 && moved > 0, "up-only mutation: a flipped svh_u block did not move the mid");
            printf("  up-only mutation: a flipped svh_u block moved %zu mid values\n", moved);
            /* refusals: mid not a 128 multiple; no up table; no output */
            CHECK(ds4_exl3_moe_fold_up_launch(d_zu, d_sel, d_w, su.table, sd.table, K, 2600, r.n_assign, nullptr, nullptr,
                                              0, 0, d_mb) != 0, "up-only fold at mid 2600 was accepted");
            CHECK(ds4_exl3_moe_fold_up_launch(d_zu, d_sel, d_w, nullptr, sd.table, K, MID, r.n_assign, nullptr, nullptr, 0,
                                              0, d_mb) != 0, "up-only fold without an up table was accepted");
            CHECK(ds4_exl3_moe_fold_up_launch(d_zu, d_sel, d_w, su.table, sd.table, K, MID, r.n_assign, nullptr, nullptr, 0,
                                              0, nullptr) != 0, "up-only fold without an output was accepted");
            printf("  refusals: mid 2600, no up table, no output -- refused\n");
        }
        cudaFree(su.arena); cudaFree((void *)su.table); cudaFree(sg.arena); cudaFree((void *)sg.table);
        cudaFree(sd.arena); cudaFree((void *)sd.table);
    }
    cudaFree(d_ids); cudaFree(d_bounds); cudaFree(d_sel); cudaFree(d_src); cudaFree(d_w); free_act(d_act);
    cudaFree(d_zu); cudaFree(d_zg); cudaFree(d_zp); cudaFree(d_zd); cudaFree(d_out); cudaFree(d_nf); cudaFree(d_mb);
    cudaFree(d_mq); cudaFree(d_msf);
    return 0;
}
