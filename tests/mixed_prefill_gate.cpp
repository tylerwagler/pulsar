/* plan-34 phase-2 increment 3 — K-row single-bank PREFILL through the mixed entry.
 *
 * Routes a K-row prefill chunk (one bank, positions p0..p0+K-1) through
 * pulsar_session_decode_mixed and checks it is (1) COHERENT vs classic chunked
 * prefill, (2) correct across the K>ratio compressor read-after-write boundary,
 * and (3) TENSOR-CORE FAST (prefix=0 => no forced-custom kernel). NO decode
 * co-scheduling (that + the decode/prefill split + neutrality is inc 4).
 *
 * ORACLE = output-token coherence (NOT byte-identity): the mixed prefill resumes
 * with the accepted last-ulp warm-continuation KV delta, so greedy tokens are
 * compared, not logit bytes (same class as fork-gate P5).
 *
 * Shape: classic-prefill a small FIRST chunk [0,c0) to lift the bank frontier off
 * 0 (step_begin rejects pos-0), then feed [c0,c0+K) as ONE K-row mixed run; then
 * greedy-decode NGEN tokens (1-row decode_mixed steps) and compare to a fully
 * classic-prefilled reference decoded the same way.
 *
 * Run under PULSAR_MSEQ_BANKS>=1, pack on/off x idx-fp4 on/off. GPU discipline.
 *   usage: PULSAR_MSEQ_BANKS=2 ./tests/mixed_prefill_gate MODEL
 */
#include "pulsar.h"
#include "pulsar_engine_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static pulsar_engine *g_e;
static pulsar_tokens g_toks;
static int g_fail;
#define NGEN 24
#define C0   128            /* first (classic) chunk: lifts frontier off 0, ratio-aligned */

static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static char *read_file(const char *p, size_t *n){ FILE *f=fopen(p,"rb"); if(!f)return NULL;
    fseek(f,0,SEEK_END); long s=ftell(f); fseek(f,0,SEEK_SET); char *b=(char *)malloc(s+1);
    if(!b||fread(b,1,s,f)!=(size_t)s){fclose(f);free(b);return NULL;} fclose(f); b[s]=0; if(n)*n=s; return b; }

/* greedy-decode NGEN tokens on bank 0, continuing from a frontier at F with the
 * prefill's first predicted token t0 already in out[0]; uses 1-row decode_mixed. */
static bool decode_cont(pulsar_session *s, int F, int t0, int *out){
    out[0]=t0;
    const int vocab=(int)PULSAR_N_VOCAB;
    float *lg = (float *)malloc((size_t)vocab*sizeof(float));
    char e[256]; bool ok=true;
    for(int i=1; i<NGEN && ok; i++){
        pulsar_multiseq_req r={.bank=0,.pos=F+i-1,.token=out[i-1]};
        uint32_t nr=0;
        if(pulsar_session_decode_mixed(s,&r,1,lg,vocab,&nr,0u,e,sizeof e)!=0){ fprintf(stderr,"cont step %d: %s\n",i,e); ok=false; break; }
        out[i]=(int)argmax_f32(lg,(uint64_t)vocab);
    }
    free(lg); return ok;
}

/* classic RESUME reference: prefill [0,F) via pulsar_session_sync (the second sync
 * resumes at c0>0 = same decode-attention kernel as the mixed path), copy the
 * last-position full-vocab logits, then decode NGEN. */
static bool classic_stream(int c0, int F, int *out, float *out_lg){
    pulsar_session *s=NULL; if(pulsar_session_create(&s,g_e,4096)!=0) return false;
    pulsar_gpu_graph *g=&s->graph; char e[256]; bool ok=true;
    if(g->banks.n_banks && !gpu_graph_bank_repoint(g,0)){ pulsar_session_free(s); return false; }
    pulsar_session_invalidate(s);
    pulsar_tokens p0={.v=g_toks.v,.len=c0,.cap=c0};
    if(pulsar_session_sync(s,&p0,e,sizeof e)!=0){ fprintf(stderr,"classic first-chunk: %s\n",e); ok=false; }
    pulsar_tokens p={.v=g_toks.v,.len=F,.cap=F};           /* RESUMES at c0 (pos0>0) */
    if(ok && pulsar_session_sync(s,&p,e,sizeof e)!=0){ fprintf(stderr,"classic resume: %s\n",e); ok=false; }
    if(ok){ gpu_graph_bank_counters_capture(g,0);
            pulsar_session_copy_logits(s,out_lg,(int)PULSAR_N_VOCAB);   /* last-position logits */
            ok=decode_cont(s,F,pulsar_session_argmax(s),out); }
    pulsar_session_free(s); return ok;
}

/* mixed: classic [0,c0), then K-row mixed run [c0,c0+K); decode NGEN. *secs = the
 * timed mixed-run seconds when non-NULL. */
static bool mixed_stream(int c0, int K, int *out, double *secs, float *out_lg){
    pulsar_session *s=NULL; if(pulsar_session_create(&s,g_e,4096)!=0) return false;
    pulsar_gpu_graph *g=&s->graph; char e[256]; bool ok=true;
    if(g->banks.n_banks && !gpu_graph_bank_repoint(g,0)){ pulsar_session_free(s); return false; }
    pulsar_session_invalidate(s);
    pulsar_tokens p={.v=g_toks.v,.len=c0,.cap=c0};
    if(pulsar_session_sync(s,&p,e,sizeof e)!=0){ fprintf(stderr,"mixed first-chunk sync: %s\n",e); ok=false; }
    if(ok) gpu_graph_bank_counters_capture(g,0);
    const int vocab=(int)PULSAR_N_VOCAB;
    float *lg = (float *)malloc((size_t)vocab*sizeof(float));    /* 1 run => 1 logit row */
    pulsar_multiseq_req *rq = (pulsar_multiseq_req *)malloc((size_t)K*sizeof(*rq));
    for(int j=0;j<K;j++){ rq[j].bank=0; rq[j].pos=c0+j; rq[j].token=g_toks.v[c0+j]; }
    uint32_t nr=0;
    double t0=secs?now_s():0.0;
    if(ok && pulsar_session_decode_mixed(s,rq,(uint32_t)K,lg,vocab,&nr,0u,e,sizeof e)!=0){ fprintf(stderr,"mixed K-run: %s\n",e); ok=false; }
    if(secs) *secs=now_s()-t0;
    if(ok && nr!=1){ fprintf(stderr,"mixed run n_rows=%u expected 1\n",nr); ok=false; }
    if(ok && out_lg) memcpy(out_lg,lg,(size_t)vocab*sizeof(float));   /* last-position logits */
    if(ok) ok=decode_cont(s,c0+K,(int)argmax_f32(lg,(uint64_t)vocab),out);
    free(lg); free(rq); pulsar_session_free(s); return ok;
}

/* classic prefill of K tokens from a c0 frontier (for the speed baseline). */
static bool classic_prefill_time(int c0, int K, double *secs){
    pulsar_session *s=NULL; if(pulsar_session_create(&s,g_e,4096)!=0) return false;
    pulsar_gpu_graph *g=&s->graph; char e[256]; bool ok=true;
    if(g->banks.n_banks && !gpu_graph_bank_repoint(g,0)){ pulsar_session_free(s); return false; }
    pulsar_session_invalidate(s);
    pulsar_tokens p0={.v=g_toks.v,.len=c0,.cap=c0};
    if(pulsar_session_sync(s,&p0,e,sizeof e)!=0) ok=false;              /* untimed first chunk */
    pulsar_tokens p1={.v=g_toks.v,.len=c0+K,.cap=c0+K};
    double t0=now_s();
    if(ok && pulsar_session_sync(s,&p1,e,sizeof e)!=0) ok=false;        /* resumes at c0, prefills K */
    *secs=now_s()-t0;
    pulsar_session_free(s); return ok;
}

static int firstdiff(const int*a,const int*b,int n){ for(int i=0;i<n;i++) if(a[i]!=b[i]) return i; return -1; }

int main(int argc,char**argv){
    if(argc<2){ fprintf(stderr,"usage: %s MODEL\n",argv[0]); return 2; }
    pulsar_engine_options o; memset(&o,0,sizeof o); o.model_path=argv[1]; o.backend=PULSAR_BACKEND_CUDA;
    if(pulsar_engine_open(&g_e,&o)!=0){ fprintf(stderr,"engine open failed\n"); return 1; }
    printf("CONFIG: packed attn comp cache + MXFP4 indexer cache (the only formats)\n");
    size_t tl=0; char*txt=read_file("tests/long_context_story_prompt.txt",&tl);
    if(!txt){ fprintf(stderr,"prompt read failed\n"); return 1; }
    memset(&g_toks,0,sizeof g_toks); pulsar_tokenize_text(g_e,txt,&g_toks); free(txt);

    const int K1=512;                 /* gates 1&2: spans many ratio-4 (and ratio-128) groups */
    if(C0+K1+NGEN > g_toks.len){ fprintf(stderr,"prompt too short (%d)\n",g_toks.len); return 1; }

    /* GATE 1 (rigorous oracle) + GATE 2 (K>ratio boundary). K1 spans >1 ratio-4/128
     * group; step_end self-check runs inside decode_mixed (rc=-1 on a frontier miss).
     * THREE assertions distinguish accepted last-ulp drift from KV corruption:
     *   (1) NEXT-TOKEN exact vs classic-resume (prefill boundary + committed KV right),
     *   (2) last-position full-vocab logit rel-RMS < 1e-2 (corruption => large error),
     *   (3) the NGEN continuation is coherent (valid, non-degenerate), not required to
     *       match classic past the drift point. */
    const int vocab=(int)PULSAR_N_VOCAB;
    int ref[NGEN], mix[NGEN];
    float *ref_lg = (float *)malloc((size_t)vocab*sizeof(float)), *mix_lg=(float *)malloc((size_t)vocab*sizeof(float));
    if(!classic_stream(C0, C0+K1, ref, ref_lg)){ fprintf(stderr,"GATE FAIL: classic-resume reference failed\n"); g_fail=1; free(ref_lg);free(mix_lg); goto done; }
    if(!mixed_stream(C0, K1, mix, NULL, mix_lg)){ fprintf(stderr,"GATE FAIL: mixed prefill failed (step_end/coherence)\n"); g_fail=1; free(ref_lg);free(mix_lg); goto done; }
    {
        /* (2) relative RMS of the last-position logit vectors. */
        double se=0, sr=0;
        for(int i=0;i<vocab;i++){ double d=(double)mix_lg[i]-ref_lg[i]; se+=d*d; sr+=(double)ref_lg[i]*ref_lg[i]; }
        double rel_rms = sr>0 ? sqrt(se/sr) : (se>0?1e9:0.0);
        /* (3) coherence: valid ids + not a single repeated token (non-degenerate). */
        int distinct=1; for(int i=1;i<NGEN;i++) if(mix[i]!=mix[0]){ distinct=2; break; }
        int allvalid=1; for(int i=0;i<NGEN;i++) if(mix[i]<0||mix[i]>=vocab){ allvalid=0; break; }
        int cd=firstdiff(mix,ref,NGEN);   /* informational: continuation divergence point */
        printf("GATE 1: next-token mixed=%d classic=%d %s | last-pos logit rel-RMS=%.3e (<1e-2: %s) | continuation coherent=%s (valid=%d,distinct=%d, matches-classic-for=%d tok)\n",
               mix[0],ref[0], mix[0]==ref[0]?"MATCH":"MISMATCH", rel_rms, rel_rms<1e-2?"YES":"NO",
               (allvalid&&distinct>=2)?"YES":"NO", allvalid, distinct, cd<0?NGEN:cd);
        printf("GATE 2: K=%d spans >1 ratio group; step_end frontier self-check PASSED (mixed run returned valid tokens, no rc=-1)\n",K1);
        if(mix[0]!=ref[0]){ fprintf(stderr,"GATE 1 FAIL: next-token mismatch (prefill boundary/KV wrong)\n"); g_fail=1; }
        if(rel_rms>=1e-2){ fprintf(stderr,"GATE 1 FAIL: last-pos logit rel-RMS %.3e >= 1e-2 (KV corruption, not last-ulp drift)\n",rel_rms); g_fail=1; }
        if(!(allvalid&&distinct>=2)){ fprintf(stderr,"GATE 1 FAIL: continuation degenerate/garbage\n"); g_fail=1; }
    }
    free(ref_lg); free(mix_lg);

    /* GATE 3: SPEED — mixed K-run vs classic prefill of K, at K in {512,2048}. */
    int Ks[2];
    Ks[0]=512; Ks[1]=2048;
    for(int ki=0; ki<2; ki++){
        int K=Ks[ki];
        if(C0+K+NGEN > g_toks.len){ printf("GATE 3: K=%d skipped (prompt too short)\n",K); continue; }
        double t_mix=0, t_cls=0; int tmp[NGEN];
        if(!mixed_stream(C0,K,tmp,&t_mix,NULL)){ fprintf(stderr,"GATE 3 FAIL: mixed K=%d\n",K); g_fail=1; continue; }
        if(!classic_prefill_time(C0,K,&t_cls)){ fprintf(stderr,"GATE 3 FAIL: classic K=%d\n",K); g_fail=1; continue; }
        double sp_mix=K/t_mix, sp_cls=K/t_cls, ratio=t_mix/t_cls;
        /* Enforce parity only at a REALISTIC chunk (K<=512) vs classic-RESUME. Large
         * K (2048) is INFORMATIONAL: the mseq banked per-KEY indirection scales with
         * attended keys (~K^2 for prefill), so it grows with K. That banked cost is
         * the honest fused-step price inc-4's MULTI-bank step pays (it CANNOT use the
         * single-bank descr==0 shortcut) and inc-5 measures against the jitter win —
         * so it is documented here, deliberately NOT optimized away in a single-bank
         * special case inc-4 discards. */
        const char *tag = (K<=512) ? "PARITY(enforced)" : "INFORMATIONAL(banked per-key cost; inc-4 pays it)";
        printf("GATE 3 SPEED K=%d [%s]: mixed %.0f tok/s (%.3fs) vs classic-RESUME %.0f tok/s (%.3fs) -> %.2fx classic time\n",
               K,tag,sp_mix,t_mix,sp_cls,t_cls,ratio);
        if(K<=512 && ratio>1.5){ fprintf(stderr,"GATE 3 FAIL: K=%d mixed %.2fx slower than classic-RESUME (>1.5x)\n",K,ratio); g_fail=1; }
    }


    /* GATE 4 (L112 inc A/B): TWO K-row prefill runs from TWO banks in ONE mixed
     * sweep. Oracle: each run's next token matches its own SINGLE-run mixed
     * reference exactly (gates 1-3 tie single-run to classic), and last-position
     * logit rel-RMS < 1e-2 -- co-batching a second prompt must not perturb the
     * first (bank isolation), in either direction. Distinct content per bank
     * (offset 700) so cross-bank bleed cannot alias as equality. */
    {
        /* K=257: NOT a multiple of ratio, so the single-run reference and the
         * co-batched sweep BOTH take the per-token compressor path -- the
         * aligned-run replay kernel is numerically close but not identical to
         * the per-token kernel, and an oracle comparing across that pair
         * aliases path numerics as co-batch error (first observed 2026-08-26:
         * K=256 refs on the replay path read rel-RMS 5e-2/2.5e-1). */
        const int OFFB = 700, K4 = 257;
        pulsar_session *s4 = NULL;
        if (pulsar_session_create(&s4, g_e, 4096) != 0) { fprintf(stderr,"GATE 4 FAIL: session\n"); g_fail=1; goto done; }
        if (s4->graph.banks.n_banks < 2) {
            printf("GATE 4: skipped (pool %u < 2 banks; run with PULSAR_MSEQ_BANKS=2)\n",
                   s4->graph.banks.n_banks);
            pulsar_session_free(s4);
        } else if (OFFB + C0 + K4 > g_toks.len) {
            printf("GATE 4: skipped (prompt too short)\n");
            pulsar_session_free(s4);
        } else {
            float *ref0 = (float *)malloc((size_t)vocab*sizeof(float));
            float *ref1 = (float *)malloc((size_t)vocab*sizeof(float));
            float *lg2  = (float *)malloc((size_t)2*vocab*sizeof(float));
            int rnext[2] = {-1,-1};
            bool ok = true;
            /* single-run references, one fresh session each. */
            for (int r = 0; ok && r < 2; r++) {
                const int off = r ? OFFB : 0;
                pulsar_session *sr = NULL;
                if (pulsar_session_create(&sr, g_e, 4096) != 0) { ok=false; break; }
                pulsar_gpu_graph *gr = &sr->graph; char e[256];
                if (gr->banks.n_banks && !gpu_graph_bank_repoint(gr,0)) ok=false;
                if (ok) {
                    pulsar_session_invalidate(sr);
                    pulsar_tokens pc = {.v=g_toks.v+off,.len=C0,.cap=C0};
                    if (pulsar_session_sync(sr,&pc,e,sizeof e)!=0) ok=false;
                }
                if (ok) {
                    gpu_graph_bank_counters_capture(gr,0);
                    pulsar_multiseq_req *rq=(pulsar_multiseq_req*)malloc((size_t)K4*sizeof(*rq));
                    for (int j=0;j<K4;j++){ rq[j].bank=0; rq[j].pos=C0+j; rq[j].token=g_toks.v[off+C0+j]; }
                    uint32_t nr=0;
                    float *dst = r ? ref1 : ref0;
                    if (pulsar_session_decode_mixed(sr,rq,(uint32_t)K4,dst,vocab,&nr,0u,e,sizeof e)!=0 || nr!=1) ok=false;
                    else rnext[r]=(int)argmax_f32(dst,(uint64_t)vocab);
                    free(rq);
                }
                pulsar_session_free(sr);
            }
            /* the co-batched sweep: bank0 = offset 0, bank1 = offset OFFB. */
            if (ok) {
                pulsar_gpu_graph *g4=&s4->graph; char e[256];
                for (int b=0; ok && b<2; b++) {
                    const int off = b ? OFFB : 0;
                    if (!gpu_graph_bank_repoint(g4,(uint32_t)b)) { ok=false; break; }
                    pulsar_session_invalidate(s4);
                    pulsar_tokens pc={.v=g_toks.v+off,.len=C0,.cap=C0};
                    if (pulsar_session_sync(s4,&pc,e,sizeof e)!=0){ ok=false; break; }
                    gpu_graph_bank_counters_capture(g4,(uint32_t)b);
                }
                if (ok) {
                    pulsar_multiseq_req *rq=(pulsar_multiseq_req*)malloc((size_t)2*K4*sizeof(*rq));
                    for (int j=0;j<K4;j++){ rq[j].bank=0;    rq[j].pos=C0+j;    rq[j].token=g_toks.v[C0+j]; }
                    for (int j=0;j<K4;j++){ rq[K4+j].bank=1; rq[K4+j].pos=C0+j; rq[K4+j].token=g_toks.v[OFFB+C0+j]; }
                    uint32_t nr=0;
                    if (pulsar_session_decode_mixed(s4,rq,(uint32_t)(2*K4),lg2,2*vocab,&nr,0u,e,sizeof e)!=0){
                        fprintf(stderr,"GATE 4 FAIL: co-batched sweep: %s\n",e); ok=false;
                    } else if (nr!=2) {
                        fprintf(stderr,"GATE 4 FAIL: n_runs=%u expected 2\n",nr); ok=false;
                    }
                    free(rq);
                }
            }
            if (ok) {
                for (int r=0; r<2; r++) {
                    const float *ref = r ? ref1 : ref0;
                    const float *row = lg2 + (size_t)r*vocab;
                    const int nx = (int)argmax_f32(row,(uint64_t)vocab);
                    double se=0, sr2=0;
                    for (int i2=0;i2<vocab;i2++){ double d=(double)row[i2]-ref[i2]; se+=d*d; sr2+=(double)ref[i2]*ref[i2]; }
                    double rel=sr2>0?sqrt(se/sr2):(se>0?1e9:0.0);
                    printf("GATE 4 run %d: next co-batched=%d single-run=%d %s | rel-RMS=%.3e (<1e-2: %s)\n",
                           r,nx,rnext[r],nx==rnext[r]?"MATCH":"MISMATCH",rel,rel<1e-2?"YES":"NO");
                    if (nx!=rnext[r]){ fprintf(stderr,"GATE 4 FAIL: run %d next-token mismatch (co-batch perturbs the run)\n",r); g_fail=1; }
                    if (rel>=1e-2){ fprintf(stderr,"GATE 4 FAIL: run %d rel-RMS %.3e >= 1e-2\n",r,rel); g_fail=1; }
                }
            } else { fprintf(stderr,"GATE 4 FAIL: setup/sweep failed\n"); g_fail=1; }
            free(ref0); free(ref1); free(lg2);
            pulsar_session_free(s4);

            /* TWIN check: co-batch two banks fed IDENTICAL text (offset 0).
             * Same positions, same tokens, independent per-bank state -- the
             * two runs' last-row logits must be BITWISE identical; any
             * difference is cross-run leakage, free of every path-numerics
             * alias. */
            pulsar_session *st = NULL;
            if (pulsar_session_create(&st, g_e, 4096) != 0 ||
                st->graph.banks.n_banks < 2) {
                printf("GATE 4 TWIN: skipped (session/banks)\n");
                if (st) pulsar_session_free(st);
            } else {
                float *tl = (float *)malloc((size_t)2*vocab*sizeof(float));
                bool ok2 = true;
                pulsar_gpu_graph *gt = &st->graph; char e2[256];
                for (int b = 0; ok2 && b < 2; b++) {
                    if (!gpu_graph_bank_repoint(gt,(uint32_t)b)) { ok2=false; break; }
                    pulsar_session_invalidate(st);
                    pulsar_tokens pc = {.v=g_toks.v,.len=C0,.cap=C0};
                    if (pulsar_session_sync(st,&pc,e2,sizeof e2)!=0){ ok2=false; break; }
                    gpu_graph_bank_counters_capture(gt,(uint32_t)b);
                }
                if (ok2) {
                    pulsar_multiseq_req *rq=(pulsar_multiseq_req*)malloc((size_t)2*K4*sizeof(*rq));
                    for (int j=0;j<K4;j++){ rq[j].bank=0;    rq[j].pos=C0+j; rq[j].token=g_toks.v[C0+j]; }
                    for (int j=0;j<K4;j++){ rq[K4+j].bank=1; rq[K4+j].pos=C0+j; rq[K4+j].token=g_toks.v[C0+j]; }
                    uint32_t nr=0;
                    if (pulsar_session_decode_mixed(st,rq,(uint32_t)(2*K4),tl,2*vocab,&nr,0u,e2,sizeof e2)!=0 || nr!=2)
                        ok2=false;
                    free(rq);
                }
                if (ok2) {
                    long fd=-1;
                    for (int i2=0;i2<vocab;i2++)
                        if (tl[i2]!=tl[(size_t)vocab+i2]) { fd=i2; break; }
                    if (fd<0) printf("GATE 4 TWIN: identical-content runs BITWISE IDENTICAL (no cross-run leakage)\n");
                    else {
                        fprintf(stderr,"GATE 4 TWIN FAIL: identical-content runs differ at logit %ld (%.6e vs %.6e) -- CROSS-RUN LEAKAGE\n",
                                fd,(double)tl[fd],(double)tl[(size_t)vocab+fd]);
                        g_fail=1;
                    }
                } else { fprintf(stderr,"GATE 4 TWIN FAIL: setup/sweep failed\n"); g_fail=1; }
                free(tl);
                pulsar_session_free(st);
            }
        }
    }

done:
    pulsar_engine_close(g_e);
    if(g_fail){ fprintf(stderr,"MIXED-PREFILL GATE: FAIL\n"); return 1; }
    printf("MIXED-PREFILL GATE: PASS\n"); return 0;
}
