/* IQ2_XXS gate/up SoA-REPACK microbench -- Phase 0 for the MMQ port.
 * Forked from iq2_lds_bench.cu (harness: routing, tiles, occupancy, timing).
 *
 * FINDING UNDER TEST: block_iq2_xxs is 66 BYTES with qs[] at offset 2, so the
 * weight stream is only 2-byte aligned and nvcc must emit LDG.E.U16 -- two
 * 16-bit loads per 32-bit weight word.  Confirmed in SASS on our shipped
 * object (zero 64/128-bit global loads in the gate/up kernel) and
 * independently in Entrpi/ds4 cuda/mmq/test/proto_iq2_aligned.cu, which
 * measures the resulting matvec at ~142 GB/s against a ~200 GB/s ceiling and
 * prescribes the same fix.
 *
 * A/B -- identical arithmetic, identical accumulation order, ONLY the weight
 * memory LAYOUT varies:
 *   SOA=0  packed : the shipped [d | qs[32]] 66-byte block (2-byte aligned)
 *   SOA=1  soa    : d[] and qs[] split into separate planes; qs is 16-byte
 *                   aligned per block and read as uint4 (one 16-byte load
 *                   covers two ib32 groups instead of eight LDG.E.U16)
 * Byte count is IDENTICAL either way (66 B/block); only alignment changes.
 * Output must be bit-identical: same values, same order, different loads.
 *
 * A positive result is also a PREREQUISITE landed: MMQ's
 * load_tiles_iq2_xxs_soa consumes exactly this artifact.  A negative result
 * kills the expensive half (the ~10k-line MMQ vendoring) early and cheap.
 *
 *   nvcc -O3 --use_fast_math -arch=sm_120f -Isrc -o /tmp/iq2_soa_bench tests/iq2_soa_bench.cu
 */
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CUDA_QK_K 256
#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e_)); exit(1);} } while(0)

typedef struct { uint16_t d; uint16_t qs[CUDA_QK_K/8]; } cuda_block_iq2_xxs;
typedef struct { float d; int8_t qs[CUDA_QK_K]; int16_t bsums[CUDA_QK_K/16]; } cuda_block_q8_K;

#include "cuda/pulsar_iq2_tables_cuda.inc"   /* cuda_iq2xxs_grid[256], cuda_ksigns_iq2xs[128] */

__device__ static float dev_f16_to_f32(uint16_t v){ return __half2float(*reinterpret_cast<const __half*>(&v)); }
__device__ __forceinline__ static uint32_t dev_unpack_iq2_signs(uint32_t v){ const uint32_t p=__popc(v)&1u; const uint32_t s=v^(p<<7u); return s*0x01010101u; }
__device__ __forceinline__ static void dev_iq2_i8x8_lut(const uint64_t*grid,const uint8_t*signs,uint8_t gi,uint32_t si,int32_t*w0,int32_t*w1){
    const uint32_t s=dev_unpack_iq2_signs(signs[si]);
    const int32_t sm0=__vcmpne4(s&0x08040201u,0); const int32_t sm1=__vcmpne4(s&0x80402010u,0);
    const uint64_t g=grid[gi]; *w0=__vsub4((int32_t)(uint32_t)g^sm0,sm0); *w1=__vsub4((int32_t)(uint32_t)(g>>32)^sm1,sm1);
}
__device__ static float quarter_warp_sum_f32(float v, uint32_t){ uint32_t mask=0xffu<<(threadIdx.x&24u); for(int o=4;o>0;o>>=1) v+=__shfl_down_sync(mask,v,o,8); return v; }


/* ---- SoA layout ------------------------------------------------------------
 * packed (shipped):  block b at  base + b*66  ->  [u16 d][u16 qs[32]]
 * soa:               d plane at  dbase + b*2   (u16, contiguous)
 *                    q plane at  qbase + b*64  (16-byte aligned, uint4-loadable)
 * Same 66 B/block total.  The q plane's 64-byte stride is what makes every
 * load full width and aligned. */
#define SOA_QBYTES 64u

/* dot: SOA selects the weight LAYOUT only.  The ib32 loop, the LUT decode, the
 * dp4a order and the final fold are byte-for-byte the shipped ones. */
template <uint32_t NT, int SOA>
__device__ static void dev_dot_iq2_soa(
        const uint8_t *qbase, const uint16_t *dbase, uint64_t blk,
        const cuda_block_q8_K *const *ys, uint32_t n, float acc[NT],
        const uint64_t *grid, const uint8_t *signs){
    const float xd = SOA ? dev_f16_to_f32(dbase[blk])
                         : dev_f16_to_f32(*(const uint16_t *)(qbase + blk*66u));
    /* packed: qs starts 2 B into the block (hence the 2-byte alignment).
     * soa: qs is its own 64 B-aligned row, so uint4 loads are legal. */
    const uint16_t *q2p = SOA ? (const uint16_t *)(qbase + blk*SOA_QBYTES)
                              : (const uint16_t *)(qbase + blk*66u + 2u);
    int32_t bsum[NT];
    #pragma unroll
    for(uint32_t p=0;p<NT;p++) bsum[p]=0;
    for(int ib32=0; ib32<CUDA_QK_K/32; ib32++){
        uint32_t aux0, aux1;
        if (SOA) {
            /* one aligned 8-byte load replaces four LDG.E.U16 */
            const uint2 w = *(const uint2 *)(q2p + ib32*4);
            aux0 = w.x; aux1 = w.y;
        } else {
            const uint16_t *q2 = q2p + ib32*4;
            aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1]<<16);
            aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3]<<16);
        }
        const int32_t ls=(int32_t)(2u*(aux1>>28)+1u); int32_t w[8];
        dev_iq2_i8x8_lut(grid,signs,(uint8_t)(aux0&0xffu),(aux1>>0)&127u,&w[0],&w[1]);
        dev_iq2_i8x8_lut(grid,signs,(uint8_t)((aux0>>8)&0xffu),(aux1>>7)&127u,&w[2],&w[3]);
        dev_iq2_i8x8_lut(grid,signs,(uint8_t)((aux0>>16)&0xffu),(aux1>>14)&127u,&w[4],&w[5]);
        dev_iq2_i8x8_lut(grid,signs,(uint8_t)((aux0>>24)&0xffu),(aux1>>21)&127u,&w[6],&w[7]);
        #pragma unroll
        for(uint32_t p=0;p<NT;p++){ if(p>=n)break; const int8_t*q=ys[p]->qs+ib32*32; int32_t sumi=0;
            sumi=__dp4a(w[0],*(const int32_t*)(q+0),sumi);  sumi=__dp4a(w[1],*(const int32_t*)(q+4),sumi);
            sumi=__dp4a(w[2],*(const int32_t*)(q+8),sumi);  sumi=__dp4a(w[3],*(const int32_t*)(q+12),sumi);
            sumi=__dp4a(w[4],*(const int32_t*)(q+16),sumi); sumi=__dp4a(w[5],*(const int32_t*)(q+20),sumi);
            sumi=__dp4a(w[6],*(const int32_t*)(q+24),sumi); sumi=__dp4a(w[7],*(const int32_t*)(q+28),sumi);
            bsum[p]+=sumi*ls; }
    }
    #pragma unroll
    for(uint32_t p=0;p<NT;p++){ if(p>=n)break; acc[p]+=0.125f*xd*ys[p]->d*(float)bsum[p]; }
}

__host__ __device__ __forceinline__ static uint32_t iq2_smem_bytes(uint32_t nt,uint32_t xqb){ return (xqb<=16u)?(uint32_t)(nt*xqb*(uint32_t)sizeof(cuda_block_q8_K)):0u; }

template <uint32_t ROW_SPAN, uint32_t NT, int SOA>
__global__ static void iq2_soa_kernel(float*mid_out,const uint8_t*gq,const uint16_t*gd,
        const uint8_t*uq,const uint16_t*ud,const cuda_block_q8_K*xq,
        const uint32_t*sorted_pairs,const uint32_t*offsets,const uint32_t*counts,const uint32_t*tile_total,
        const uint32_t*tile_experts,const uint32_t*tile_starts,const float*weights,
        uint64_t blocks_per_expert,uint64_t blocks_per_row,
        uint32_t xq_blocks,uint32_t expert_mid_dim,uint32_t n_expert,float clamp){
    uint32_t tile=blockIdx.y; if(tile>=*tile_total)return;
    uint32_t lane=threadIdx.x&7u; uint32_t row_lane=threadIdx.x>>3u;
    uint32_t expert=tile_experts[tile]; uint32_t local_start=tile_starts[tile];
    extern __shared__ cuda_block_q8_K sxq[];
    __shared__ uint64_t s_iq2_grid[256]; __shared__ uint8_t s_iq2_signs[128];
    uint32_t pair[NT],tok[NT],slot[NT]; const cuda_block_q8_K*xqb[NT];
    #pragma unroll
    for(uint32_t i=0;i<NT;i++){pair[i]=0;tok[i]=0;slot[i]=0;xqb[i]=NULL;}
    const uint32_t cnt=counts[expert]; const uint32_t avail=(local_start<cnt)?(cnt-local_start):0u; const uint32_t np=avail<NT?avail:NT;
    #pragma unroll
    for(uint32_t i=0;i<NT;i++){ if(i>=np)break; pair[i]=sorted_pairs[offsets[expert]+local_start+i]; tok[i]=pair[i]/n_expert; slot[i]=pair[i]-tok[i]*n_expert; xqb[i]=xq+(uint64_t)tok[i]*xq_blocks; }
    for(uint32_t i=threadIdx.x;i<np*xq_blocks;i+=blockDim.x){ uint32_t p=i/xq_blocks; uint32_t b=i-p*xq_blocks; sxq[p*xq_blocks+b]=xqb[p][b]; }
    for(uint32_t i=threadIdx.x;i<256u;i+=blockDim.x) s_iq2_grid[i]=cuda_iq2xxs_grid[i];
    for(uint32_t i=threadIdx.x;i<128u;i+=blockDim.x) s_iq2_signs[i]=cuda_ksigns_iq2xs[i];
    __syncthreads();
    #pragma unroll
    for(uint32_t p=0;p<NT;p++){ xqb[p]=sxq+p*xq_blocks; }
    for(uint32_t rr=0;rr<ROW_SPAN/32u;rr++){
        uint32_t row=blockIdx.x*ROW_SPAN+row_lane+rr*32u; if(row>=expert_mid_dim)continue;
        const uint64_t blk0 = (uint64_t)expert*blocks_per_expert + (uint64_t)row*blocks_per_row;
        float gate[NT],up[NT];
        #pragma unroll
        for(uint32_t p=0;p<NT;p++){gate[p]=0.0f;up[p]=0.0f;}
        for(uint32_t b=lane;b<xq_blocks;b+=8u){
            const cuda_block_q8_K*yb[NT];
            #pragma unroll
            for(uint32_t p=0;p<NT;p++){ yb[p]=(p<np)?xqb[p]+b:NULL; }
            dev_dot_iq2_soa<NT,SOA>(gq,gd,blk0+b,yb,np,gate,s_iq2_grid,s_iq2_signs);
            dev_dot_iq2_soa<NT,SOA>(uq,ud,blk0+b,yb,np,up,s_iq2_grid,s_iq2_signs);
        }
        #pragma unroll
        for(uint32_t p=0;p<NT;p++){ if(p>=np)break;
            gate[p]=quarter_warp_sum_f32(gate[p],lane); up[p]=quarter_warp_sum_f32(up[p],lane);
            if(lane==0){ if(clamp>1.0e-6f){ if(gate[p]>clamp)gate[p]=clamp; if(up[p]>clamp)up[p]=clamp; if(up[p]<-clamp)up[p]=-clamp; }
                const uint64_t off=(uint64_t)pair[p]*expert_mid_dim+row;
                mid_out[off]=(gate[p]/(1.0f+expf(-gate[p])))*up[p]*weights[(uint64_t)tok[p]*n_expert+slot[p]]; }
        }
    }
}

static const uint32_t N_EXPERT_TOTAL=192, N_EXPERT_USED=6, IN_DIM=4096, MID_DIM=2048;

struct Routing { uint32_t n_tokens,pair_count; uint32_t*d_sorted_pairs,*d_offsets,*d_counts; };

static uint32_t rng=98765u; static uint32_t xrand(){ rng^=rng<<13; rng^=rng>>17; rng^=rng<<5; return rng; }

/* mode 0 = balanced (round-robin), mode 1 = imbalanced (random top-6 per token). */
static void build_routing(Routing*r,uint32_t n_tokens,int mode){
    r->n_tokens=n_tokens; r->pair_count=n_tokens*N_EXPERT_USED;
    uint32_t*h_pairs=(uint32_t*)malloc(r->pair_count*sizeof(uint32_t));
    uint32_t*h_counts=(uint32_t*)calloc(N_EXPERT_TOTAL,sizeof(uint32_t));
    uint32_t*h_off=(uint32_t*)calloc(N_EXPERT_TOTAL+1,sizeof(uint32_t));
    uint32_t*exp_of=(uint32_t*)malloc(r->pair_count*sizeof(uint32_t));
    for(uint32_t t=0;t<n_tokens;t++){
        if(mode==0){ for(uint32_t s=0;s<N_EXPERT_USED;s++){ uint32_t p=t*N_EXPERT_USED+s; uint32_t e=(t*N_EXPERT_USED+s)%N_EXPERT_TOTAL; exp_of[p]=e; h_counts[e]++; } }
        else { /* pick 6 distinct experts with a skewed distribution (square of uniform -> favors low ids) */
            uint32_t chosen[N_EXPERT_USED]; uint32_t nc=0;
            while(nc<N_EXPERT_USED){ double u=(double)(xrand()%100000)/100000.0; uint32_t e=(uint32_t)(u*u*N_EXPERT_TOTAL); if(e>=N_EXPERT_TOTAL)e=N_EXPERT_TOTAL-1; int dup=0; for(uint32_t k=0;k<nc;k++) if(chosen[k]==e)dup=1; if(!dup)chosen[nc++]=e; }
            for(uint32_t s=0;s<N_EXPERT_USED;s++){ uint32_t p=t*N_EXPERT_USED+s; exp_of[p]=chosen[s]; h_counts[chosen[s]]++; }
        }
    }
    for(uint32_t e=0;e<N_EXPERT_TOTAL;e++) h_off[e+1]=h_off[e]+h_counts[e];
    uint32_t*cur=(uint32_t*)calloc(N_EXPERT_TOTAL,sizeof(uint32_t));
    for(uint32_t t=0;t<n_tokens;t++) for(uint32_t s=0;s<N_EXPERT_USED;s++){ uint32_t p=t*N_EXPERT_USED+s; uint32_t e=exp_of[p]; h_pairs[h_off[e]+cur[e]++]=t*N_EXPERT_USED+s; }
    CHECK(cudaMalloc(&r->d_sorted_pairs,r->pair_count*sizeof(uint32_t)));
    CHECK(cudaMalloc(&r->d_offsets,(N_EXPERT_TOTAL+1)*sizeof(uint32_t)));
    CHECK(cudaMalloc(&r->d_counts,N_EXPERT_TOTAL*sizeof(uint32_t)));
    CHECK(cudaMemcpy(r->d_sorted_pairs,h_pairs,r->pair_count*sizeof(uint32_t),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(r->d_offsets,h_off,(N_EXPERT_TOTAL+1)*sizeof(uint32_t),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(r->d_counts,h_counts,N_EXPERT_TOTAL*sizeof(uint32_t),cudaMemcpyHostToDevice));
    /* report imbalance */
    uint32_t mx=0,nz=0; for(uint32_t e=0;e<N_EXPERT_TOTAL;e++){ if(h_counts[e]>mx)mx=h_counts[e]; if(h_counts[e])nz++; }
    printf("# routing mode=%d: %u active experts, max_count=%u, avg=%.1f\n",mode,nz,mx,(double)r->pair_count/nz);
    free(h_pairs);free(h_counts);free(h_off);free(cur);free(exp_of);
}

struct Tiles{ uint32_t*d_total,*d_experts,*d_starts; uint32_t n; };
static void build_tiles(Tiles*t,const Routing*r,uint32_t tw){
    uint32_t*hc=(uint32_t*)malloc(N_EXPERT_TOTAL*sizeof(uint32_t));
    CHECK(cudaMemcpy(hc,r->d_counts,N_EXPERT_TOTAL*sizeof(uint32_t),cudaMemcpyDeviceToHost));
    uint32_t cap=0; for(uint32_t e=0;e<N_EXPERT_TOTAL;e++) cap+=(hc[e]+tw-1)/tw; if(cap==0)cap=1;
    uint32_t*he=(uint32_t*)malloc(cap*sizeof(uint32_t)); uint32_t*hs=(uint32_t*)malloc(cap*sizeof(uint32_t)); uint32_t n=0;
    for(uint32_t e=0;e<N_EXPERT_TOTAL;e++) for(uint32_t s=0;s<hc[e];s+=tw){ he[n]=e;hs[n]=s;n++; }
    t->n=n;
    CHECK(cudaMalloc(&t->d_total,sizeof(uint32_t))); CHECK(cudaMalloc(&t->d_experts,(n?n:1)*sizeof(uint32_t))); CHECK(cudaMalloc(&t->d_starts,(n?n:1)*sizeof(uint32_t)));
    CHECK(cudaMemcpy(t->d_total,&n,sizeof(uint32_t),cudaMemcpyHostToDevice));
    if(n){ CHECK(cudaMemcpy(t->d_experts,he,n*sizeof(uint32_t),cudaMemcpyHostToDevice)); CHECK(cudaMemcpy(t->d_starts,hs,n*sizeof(uint32_t),cudaMemcpyHostToDevice)); }
    free(hc);free(he);free(hs);
}
static void free_tiles(Tiles*t){ cudaFree(t->d_total);cudaFree(t->d_experts);cudaFree(t->d_starts); }

template<typename K>
static int report_kernel(const char*name,K kern,uint32_t threads,size_t dyn){
    cudaFuncAttributes a; CHECK(cudaFuncGetAttributes(&a,(const void*)kern));
    int blocks=0; cudaError_t e=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,(const void*)kern,threads,dyn);
    if(e!=cudaSuccess){ (void)cudaGetLastError(); }
    int dev;CHECK(cudaGetDevice(&dev)); int mts;CHECK(cudaDeviceGetAttribute(&mts,cudaDevAttrMaxThreadsPerMultiProcessor,dev));
    printf("    %-16s regs=%-4d static_smem=%-6zu dyn_smem=%-7zu spill=%zu  blocks/SM=%d  occ=%.1f%%\n",
           name,a.numRegs,(size_t)a.sharedSizeBytes,dyn,(size_t)a.localSizeBytes,blocks,100.0*blocks*threads/mts);
    return blocks;
}

int main(int argc,char**argv){
    uint32_t n_tokens=argc>1?(uint32_t)atoi(argv[1]):2048;
    int iters=argc>2?atoi(argv[2]):20;
    const int TRIALS=argc>3?atoi(argv[3]):8;
    const uint32_t xq_blocks=IN_DIM/CUDA_QK_K;                 /* 16 */
    const uint64_t bpr = IN_DIM/CUDA_QK_K;                     /* blocks per row */
    const uint64_t bpe = bpr*MID_DIM;                          /* blocks per expert */
    const uint64_t nblk = bpe*N_EXPERT_TOTAL;                  /* total blocks */
    const size_t smem=iq2_smem_bytes(8,xq_blocks);
    printf("# IQ2 SoA-vs-packed bench: tokens=%u iters=%d trials=%d blocks=%lu\n",
           n_tokens,iters,TRIALS,(unsigned long)nblk);
    printf("# packed=%.2f GB  soa(q)=%.2f GB + d=%.2f GB  (identical total)\n",
           nblk*66.0/1e9, nblk*64.0/1e9, nblk*2.0/1e9);

    /* packed planes (66 B/blk) and soa planes (64 B q + 2 B d), same content */
    uint8_t *d_gp,*d_up_p; uint8_t *d_gq,*d_uq; uint16_t *d_gd,*d_ud;
    CHECK(cudaMalloc(&d_gp,nblk*66)); CHECK(cudaMalloc(&d_up_p,nblk*66));
    CHECK(cudaMalloc(&d_gq,nblk*SOA_QBYTES)); CHECK(cudaMalloc(&d_uq,nblk*SOA_QBYTES));
    CHECK(cudaMalloc(&d_gd,nblk*2)); CHECK(cudaMalloc(&d_ud,nblk*2));
    { const uint64_t CH=1u<<20;                      /* fill in block chunks */
      uint8_t*hp=(uint8_t*)malloc(CH*66); uint8_t*hq=(uint8_t*)malloc(CH*SOA_QBYTES);
      uint16_t*hd=(uint16_t*)malloc(CH*2);
      for(uint64_t off=0; off<nblk; off+=CH){
        uint64_t nn=(nblk-off<CH)?nblk-off:CH;
        for(int side=0; side<2; side++){
          for(uint64_t b=0;b<nn;b++){
            uint16_t dv=(uint16_t)(0x3800u | (xrand()&0xff));   /* plausible fp16 */
            hd[b]=dv; *(uint16_t*)(hp+b*66)=dv;
            for(int j=0;j<32;j++){ uint16_t q=(uint16_t)xrand();
              *(uint16_t*)(hp+b*66+2+j*2)=q; *(uint16_t*)(hq+b*SOA_QBYTES+j*2)=q; }
          }
          uint8_t *dstp = side? d_up_p : d_gp; uint8_t *dstq = side? d_uq : d_gq;
          uint16_t *dstd = side? d_ud : d_gd;
          CHECK(cudaMemcpy(dstp+off*66,hp,nn*66,cudaMemcpyHostToDevice));
          CHECK(cudaMemcpy(dstq+off*SOA_QBYTES,hq,nn*SOA_QBYTES,cudaMemcpyHostToDevice));
          CHECK(cudaMemcpy(dstd+off,hd,nn*2,cudaMemcpyHostToDevice));
        }
      }
      free(hp);free(hq);free(hd); }

    size_t xq_count=(size_t)n_tokens*xq_blocks; cuda_block_q8_K*d_xq;
    CHECK(cudaMalloc(&d_xq,xq_count*sizeof(cuda_block_q8_K)));
    { cuda_block_q8_K*hx=(cuda_block_q8_K*)malloc(xq_count*sizeof(cuda_block_q8_K));
      for(size_t i=0;i<xq_count;i++){ hx[i].d=0.02f+0.001f*(i%7);
        for(int j=0;j<CUDA_QK_K;j++)hx[i].qs[j]=(int8_t)((int)(xrand()&0x3f)-32);
        for(int j=0;j<CUDA_QK_K/16;j++)hx[i].bsums[j]=0; }
      CHECK(cudaMemcpy(d_xq,hx,xq_count*sizeof(cuda_block_q8_K),cudaMemcpyHostToDevice)); free(hx); }

    float*d_weights,*d_mid; size_t mid_elems=(size_t)n_tokens*N_EXPERT_USED*MID_DIM;
    CHECK(cudaMalloc(&d_weights,(size_t)n_tokens*N_EXPERT_USED*sizeof(float)));
    CHECK(cudaMalloc(&d_mid,mid_elems*sizeof(float)));
    { float*hw=(float*)malloc((size_t)n_tokens*N_EXPERT_USED*sizeof(float));
      for(size_t i=0;i<(size_t)n_tokens*N_EXPERT_USED;i++)hw[i]=0.5f+0.5f*((xrand()&255)/255.0f);
      CHECK(cudaMemcpy(d_weights,hw,(size_t)n_tokens*N_EXPERT_USED*sizeof(float),cudaMemcpyHostToDevice)); free(hw); }

    const float clamp=0.0f;
    cudaEvent_t e0,e1; CHECK(cudaEventCreate(&e0)); CHECK(cudaEventCreate(&e1));
    float*hr=(float*)malloc(mid_elems*sizeof(float)); float*ho=(float*)malloc(mid_elems*sizeof(float));

    for(int mode=0;mode<2;mode++){
        printf("\n===== ROUTING %s =====\n", mode==0?"BALANCED":"IMBALANCED");
        Routing rt; build_routing(&rt,n_tokens,mode);
        Tiles t8; build_tiles(&t8,&rt,8);
        printf("# tiles=%u\n",t8.n);
        report_kernel("packed", iq2_soa_kernel<1024u,8u,0>,256,smem);
        report_kernel("soa",    iq2_soa_kernel<1024u,8u,1>,256,smem);

        #define RUNQ(SOAV,MSOUT) do{ \
            dim3 grid_q((MID_DIM+1023u)/1024u,t8.n,1); \
            float best_q=1e30f; \
            for(int it=0;it<iters;it++){ CHECK(cudaEventRecord(e0)); \
                iq2_soa_kernel<1024u,8u,SOAV><<<grid_q,256,smem>>>(d_mid,d_gq,d_gd,d_uq,d_ud,d_xq, \
                    rt.d_sorted_pairs,rt.d_offsets,rt.d_counts,t8.d_total,t8.d_experts,t8.d_starts, \
                    d_weights,bpe,bpr,xq_blocks,MID_DIM,N_EXPERT_USED,clamp); \
                CHECK(cudaEventRecord(e1)); CHECK(cudaEventSynchronize(e1)); \
                float ms_q; CHECK(cudaEventElapsedTime(&ms_q,e0,e1)); if(ms_q<best_q)best_q=ms_q; } \
            MSOUT=best_q; }while(0)
        /* packed reads its own 66 B planes; pass them in the q slots */
        #define RUNP(MSOUT) do{ \
            dim3 grid_q((MID_DIM+1023u)/1024u,t8.n,1); \
            float best_q=1e30f; \
            for(int it=0;it<iters;it++){ CHECK(cudaEventRecord(e0)); \
                iq2_soa_kernel<1024u,8u,0><<<grid_q,256,smem>>>(d_mid,d_gp,d_gd,d_up_p,d_ud,d_xq, \
                    rt.d_sorted_pairs,rt.d_offsets,rt.d_counts,t8.d_total,t8.d_experts,t8.d_starts, \
                    d_weights,bpe,bpr,xq_blocks,MID_DIM,N_EXPERT_USED,clamp); \
                CHECK(cudaEventRecord(e1)); CHECK(cudaEventSynchronize(e1)); \
                float ms_q; CHECK(cudaEventElapsedTime(&ms_q,e0,e1)); if(ms_q<best_q)best_q=ms_q; } \
            MSOUT=best_q; }while(0)

        /* bit-exactness: packed vs soa must agree exactly */
        float t_p=0,t_s=0;
        RUNP(t_p); CHECK(cudaMemcpy(hr,d_mid,mid_elems*sizeof(float),cudaMemcpyDeviceToHost));
        RUNQ(1,t_s); CHECK(cudaMemcpy(ho,d_mid,mid_elems*sizeof(float),cudaMemcpyDeviceToHost));
        { int bad=0; for(size_t i=0;i<mid_elems&&bad<5;i++) if(memcmp(&hr[i],&ho[i],4)!=0) bad++;
          printf("    bitexact soa vs packed: %s\n", bad?"** DIFFERS **":"identical"); }

        printf("  -- interleaved repeated A/B, %d trials --\n", TRIALS);
        float worst=1e30f,sum=0; int neg=0;
        for(int tr=0;tr<TRIALS;tr++){
            float a=0,b=0; RUNP(a); RUNQ(1,b);
            float r=a/b; sum+=r; if(r<worst)worst=r; if(r<=1.0f)neg++;
            printf("     trial %2d: packed=%.3f ms  soa=%.3f ms  ratio=%.4fx%s\n",
                   tr+1,a,b,r, r<=1.0f?"   <-- NOT POSITIVE":"");
        }
        printf("  >> SoA ratio over %d trials: worst=%.4fx  mean=%.4fx  (%d/%d not positive)\n",
               TRIALS, worst, sum/TRIALS, neg, TRIALS);
        free_tiles(&t8);
        cudaFree(rt.d_sorted_pairs);cudaFree(rt.d_offsets);cudaFree(rt.d_counts);
    }
    free(hr);free(ho);
    return 0;
}
