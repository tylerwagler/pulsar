"""Verify the repacked GGUF decodes to the SAME values as the original.
Reads one IQ2 tensor from each file, unpacks SoA back to AoS, byte-compares."""
import struct, sys, numpy as np
sys.path.insert(0, '/home/claude/repack_dir')
QK_K=256; BLK=66
def _rs(f):
    (n,)=struct.unpack("<Q",f.read(8)); return f.read(n).decode("utf-8","replace")
def _skip(f,t):
    if t in (0,1,7): f.read(1)
    elif t in (2,3): f.read(2)
    elif t in (4,5,6): f.read(4)
    elif t in (10,11,12): f.read(8)
    elif t==8: _rs(f)
    elif t==9:
        (et,)=struct.unpack("<I",f.read(4)); (n,)=struct.unpack("<Q",f.read(8))
        for _ in range(n): _skip(f,et)
def scan(path):
    with open(path,"rb") as f:
        assert f.read(4)==b"GGUF"; struct.unpack("<I",f.read(4))
        (nt,)=struct.unpack("<Q",f.read(8)); (nkv,)=struct.unpack("<Q",f.read(8))
        align=32
        for _ in range(nkv):
            k=_rs(f); (vt,)=struct.unpack("<I",f.read(4))
            if k=="general.alignment" and vt==4: (align,)=struct.unpack("<I",f.read(4))
            else: _skip(f,vt)
        ts=[]
        for _ in range(nt):
            name=_rs(f); (nd,)=struct.unpack("<I",f.read(4))
            dims=[struct.unpack("<Q",f.read(8))[0] for _ in range(nd)]
            (tt,)=struct.unpack("<I",f.read(4)); (off,)=struct.unpack("<Q",f.read(8))
            ne=1
            for d in dims: ne*=d
            ts.append((name,tt,off,ne))
        ds=f.tell()
    return ts, ds+((align-(ds%align))%align)
a,da=scan(sys.argv[1]); b,db=scan(sys.argv[2])
assert da==db, (da,db)
tgt=[(n,t,o,ne) for n,t,o,ne in a if t==16][:2]
print(f"data_start identical: {da==db}")
ok=True
for name,t,off,ne in tgt:
    nblk=ne//QK_K
    NB=20000
    with open(sys.argv[1],"rb") as f:
        f.seek(da+off); packed=np.frombuffer(f.read(NB*BLK),dtype=np.uint8).reshape(NB,BLK)
    with open(sys.argv[2],"rb") as f:
        f.seek(db+off); q=np.frombuffer(f.read(NB*64),dtype=np.uint8).reshape(NB,64)
        f.seek(db+off+nblk*64); d=np.frombuffer(f.read(NB*2),dtype=np.uint8).reshape(NB,2)
    rebuilt=np.concatenate([d,q],axis=1)
    same=np.array_equal(rebuilt,packed)
    tt=[x[1] for x in b if x[0]==name][0]
    print(f"  {name}: soa_type={tt} first{NB}blk unpack==original: {'EXACT' if same else '** MISMATCH **'}")
    ok &= same
print("VERDICT:", "values preserved" if ok else "** CORRUPTION **")
