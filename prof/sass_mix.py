import re, sys, collections
if len(sys.argv) != 3:
    sys.exit('usage: sass_mix.py SASS_DUMP.txt FUNCTION_SUBSTRING (the dump: cuobjdump -sass of the MoE TU)')
txt = open(sys.argv[1]).read()
funcs = re.split(r'\n\s*Function : ', txt)
def classify(op):
    b = op.split('.')[0]
    if b in ('IDP4A','DP4A') or b.startswith('IDP'): return 'dp4a (int MMA work)'
    if b=='LDC': return 'LDC (constant LUT gather)'        # IQ2 grid/signs table
    if op=='IMAD.SHL' or b in ('PRMT','LOP3','SHF','BFE','SGXT','VABSDIFF4','VSETP','FLO','POPC'): return 'decode (bit ops)'
    if b=='LDS': return 'LDS (smem act read)'
    if b=='LDG': return 'LDG (global read)'
    if b=='LDGSTS': return 'LDGSTS (cp.async stage)'
    if b=='STS': return 'STS (smem stage)'
    if b=='STG': return 'STG (out)'
    if b.startswith('LDL') or b.startswith('STL'): return 'LOCAL (spill!)'
    if b in ('FADD','FFMA','FMUL','I2F','F2F','FSEL','FSETP'): return 'float fold'
    if b=='SHFL': return 'shfl (warp reduce)'
    if b in ('IMAD','IADD3','IADD','ISETP','LEA','IMNMX','IABS'): return 'addr/int'
    if b in ('BRA','BSSY','BSYNC','EXIT','BAR','NOP','WARPSYNC','CALL','RET','JMP'): return 'ctrl'
    if b in ('MOV','SEL','SHFL','PRMT','R2P','P2R','CS2R','S2R','R2UR','UMOV','ULDC','MOV32I'): return 'mov/misc'
    return 'other'
want = sys.argv[2]
for f in funcs:
    name = f.split('\n')[0].strip()
    if want not in name: continue
    ops = re.findall(r'/\*[0-9a-f]{4,}\*/\s+@?!?P[0-9T]\s+([A-Z][A-Z0-9_.]*)|'
                     r'/\*[0-9a-f]{4,}\*/\s+([A-Z][A-Z0-9_.]*)', f)
    ops=[a or b for a,b in ops]
    cat=collections.Counter(); raw=collections.Counter()
    for o in ops:
        raw[o]+=1; cat[classify(o)]+=1
    total=sum(cat.values())
    print('=== %s' % name[:88])
    print('    total SASS: %d' % total)
    for k,v in cat.most_common():
        print('    %-28s %6d  (%5.1f%%)' % (k,v,100.0*v/total))
    print('    -- top raw --')
    for k,v in raw.most_common(10): print('       %-20s %5d'%(k,v))
    print()
