#!/usr/bin/env python3
"""Independent numerical charge-block/Schmidt Fisher decomposition of a saved state.

Input is flux-major and contains a normalized pure ground vector psi, orthogonal
mass derivative z, sorted per-flux integer masks occ and list flux. The output
contains numerical diagnostics, NOT a rigorous bound on the exact physical state.
"""
from __future__ import annotations
import argparse
import itertools
import json
import math
from pathlib import Path
import numpy as np


def pattern_masks(length, particles):
    if particles<0 or particles>length: return []
    return sorted(sum(1<<j for j in c) for c in itertools.combinations(range(length),particles))


def reshape_charge(psi,z,occ,flux,n,k,region=6):
    half=n//2
    rmasks=pattern_masks(region,k)
    emasks=pattern_masks(n-region,half-k)
    rindex=np.full(1<<region,-1,dtype=np.int16)
    rindex[rmasks]=np.arange(len(rmasks),dtype=np.int16)
    rbits=occ & np.uint32((1<<region)-1)
    charge=np.array([int(x).bit_count() for x in rbits],dtype=np.int8)
    take=np.flatnonzero(charge==k)
    exterior=(occ[take]>>np.uint32(region)).astype(np.uint32)
    emasks=np.asarray(emasks,dtype=np.uint32)
    col=np.searchsorted(emasks,exterior)
    if len(col) and (np.any(col>=len(emasks)) or not np.array_equal(emasks[col],exterior)):
        raise AssertionError('Invalid exterior pattern')
    row=rindex[rbits[take]]
    if np.any(row<0): raise AssertionError('Invalid regional pattern')
    nocc=len(occ)
    B=np.zeros((len(flux)*len(rmasks),len(emasks)),dtype=np.float64)
    Z=np.zeros_like(B)
    for li in range(len(flux)):
        B[li*len(rmasks)+row,col]=psi[li*nocc+take]
        Z[li*len(rmasks)+row,col]=z[li*nocc+take]
    return B,Z


def block_fisher(B,Z,charge_k):
    """Rectangular SVD formula for real matrices at simple positive spectra."""
    U,s,Vh=np.linalg.svd(B,full_matrices=False)
    d=len(s)
    p=float(np.sum(B*B)); pp=float(2*np.sum(B*Z))
    if p==0: raise ValueError(f'Zero charge weight in block {charge_k}')
    T=U.T@Z@Vh.T
    lam=s*s; diag=np.diag(T)
    spec=4*float(diag@diag)
    charge=pp*pp/p
    c=pp/(2*p)
    within=4*float(np.sum((diag-c*s)**2))
    rotation=0.
    first_rot=0.
    cross=0.
    for i in range(d):
        for j in range(i+1,d):
            den=float(lam[i]+lam[j])
            if den == 0: continue
            pair=4*float((s[j]*T[i,j]+s[i]*T[j,i])**2/den)
            rotation+=pair
            if i==0:
                first_rot+=pair
                if s[0] !=0:
                    ratio=s[j]/s[0]
                    cross+=8*float(ratio*T[j,0]*T[0,j]/(1+ratio**2))
    leftnull=Z-U@(U.T@Z)
    nullrot=4*float(np.sum(leftnull**2))
    rotation+=nullrot
    firstnull=4*float(np.linalg.norm(leftnull@Vh[0,:])**2)
    first_rot+=firstnull
    # Completely equal strictly positive eigenvalues require the whole
    # derivative within their projector as spectral, not arbitrary SVD axes.
    max_equal=0.
    for i in range(d):
        for j in range(i+1,d):
            if lam[i]>0 and lam[i]==lam[j]:
                raise ValueError('Exactly degenerate Schmidt pair: group eigenspace before decomposition')
            if lam[i]>0 and lam[j]>0:
                max_equal=max(max_equal,float(min(lam[i],lam[j])/max(lam[i],lam[j])))
    return dict(k=charge_k,sector_probability=p,sector_probability_derivative=pp,
        rank_svd=d,largest_schmidt_weight=float(lam[0]/p),
        smallest_positive_schmidt_sq=float(np.min(lam[lam>0])) if np.any(lam>0) else None,
        spectral=spec,charge=charge,within=within,
        rotation=rotation,first_mode_rotation=first_rot,
        first_mode_to_left_null=firstnull,total_to_left_null=nullrot,
        first_mode_signed_interference=cross,full_fisher=spec+rotation,
        within_charge_identity_error=within-(spec-charge),
        max_positive_eigenvalue_ratio=max_equal)


def analyze(path):
    with np.load(path,allow_pickle=False) as a:
        for key in ('occ','psi','z','mass'):
            if key not in a: raise ValueError('Missing '+key+' in '+str(path))
        occ=np.asarray(a['occ'],dtype=np.uint32)
        psi=np.asarray(a['psi'],dtype=np.float64)
        z=np.asarray(a['z'],dtype=np.float64)
        mass=np.asarray(a['mass'],dtype=np.float64)
        metadata=json.loads(str(a['metadata'].item())) if 'metadata' in a else {}
        n=int(metadata.get('N',int(occ.max()).bit_length()))
        flux=np.asarray(a['flux'],dtype=np.int32) if 'flux' in a else np.arange(-4,5,dtype=np.int32)
    if n%2 or n<8 or n>30: raise ValueError('Invalid inferred/provided N')
    if not np.array_equal(np.sort(occ),occ) or np.unique(occ).size != len(occ):
        raise ValueError('occ must be strictly increasing and unique')
    if len(occ)!=math.comb(n,n//2) or any(int(x).bit_count()!=n//2 for x in occ):
        raise ValueError('Bad half-filled occupancy basis')
    if psi.ndim!=1 or psi.size!=len(occ)*len(flux) or z.shape!=psi.shape:
        raise ValueError('Bad psi/z dimensions or flux count')
    if mass.size not in (len(occ),psi.size): raise ValueError('Invalid mass length')
    mass_all=np.tile(mass,len(flux)) if mass.size==len(occ) else mass
    norm=float(np.linalg.norm(psi))
    orth=float(np.dot(psi,z))
    if abs(norm-1)>1e-6 or abs(orth)>1e-6:
        raise ValueError(f'Unnormalized or noncentered vector: norm={norm} dot={orth}')
    if not np.all(np.isfinite(psi)) or not np.all(np.isfinite(z)):
        raise ValueError('Nonfinite vector')
    blocks=[]
    for k in range(7):
        if n//2-k<0 or n//2-k>n-6: continue
        B,Z=reshape_charge(psi,z,occ,flux,n,k)
        blocks.append(block_fisher(B,Z,k))
    Fsp=sum(b['spectral'] for b in blocks)
    Fcharge=sum(b['charge'] for b in blocks)
    Fwithin=sum(b['within'] for b in blocks)
    Frot=sum(b['rotation'] for b in blocks)
    Ffirst=sum(b['first_mode_rotation'] for b in blocks)
    Fq=Fsp+Frot
    result=dict(source_path=str(path),N=n,flux=flux.tolist(),
        state_norm=norm,derivative_orthogonality=orth,
        mass_expectation=float(psi@(mass_all*psi)),
        ground_state_derivative_norm=float(np.linalg.norm(z)),
        full_global_pure_state_Fisher=4*float(np.dot(z,z)),
        regional_Fisher=Fq,spectral_Fisher=Fsp,rotation_Fisher=Frot,
        spectral_fraction=Fsp/Fq,charge_Fisher=Fcharge,within_charge_spectral_Fisher=Fwithin,
        first_mode_rotation=Ffirst,first_mode_rotation_fraction=Ffirst/Frot,
        signed_interference=sum(b['first_mode_signed_interference'] for b in blocks),
        check_probability_sum=sum(b['sector_probability'] for b in blocks),
        check_spectral_charge_within=Fsp-Fcharge-Fwithin,
        numerical_only=True,blocks=blocks,
        warning='No rigorous state/response/projector enclosure, no full-gap certification, no 6–8% theorem.')
    return result


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('state',type=Path)
    ap.add_argument('--output',type=Path,default=Path('state_analysis.json'))
    args=ap.parse_args(argv)
    result=analyze(args.state)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(f'N={result["N"]} FQ={result["regional_Fisher"]:.12g} '
          f'Fsp={result["spectral_Fisher"]:.12g} '
          f'Fsp/FQ={result["spectral_fraction"]:.9%} '
          f'first-rotation={result["first_mode_rotation_fraction"]:.8%}',flush=True)
    print('SAVED',args.output,flush=True)

if __name__=='__main__': main()
