"""Índices MetPy con selección explícita de parcelas y niveles ausentes como null."""
import numpy as np
import metpy.calc as mp
from metpy.units import units

def valor(q):
    v=float(np.asarray(q.m).reshape(-1)[0]);return v if np.isfinite(v) else None

def calcular(a):
    p=a[:,0]/100*units.hPa;t=(a[:,2]-273.15)*units.degC;td=(a[:,3]-273.15)*units.degC
    def analizar(pp,tt,dd):
        pr,te,de,parcel=mp.parcel_profile_with_lcl(pp,tt,dd)
        cape,cin=mp.cape_cin(pr,te,de,parcel,which_lfc='bottom',which_el='top')
        lcl,_=mp.lcl(pp[0],tt[0],dd[0])
        # LFC/EL de temperatura ordinaria: explícitamente separados de la integración virtual.
        lfc,_=mp.lfc(pr,te,de,parcel_temperature_profile=parcel,which='bottom')
        el,_=mp.el(pr,te,de,parcel_temperature_profile=parcel,which='top')
        c=valor(cape);lf=valor(lfc)
        return {'cape_jkg':c,'cin_jkg':valor(cin) if c and c>0 else None,'lcl_hpa':valor(lcl),'lfc_hpa':lf,'el_hpa':valor(el),'start_hpa':valor(pp[0]),'li_c':valor(mp.lifted_index(pr,te,parcel)) if pp[-1].m<=500<=pp[0].m else None}
    out={'surface':analizar(p,t,td),'mixed_100hpa':None,'unstable_300hpa':None}
    if p[0].m-p[-1].m>=100:
        pm,tm,dm=mp.mixed_parcel(p,t,td,depth=100*units.hPa)
        mask=p<(p[0]-100*units.hPa)
        out['mixed_100hpa']=analizar(np.r_[pm.m,p[mask].m]*units.hPa,np.r_[tm.to('degC').m,t[mask].m]*units.degC,np.r_[dm.to('degC').m,td[mask].m]*units.degC)
    if p[0].m-p[-1].m>=300:
        _,_,_,i=mp.most_unstable_parcel(p,t,td,depth=300*units.hPa)
        out['unstable_300hpa']=analizar(p[i:],t[i:],td[i:])
    out['k_c']=valor(mp.k_index(p,t,td)) if p[-1].m<=500 and p[0].m>=850 else None
    out['total_totals_c']=valor(mp.total_totals_index(p,t,td)) if p[-1].m<=500 and p[0].m>=850 else None
    out['method']='MetPy: CAPE/CIN con temperatura virtual; LFC inferior y EL superior en integración. LCL/LFC/EL tabulados con temperatura ordinaria (LFC inferior/EL superior). CIN nula se muestra como no aplicable cuando CAPE=0. Parcela MU: máximo theta-e en 300 hPa inferiores.'
    return out
