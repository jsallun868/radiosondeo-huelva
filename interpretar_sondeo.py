"""Interpretación determinista de perfiles AEMET TEMP. No utiliza IA ni servicios de texto."""
import argparse
import json
import re
import tarfile
from pathlib import Path
import numpy as np
from indices import calcular
import metpy.calc as mp
from metpy.units import units

FIELDS = ['pressure','nonCoordinateGeopotentialHeight','airTemperature','dewpointTemperature','windDirection','windSpeed']

def elegir_mensaje(archivo, estacion='08383', hora=None):
    candidatos=[]
    with tarfile.open(archivo, 'r:*') as tar:
        for miembro in tar:
            if not miembro.isfile() or not miembro.name.endswith('.json'):
                continue
            if miembro.size > 40_000_000:
                continue
            doc=json.load(tar.extractfile(miembro))
            entradas=doc.get('messages', [])
            if not isinstance(entradas,list) or not all(isinstance(x,dict) for x in entradas):
                continue
            meta={x.get('key'):x.get('value') for x in entradas if x.get('key') in ['blockNumber','stationNumber']}
            if meta.get('blockNumber') is None or meta.get('stationNumber') is None:
                continue
            codigo=f"{int(meta['blockNumber']):02d}{int(meta['stationNumber']):03d}"
            fecha=re.search(r'(\d{10})_',Path(miembro.name).name)
            if codigo != estacion or not fecha or (hora and fecha[1] != hora):
                continue
            # Un boletín con varios subsets requiere separación explícita.
            if sum(x.get('key')=='blockNumber' for x in entradas) != 1:
                continue
            candidatos.append((fecha[1], 'IUK' in miembro.name, miembro.name, entradas))
    if not candidatos:
        raise ValueError('No hay un mensaje compatible para la estación y fecha solicitadas.')
    return max(candidatos,key=lambda x:(x[0],x[1]))

def extraer(entradas):
    filas=[]; fila=None; meta={}
    for x in entradas:
        k=x.get('key');v=x.get('value')
        if fila is None and k in ['year','month','day','hour','minute','second','latitude','longitude']:
            meta[k]=v
        if k=='timePeriod':
            if fila and 'pressure' in fila: filas.append(fila)
            fila={}
        if fila is not None: fila[k]=v
    if fila and 'pressure' in fila: filas.append(fila)
    a=np.array([[f.get(k) if f.get(k) is not None else np.nan for k in FIELDS] for f in filas],dtype=float)
    if not len(a): raise ValueError('No se han encontrado niveles.')
    # Viento ausente no invalida el perfil termodinámico.
    a=a[np.all(np.isfinite(a[:,:4]),axis=1)]
    a=a[(a[:,0]>0)&(a[:,2]>150)&(a[:,2]<340)&(a[:,3]>100)&(a[:,3]<=a[:,2]+.5)]
    a=a[np.argsort(-a[:,0],kind='stable')]
    a=a[np.r_[True,np.diff(a[:,0])<0]]
    if len(a)<20 or np.any(np.diff(a[:,1]) < -30):
        raise ValueError('Perfil insuficiente o alturas inconsistentes.')
    return a,meta,len(filas)

def fmt(x,dec=0):
    return f'{x:.{dec}f}'.replace('.',',')

def invertir(a):
    # Agrupa incrementos térmicos en una malla de 100 m; exige >=0,5 °C y >=100 m.
    # Se excluyen intervalos con huecos observados >300 m.
    iz=np.argsort(a[:,1]);z=a[iz,1];t=a[iz,2]-273.15
    grid=np.arange(np.ceil(z[0]/100)*100,min(5000,z[-1])+1,100)
    if len(grid)<2:return []
    tg=np.interp(grid,z,t); gaps=np.diff(z)
    valid=np.ones(len(grid)-1,dtype=bool)
    for i in np.where(gaps>300)[0]:valid &= ~((grid[:-1]<z[i+1])&(grid[1:]>z[i]))
    ids=np.where((np.diff(tg)>0)&valid)[0]
    result=[]
    for g in np.split(ids,np.where(np.diff(ids)>1)[0]+1):
        if len(g) and tg[g[-1]+1]-tg[g[0]]>=.5:
            result.append({'base_m':float(grid[g[0]]),'top_m':float(grid[g[-1]+1]),'increase_c':float(tg[g[-1]+1]-tg[g[0]])})
    return sorted(result,key=lambda x:x['increase_c'],reverse=True)

def interpretar(a):
    p=a[:,0]/100*units.hPa;t=(a[:,2]-273.15)*units.degC;td=(a[:,3]-273.15)*units.degC
    rh=mp.relative_humidity_from_dewpoint(t,td).magnitude*100
    parcelas=mp.parcel_profile(p,t[0],td[0])
    cape,cin=mp.cape_cin(p,t,td,parcelas)
    inv=invertir(a)
    partes=[];metricas={'levels':len(a),'top_m':float(a[-1,1]),'top_hpa':float(p[-1].m),'surface_cape_jkg':float(cape.m),'inversions_below_5km':inv}
    if inv:
        x=inv[0]
        partes.append(('Una capa que frena las ascendencias.',f"Entre aproximadamente {fmt(x['base_m'])} y {fmt(x['top_m'])} m, la temperatura aumenta {fmt(x['increase_c'],1)} °C con la altura. Esta inversión dificulta la mezcla vertical y puede limitar el crecimiento de las nubes que ascienden desde abajo."))
    else:
        partes.append(('Inversiones térmicas.', 'No se detectan inversiones de al menos 0,5 °C en la malla de 100 m analizada por debajo de 5 km. Pueden existir capas más finas que este criterio no identifica.'))
    def nivel(h):
        if h<a[0,1] or h>a[-1,1]:return None
        i=int(np.argmin(abs(a[:,1]-h)))
        if abs(a[i,1]-h)>150:return None
        return i
    hum=[]
    for h in [3000,5000]:
        i=nivel(h)
        if i is not None:
            metricas[f'rh_{h}m_percent']=float(rh[i]);hum.append((h,float(rh[i])))
    if hum:
        descr='; '.join(f"a {fmt(h/1000)} km, aproximadamente {fmt(r)} %" for h,r in hum)
        texto='Humedad relativa: '+descr+'. '
        if all(r<30 for _,r in hum):texto+='En esos niveles el aire está lejos de la saturación.'
        elif any(r>=90 for _,r in hum):texto+='Al menos uno de esos niveles presenta humedad elevada, compatible con nubosidad; no confirma por sí sola la presencia de nubes.'
        else:texto+='Estos valores describen niveles concretos; no determinan por sí solos el estado del cielo.'
        partes.append(('Humedad en niveles medios.',texto))
    # Un perfil truncado no permite afirmar que la CAPE total es cero.
    cobertura=p[-1].m<=150
    if not cobertura:
        texto=f"La CAPE de la parcela superficial integrada hasta {fmt(p[-1].m)} hPa es {fmt(cape.m)} J/kg. El perfil termina demasiado bajo para valorar toda la convección profunda."
    elif cape.m<1:
        texto='La CAPE calculada para una parcela que parte de superficie es de 0 J/kg en el tramo observado. No aparece flotabilidad positiva para esa parcela. Esto no descarta lluvia, convección elevada ni cambios posteriores.'
    else:
        texto=f"La CAPE calculada para una parcela que parte de superficie es de {fmt(cape.m)} J/kg en el tramo observado. Hay energía disponible para ascendencias, pero su aprovechamiento depende de los mecanismos de disparo y de las capas que puedan frenarlas. No es una probabilidad de tormenta."
    partes.append(('Energía para las ascendencias.',texto))
    nombres=['norte','noreste','este','sureste','sur','suroeste','oeste','noroeste'];viento=[]
    for h in [1000,10000]:
        i=nivel(h)
        if i is not None and np.all(np.isfinite(a[i,4:])) and 0<=a[i,4]<=360 and a[i,5]>=0:
            d=nombres[int((a[i,4]%360+22.5)//45)%8];vel=a[i,5]*3.6
            metricas[f'wind_{h}m']={'direction':d,'speed_kmh':float(vel)}
            viento.append(f"a {fmt(h/1000)} km, del {d} a unos {fmt(vel)} km/h")
    if viento:partes.append(('Viento en altura.','El viento sopla '+ '; '.join(viento)+'. Son valores de niveles concretos; la hodógrafa muestra la evolución entre ellos.'))
    aviso='Alturas sobre el nivel del mar. Observación de Huelva en ese momento: no es una predicción para La Línea. No se extrapolan valores por encima del perfil disponible.'
    return {'metrics':metricas,'sections':[{'title':x,'text':y} for x,y in partes],'note':aviso},parcelas.to('degC').m,rh

def dibujar(a,meta,fecha,estacion,res,parcelas,rh,destino):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from metpy.plots import SkewT,Hodograph
    import textwrap
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
    fig=plt.figure(figsize=(15,16),facecolor='#fff8df')
    fig.text(.055,.966,'LA ATMÓSFERA EN ALTURA',fontsize=24,weight='bold',color='#473b2a')
    fig.text(.055,.939,f'Radiosondeo observado · Huelva–Ronda Este · OMM {estacion}',fontsize=16,color='#473b2a')
    fecha_txt=f'{fecha[6:8]}/{fecha[4:6]}/{fecha[:4]} · {fecha[8:10]} UTC'
    lanzamiento=f"{int(meta['hour']):02d}:{int(meta['minute']):02d}:{int(meta.get('second',0)):02d} UTC"
    fig.text(.055,.917,fecha_txt+'   |   Lanzamiento: '+lanzamiento,fontsize=12,color='#6a624e')
    sk=SkewT(fig,rotation=45,rect=[.075,.39,.59,.48]);sk.ax.set_facecolor('#fffdf6')
    p=a[:,0]/100*units.hPa
    sk.plot(p,(a[:,2]-273.15)*units.degC,color='#d23837',lw=2,label='Temperatura')
    sk.plot(p,(a[:,3]-273.15)*units.degC,color='#188258',lw=2,label='Punto de rocío')
    sk.plot(p,parcelas*units.degC,color='#6b4c9a',lw=1.6,ls='--',label='Parcela superficial')
    sk.ax.set_ylim(max(1050,p[0].m),max(50,min(100,p[-1].m)));sk.ax.set_xlim(-40,40)
    sk.ax.set_xlabel('Temperatura (°C)');sk.ax.set_ylabel('Presión (hPa)')
    sk.plot_dry_adiabats(alpha=.22,colors='#b58466');sk.plot_moist_adiabats(alpha=.22,colors='#598d7c');sk.plot_mixing_lines(alpha=.22,colors='#7d9972');sk.ax.grid(alpha=.4)
    ok=np.isfinite(a[:,4])&np.isfinite(a[:,5])&(a[:,5]>=0)&(a[:,4]>=0)&(a[:,4]<=360)
    b=a[ok];u,v=mp.wind_components(b[:,5]*units('m/s'),b[:,4]*units.deg)
    if len(b):
        ids=np.unique([abs(b[:,0]/100-q).argmin() for q in np.geomspace(min(1000,p[0].m),max(100,p[-1].m),27)])
        sk.plot_barbs(b[ids,0]/100*units.hPa,u[ids].to('knots'),v[ids].to('knots'),length=5.5)
        ax=fig.add_axes([.735,.69,.22,.18],facecolor='#fffdf6');lim=max(15,np.ceil(np.max(abs(np.r_[u.m,v.m]))/5)*5+5)
        hod=Hodograph(ax,component_range=lim);hod.add_grid(increment=5,alpha=.25);hod.plot_colormapped(u,v,b[:,1]/1000,cmap='viridis')
        ax.set_title('Hodógrafa · viento (m/s)',fontsize=12);ax.set_xlabel('Componente U');ax.set_ylabel('Componente V')
    sk.ax.legend(loc='upper left',fontsize=9)
    fig.text(.73,.625,'DATOS CLAVE',fontsize=13,weight='bold',color='#473b2a')
    m=res['metrics'];idx=res['indices']
    labels=['Superficie','ML 100','MU 300']
    keys=['surface','mixed_100hpa','unstable_300hpa']
    def cell(v):return '—' if v is None else fmt(v,1)
    rows=[]
    for label,field in [('CAPE (J/kg)','cape_jkg'),('CIN (J/kg)','cin_jkg'),('LCL (hPa)','lcl_hpa'),('LFC (hPa)','lfc_hpa'),('EL (hPa)','el_hpa')]:
        rows.append([label]+[cell(idx[k][field]) if idx[k] else '—' for k in keys])
    ax_table=fig.add_axes([.715,.465,.265,.14]);ax_table.axis('off')
    table=ax_table.table(cellText=rows,colLabels=['Índice']+labels,loc='center',cellLoc='center',colWidths=[.37,.23,.20,.20]);table.auto_set_font_size(False);table.set_fontsize(8.5);table.scale(1,1.65)
    for (rr,cc),c in table.get_celld().items():c.set_facecolor('#f2e5bd' if rr==0 else '#fffdf6');c.set_edgecolor('#d8cda9')
    fig.text(.73,.435,'LI superficial: '+cell(idx['surface']['li_c'])+' °C',fontsize=11)
    fig.text(.73,.415,'Índice K: '+cell(idx['k_c'])+' °C',fontsize=11)
    fig.text(.73,.395,'Total Totals: '+cell(idx['total_totals_c'])+' °C',fontsize=11)
    fig.text(.73,.370,'—: no existe / no aplicable / sin datos',fontsize=8.5,color='#6a624e')
    fig.text(.055,.335,'¿QUÉ NOS CUENTA ESTE SONDEO?',fontsize=18,weight='bold',color='#473b2a')
    y=.303
    for sec in res['sections']:
        fig.text(.055,y,sec['title'],fontsize=12,weight='bold',color='#473b2a')
        wrapped=textwrap.fill(sec['text'],145)
        fig.text(.055,y-.016,wrapped,fontsize=10.5,color='#473b2a',va='top',linespacing=1.4)
        y-=max(.055,.025+len(wrapped.splitlines())*.013)
    fig.text(.055,.052,'Fuente: AEMET OpenData · TEMP. Interpretación mediante reglas meteorológicas en Python; sin IA.',fontsize=10,color='#6a624e')
    fig.text(.055,.035,textwrap.fill(res['note'],165),fontsize=9,color='#6a624e',va='top')
    fig.text(.055,.012,'CAPE/CIN con temperatura virtual; niveles LCL/LFC/EL con temperatura ordinaria. ML: capa mezclada; MU: parcela más inestable.',fontsize=8,color='#6a624e')
    fig.savefig(destino,dpi=180,facecolor=fig.get_facecolor());plt.close(fig)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archivo',help='Archivo tar.gz de mensajes TEMP AEMET')
    parser.add_argument('--hora',help='Fecha/hora nominal AAAAMMDDHH; por defecto la última del archivo')
    parser.add_argument('--salida',default='salida')
    args=parser.parse_args()
    fecha,_,nombre,entradas=elegir_mensaje(args.archivo,'08383',args.hora)
    a,meta,total=extraer(entradas);res,parcelas,rh=interpretar(a)
    res['indices']=calcular(a)
    idx=res['indices']
    caps=[(name,idx[key]) for name,key in [('superficie','surface'),('capa mezclada de 100 hPa','mixed_100hpa'),('parcela más inestable de los 300 hPa inferiores','unstable_300hpa')] if idx[key] is not None]
    texto='CAPE: '+ '; '.join(name+', '+fmt(v['cape_jkg'])+' J/kg' for name,v in caps)+'. '
    if all(v['cape_jkg']<1 for _,v in caps):texto+='No se detecta energía positiva para estas parcelas en el perfil medido. Esto no descarta lluvia ni cambios posteriores. La CIN se indica como no aplicable al no existir CAPE.'
    else:texto+='Hay energía positiva en alguna parcela. Su presencia no garantiza tormentas: también influyen el disparo, la inhibición y la evolución posterior.'
    li=idx['surface']['li_c']
    if li is not None and li>0:texto+=' El LI positivo indica que la parcela superficial llega a 500 hPa más fría que el entorno.'
    elif li is not None and li<0:texto+=' El LI negativo indica que la parcela superficial llega a 500 hPa más cálida que el entorno.'
    res['sections'][2]={'title':'Estabilidad: comparación de parcelas.','text':texto}
    res.update({'station':'08383','nominal_utc':fecha,'launch':meta,'source_member':nombre,'received_levels':total,'method':'reglas deterministas; sin IA'})
    out=Path(args.salida);out.mkdir(parents=True,exist_ok=True);base='Huelva_'+fecha+'_indices'
    (out/(base+'.json')).write_text(json.dumps(res,ensure_ascii=False,indent=2))
    texto='RADIOSONDEO OBSERVADO DE HUELVA · '+fecha+' UTC\n\n'+'\n\n'.join(s['title']+' '+s['text'] for s in res['sections'])+'\n\n'+res['note']+'\n'
    texto+='\nÍNDICES Y MÉTODO\n'+json.dumps(res['indices'],ensure_ascii=False,indent=2)+'\n'
    (out/(base+'.txt')).write_text(texto)
    np.savetxt(out/(base+'.csv'),np.column_stack([a[:,0]/100,a[:,1],a[:,2]-273.15,a[:,3]-273.15,a[:,4],a[:,5],rh]),delimiter=',',header='pressure_hpa,height_m_asl,temperature_c,dewpoint_c,wind_direction_deg,wind_speed_ms,relative_humidity_percent',comments='')
    dibujar(a,meta,fecha,'08383',res,parcelas,rh,out/(base+'.png'))
    print(texto)

if __name__=='__main__':main()
