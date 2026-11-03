"""Descarga TEMP de AEMET y publica productos estáticos, sin enviar claves al navegador."""
import argparse,datetime,json,os,shutil,subprocess,sys,tempfile,time
from pathlib import Path
from urllib.request import Request,urlopen,HTTPRedirectHandler,build_opener
from urllib.parse import urlparse

API='https://opendata.aemet.es/opendata/api/observacion/convencional/mensajes/tipomensaje/temp'
class SinRedireccion(HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):return None

def descargar(destino):
    clave=os.environ.get('AEMET_API_KEY','').strip()
    if not clave:raise RuntimeError('Falta el secreto AEMET_API_KEY en GitHub.')
    # La clave solo viaja a este endpoint HTTPS; no se reenvía en la segunda descarga.
    opener=build_opener(SinRedireccion())
    with opener.open(Request(API,headers={'api_key':clave}),timeout=45) as r:doc=json.load(r)
    if int(doc.get('estado',0))!=200:raise RuntimeError('AEMET no ha devuelto datos TEMP disponibles.')
    url=doc.get('datos','');u=urlparse(url)
    if u.scheme!='https' or u.hostname not in ['opendata.aemet.es','www.aemet.es','aemet.es']:
        raise RuntimeError('URL de datos AEMET no reconocida.')
    with opener.open(Request(url),timeout=60) as r:data=r.read(80_000_001)
    if len(data)>80_000_000:raise RuntimeError('El archivo excede el límite de descarga.')
    destino.write_bytes(data)

def generar(archivo,out,ahora=None):
    from interpretar_sondeo import elegir_mensaje
    fecha,_,_,_=elegir_mensaje(archivo)
    instante=datetime.datetime.strptime(fecha,'%Y%m%d%H').replace(tzinfo=datetime.timezone.utc)
    now=ahora or datetime.datetime.now(datetime.timezone.utc)
    if instante>now+datetime.timedelta(minutes=90):raise RuntimeError('El boletín tiene una fecha futura inesperada.')
    with tempfile.TemporaryDirectory() as tmp:
        cmd=[sys.executable,str(Path(__file__).with_name('interpretar_sondeo.py')),str(archivo),'--hora',fecha,'--salida',tmp]
        subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL)
        base='Huelva_'+fecha+'_indices'
        doc=json.loads((Path(tmp)/(base+'.json')).read_text())
        if doc['station']!='08383':raise RuntimeError('Estación incorrecta.')
        out.mkdir(parents=True,exist_ok=True)
        for ext in ['png','csv','txt','json']:shutil.copyfile(Path(tmp)/(base+'.'+ext),out/('ultimo.'+ext))
        doc['nominalTime']=instante.isoformat();doc['generatedAt']=now.isoformat();doc['image']='ultimo.png'
        (out/'estado.json').write_text(json.dumps(doc,ensure_ascii=False,allow_nan=False))
        shutil.copyfile(Path(__file__).with_name('web')/'index.html',out/'index.html')
        (out/'.nojekyll').touch()
    print('Preparado Huelva 08383 · '+fecha+' UTC')

def main():
    p=argparse.ArgumentParser();p.add_argument('--archivo',help='Prueba con un TEMP local, sin consultar AEMET');p.add_argument('--salida',default='_site');args=p.parse_args()
    if args.archivo:generar(Path(args.archivo),Path(args.salida));return
    with tempfile.TemporaryDirectory() as tmp:
        f=Path(tmp)/'temp.tar.gz'
        for intento in range(3):
            try:descargar(f);break
            except Exception:
                if intento==2:raise RuntimeError('No se pudo descargar TEMP de AEMET tras tres intentos. Se conserva la publicación anterior.') from None
                time.sleep(5)
        generar(f,Path(args.salida))
if __name__=='__main__':main()
