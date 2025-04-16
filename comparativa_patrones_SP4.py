#%% Comparador ciclos de los patrones Superparamagneticos: NPM en Al2O3
import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#%% Funciones
def plot_ciclos_promedio(directorio):
    # Buscar recursivamente todos los archivos que coincidan con el patrón
    archivos = glob(os.path.join(directorio, '**', '*ciclo_promedio*.txt'), recursive=True)
    
    if not archivos:
        print(f"No se encontraron archivos '*ciclo_promedio.txt' en {directorio} o sus subdirectorios")
        return
    fig,ax=plt.subplots(figsize=(8, 6),constrained_layout=True)
    for archivo in archivos:
        try:
            # Leer los metadatos (primeras líneas que comienzan con #)
            metadatos = {}
            with open(archivo, 'r') as f:
                for linea in f:
                    if not linea.startswith('#'):
                        break
                    if '=' in linea:
                        clave, valor = linea.split('=', 1)
                        clave = clave.replace('#', '').strip()
                        metadatos[clave] = valor.strip()
            
            # Leer los datos numéricos
            datos = np.loadtxt(archivo, skiprows=9)  # Saltar las 8 líneas de encabezado/metadatos
            
            tiempo = datos[:, 0]
            campo = datos[:, 3]  # Campo en kA/m
            magnetizacion = datos[:, 4]  # Magnetización en A/m
            
            # Crear etiqueta para la leyenda
            nombre_base = os.path.basename(os.path.dirname(archivo))  # Nombre del subdirectorio
            etiqueta = f"{nombre_base}"
            
            # Graficar
            
            ax.plot(campo, magnetizacion, label=etiqueta)
        
        except Exception as e:
            print(f"Error procesando archivo {archivo}: {str(e)}")
            continue
    
    plt.xlabel('Campo magnético (kA/m)')
    plt.ylabel('Magnetización (A/m)')
    plt.title(f'Comparación de ciclos de histéresis {os.path.split(directorio)[-1]}')
    plt.grid(True)
    plt.legend()  # Leyenda fuera del gráfico
    plt.savefig('comparativa_ciclos_'+os.path.split(directorio)[-1]+'.png',dpi=300)
    plt.show()

def lector_resultados(path): 
    '''
    Para levantar archivos de resultados con columnas :
    Nombre_archivo	Time_m	Temperatura_(ºC)	Mr_(A/m)	Hc_(kA/m)	Campo_max_(A/m)	Mag_max_(A/m)	f0	mag0	dphi0	SAR_(W/g)	Tau_(s)	N	xi_M_0
    '''
    with open(path, 'rb') as f:
        codificacion = chardet.detect(f.read())['encoding']
        
    # Leer las primeras 6 líneas y crear un diccionario de meta
    meta = {}
    with open(path, 'r', encoding=codificacion) as f:
        for i in range(20):
            line = f.readline()
            if i == 0:
                match = re.search(r'Rango_Temperaturas_=_([-+]?\d+\.\d+)_([-+]?\d+\.\d+)', line)
                if match:
                    key = 'Rango_Temperaturas'
                    value = [float(match.group(1)), float(match.group(2))]
                    meta[key] = value
            else:
                match = re.search(r'(.+)_=_([-+]?\d+\.\d+)', line)
                if match:
                    key = match.group(1)[2:]
                    value = float(match.group(2))
                    meta[key] = value
                else:
                    # Capturar los casos con nombres de archivo en las últimas dos líneas
                    match_files = re.search(r'(.+)_=_([a-zA-Z0-9._]+\.txt)', line)
                    if match_files:
                        key = match_files.group(1)[2:]  # Obtener el nombre de la clave sin '# '
                        value = match_files.group(2)     # Obtener el nombre del archivo
                        meta[key] = value
                    
    # Leer los datos del archivo
    data = pd.read_table(path, header=14,
                         names=('name', 'Time_m', 'Temperatura',
                                'Remanencia', 'Coercitividad','Campo_max','Mag_max',
                                'frec_fund','mag_fund','dphi_fem',
                                'SAR','tau',
                                'N','xi_M_0'),
                         usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13),
                         decimal='.',
                         engine='python',
                         encoding=codificacion)
        
    files = pd.Series(data['name'][:]).to_numpy(dtype=str)
    time = pd.Series(data['Time_m'][:]).to_numpy(dtype=float)
    temperatura = pd.Series(data['Temperatura'][:]).to_numpy(dtype=float)
    Mr = pd.Series(data['Remanencia'][:]).to_numpy(dtype=float)
    Hc = pd.Series(data['Coercitividad'][:]).to_numpy(dtype=float)
    campo_max = pd.Series(data['Campo_max'][:]).to_numpy(dtype=float)
    mag_max = pd.Series(data['Mag_max'][:]).to_numpy(dtype=float)
    xi_M_0=  pd.Series(data['xi_M_0'][:]).to_numpy(dtype=float)
    SAR = pd.Series(data['SAR'][:]).to_numpy(dtype=float)
    tau = pd.Series(data['tau'][:]).to_numpy(dtype=float)
   
    frecuencia_fund = pd.Series(data['frec_fund'][:]).to_numpy(dtype=float)
    dphi_fem = pd.Series(data['dphi_fem'][:]).to_numpy(dtype=float)
    magnitud_fund = pd.Series(data['mag_fund'][:]).to_numpy(dtype=float)
    
    N=pd.Series(data['N'][:]).to_numpy(dtype=int)
    return meta, files, time,temperatura,Mr, Hc, campo_max, mag_max, xi_M_0, frecuencia_fund, magnitud_fund , dphi_fem, SAR, tau, N

#LECTOR CICLOS
def lector_ciclos(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()[:8]

    metadata = {'filename': os.path.split(filepath)[-1],
                'Temperatura':float(lines[0].strip().split('_=_')[1]),
        "Concentracion_g/m^3": float(lines[1].strip().split('_=_')[1].split(' ')[0]),
            "C_Vs_to_Am_M": float(lines[2].strip().split('_=_')[1].split(' ')[0]),
            "ordenada_HvsI ": float(lines[4].strip().split('_=_')[1].split(' ')[0]),
            'frecuencia':float(lines[5].strip().split('_=_')[1].split(' ')[0])}
    
    data = pd.read_table(os.path.join(os.getcwd(),filepath),header=7,
                        names=('Tiempo_(s)','Campo_(Vs)','Magnetizacion_(Vs)','Campo_(kA/m)','Magnetizacion_(A/m)'),
                        usecols=(0,1,2,3,4),
                        decimal='.',engine='python',
                        dtype={'Tiempo_(s)':'float','Campo_(Vs)':'float','Magnetizacion_(Vs)':'float',
                               'Campo_(kA/m)':'float','Magnetizacion_(A/m)':'float'})  
    t     = pd.Series(data['Tiempo_(s)']).to_numpy()
    H_Vs  = pd.Series(data['Campo_(Vs)']).to_numpy(dtype=float) #Vs
    M_Vs  = pd.Series(data['Magnetizacion_(Vs)']).to_numpy(dtype=float)#A/m
    H_kAm = pd.Series(data['Campo_(kA/m)']).to_numpy(dtype=float)*1000 #A/m
    M_Am  = pd.Series(data['Magnetizacion_(A/m)']).to_numpy(dtype=float)#A/m
    
    return t,H_Vs,M_Vs,H_kAm,M_Am,metadata

#%% Obtengo paths
# 81 kHz
ciclos_081=glob(os.path.join('081','**', '*ciclo_promedio*'),recursive=True)
ciclos_081.sort()
resultados_081=glob(os.path.join('081','**', '*resultados*'),recursive=True)
resultados_081.sort()

# 212 kHz   
ciclos_212=glob(os.path.join('212','**', '*ciclo_promedio*'),recursive=True)
ciclos_212.sort()
resultados_212=glob(os.path.join('212','**', '*resultados*'),recursive=True)
resultados_212.sort()

# 300 kHz   
ciclos_300=glob(os.path.join('300','**', '*ciclo_promedio*'),recursive=True)
ciclos_300.sort()
resultados_300=glob(os.path.join('300','**', '*resultados*'),recursive=True)
resultados_300.sort()

# %% Comparo SP2 (108/265) Marzo vs Abril
_,_,_,H_mar_108,M_mar_108,_=lector_ciclos(ciclos_SP2_marzo_108[1]) #elijo uno de los ciclos para comparar Antes y Despues
_,_,_,H_abr_108,M_abr_108,_=lector_ciclos(ciclos_SP2_abril_108[3])

_,_,_,H_mar_265,M_mar_265,_=lector_ciclos(ciclos_SP2_marzo_265[0])
_,_,_,H_abr_265,M_abr_265,_=lector_ciclos(ciclos_SP2_abril_265[0])

fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(10,5), constrained_layout=True,sharey=True)
ax1.set_title('108 kHz')
ax1.plot(H_mar_108,M_mar_108,label='Marzo 25')
ax1.plot(H_abr_108,M_abr_108,label='Abril 25')

ax2.set_title('265 kHz')
ax2.plot(H_mar_265,M_mar_265,label='Marzo 25')
ax2.plot(H_abr_265,M_abr_265,label='Abril 25')
    
for a in[ax1,ax2]:
    a.grid()
    a.legend(ncol=1)
    a.set_xlabel('H (A/m)')
ax1.set_ylabel('M (A/m)')
plt.suptitle('SP2 - NF en Al$_2$O$_3$')
plt.savefig('Comparativa_SP2_Marzo_vs_Abril.png', dpi=300)
plt.show()  
#%% SP2 (Marzo/Abril) 108 vs 265 kHz

fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(10,5), constrained_layout=True,sharey=True)

ax1.set_title('SP2 - Marzo')
ax1.plot(H_mar_108,M_mar_108,label='108 kHz')
ax1.plot(H_mar_265,M_mar_265,label='265 kHz')

ax2.set_title('SP2 - Abril')
ax2.plot(H_abr_108,M_abr_108,label='108 kHz')
ax2.plot(H_abr_265,M_abr_265,label='265 kHz')

for a in[ax1,ax2]:
    a.grid()
    a.legend(ncol=1)
    a.set_xlabel('H (A/m)')
ax1.set_ylabel('M (A/m)')
plt.suptitle('SP2 - NF en Al$_2$O$_3$')
plt.savefig('Comparativa_SP2_108_vs_265.png', dpi=300)

plt.show()



# %% Comparo SP2 y SP3 ambos de abril por frecuencias


_,_,_,H_SP2_108,M_SP2_108,_=lector_ciclos(ciclos_SP2_abril_108[3])
_,_,_,H_SP2_265,M_SP2_265,_=lector_ciclos(ciclos_SP2_abril_265[0])

_,_,_,H_SP3_108,M_SP3_108,_=lector_ciclos(ciclos_SP3_abril_108[2])
_,_,_,H_SP3_265,M_SP3_265,_=lector_ciclos(ciclos_SP3_abril_265[1])


fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(10,5), constrained_layout=True,sharey=True)

ax1.set_title('SP2')
ax1.plot(H_SP2_108,M_SP2_108,label='SP2 108')
ax1.plot(H_SP2_265,M_SP2_265,label='SP2 265')

ax2.set_title('SP3')
ax2.plot(H_SP3_108,M_SP3_108,label='SP3 108')
ax2.plot(H_SP3_265,M_SP3_265,label='SP3 265')

for a in[ax1,ax2]:
    a.grid()
    a.legend(ncol=1)
    a.set_xlabel('H (A/m)')
ax1.set_ylabel('M (A/m)')
plt.suptitle('SP2/SP3 - NF/NF@citrato en Al$_2$O$_3$')
#plt.savefig('Comparativa_NF_Citrato_108kHz', dpi=200, facecolor='w')
plt.show()  


#%% new 


ciclos_SP2_abril_108_new=glob(os.path.join('data_250407','108_new','**', '*ciclo_promedio*'),recursive=True)
ciclos_SP2_abril_108_new.sort()

ciclos_SP2_abril_265_new=glob(os.path.join('data_250407','265_new','**', '*ciclo_promedio*'),recursive=True)
ciclos_SP2_abril_265_new.sort()

fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(10,5), constrained_layout=True,sharey=True)

ax1.set_title('108')
for p in ciclos_SP2_abril_108_new:
    _,_,_,H_108,M_108,_=lector_ciclos(p)
    ax1.plot(H_108,M_108,label='108')
    
    
#ax1.plot(H_SP2_265,M_SP2_265,label='SP2 265')

ax2.set_title('265')
for p in ciclos_SP2_abril_265_new:
    _,_,_,H_265,M_265,_=lector_ciclos(p)
    ax2.plot(H_265,M_265,label='265')

for a in[ax1,ax2]:
    a.grid()
    a.legend(ncol=1)
    a.set_xlabel('H (A/m)')
ax1.set_ylabel('M (A/m)')
plt.suptitle('SP2/SP3 - NF/NF@citrato en Al$_2$O$_3$')
plt.savefig('Comparativa_SP2_108_265_new.png', dpi=300)
plt.show()  
#%%
fig, ax1 = plt.subplots(nrows=1,ncols=1,figsize=(6,5), constrained_layout=True,sharey=True)

ax1.set_title('108')
for p in ciclos_SP2_abril_108_new:
    _,_,_,H_108,M_108,_=lector_ciclos(p)
    ax1.plot(H_108,M_108,c='tab:blue',label='108')
    
    
#ax1.plot(H_SP2_265,M_SP2_265,label='SP2 265')

ax1.set_title('265')
for p in ciclos_SP2_abril_265_new:
    _,_,_,H_265,M_265,_=lector_ciclos(p)
    ax1.plot(H_265,M_265,c='tab:orange',label='265')

for a in[ax1]:
    a.grid()
    a.legend(ncol=2)
    a.set_xlabel('H (A/m)')
ax1.set_ylabel('M (A/m)')
plt.suptitle('SP2/SP3 - NF/NF@citrato en Al$_2$O$_3$')
plt.savefig('Comparativa_SP2_108_265_new_bis.png', dpi=300)
plt.show()  
