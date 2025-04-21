#%% Comparador ciclos y resultados de NE@citrato (NE250331C)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glob import glob
import chardet
import re
import os
from uncertainties import ufloat
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
    data = pd.read_table(path, header=17,
                         names=('name', 'Time_m', 'Temperatura',
                                'Remanencia', 'Coercitividad','Campo_max','Mag_max',
                                'frec_fund','mag_fund','dphi_fem',
                                'SAR','tau',
                                'N','xi_M_0'),
                         usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13),
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

#%% Importo NEsferas
# Tabla de resultados
res_NE_081 = glob(os.path.join('081','**','*resultados*'),recursive=True)
res_NE_212 = glob(os.path.join('212','**','*resultados*'),recursive=True)
res_NE_300 = glob(os.path.join('300','**','*resultados*'),recursive=True)
for r in [res_NE_081,res_NE_212,res_NE_300]:
    r.sort()
# ciclos
ciclos_NE_081 = glob(os.path.join('081','**','*ciclo_promedio*'),recursive=True)
ciclos_NE_212 = glob(os.path.join('212','**','*ciclo_promedio*'),recursive=True)
ciclos_NE_300 = glob(os.path.join('300','**','*ciclo_promedio*'),recursive=True)
for c in [ciclos_NE_081,ciclos_NE_212,ciclos_NE_300]:
    c.sort()
#% Coercitivo y remanencia
Coercitivo_081,Remanencia_081 = [],[]
Coercitivo_212,Remanencia_212 = [],[]
Coercitivo_300,Remanencia_300 = [],[]

for r1 in res_NE_081:
  _, _, _,_,Mr, Hc, H_max, M_max, xi_M_0, frec_fund, mag_fund , dphi_fem, SAR, tau, _= lector_resultados(r1)
  Coercitivo_081.append(Hc)
  Remanencia_081.append(Mr)

for r2 in res_NE_212:
  _, _, _,_,Mr, Hc, H_max, M_max, xi_M_0, frec_fund, mag_fund , dphi_fem, SAR, tau, _= lector_resultados(r2)
  Coercitivo_212.append(Hc)
  Remanencia_212.append(Mr)
for r3 in res_NE_300:
  _, _, _,_,Mr, Hc, H_max, M_max, xi_M_0, frec_fund, mag_fund , dphi_fem, SAR, tau, _= lector_resultados(r3)
  Coercitivo_300.append(Hc)
  Remanencia_300.append(Mr)

Hc_081 = np.concatenate(Coercitivo_081)
Hc_212 = np.concatenate(Coercitivo_212)
Hc_300 = np.concatenate(Coercitivo_300)
Hc_081_mean,Hc_081_std = np.mean(Hc_081),np.std(Hc_081)
Hc_212_mean,Hc_212_std = np.mean(Hc_212),np.std(Hc_212)
Hc_300_mean,Hc_300_std = np.mean(Hc_300),np.std(Hc_300)

Mr_081 = np.concatenate(Remanencia_081)
Mr_212 = np.concatenate(Remanencia_212)
Mr_300 = np.concatenate(Remanencia_300)
Mr_081_mean,Mr_081_std = np.mean(Mr_081),np.std(Mr_081)
Mr_212_mean,Mr_212_std = np.mean(Mr_212),np.std(Mr_212)
Mr_300_mean,Mr_300_std = np.mean(Mr_300),np.std(Mr_300)

Hc_081=ufloat(Hc_081_mean,Hc_081_std)
Hc_212=ufloat(Hc_212_mean,Hc_212_std)
Hc_300=ufloat(Hc_300_mean,Hc_300_std)
Mr_081=ufloat(Mr_081_mean,Mr_081_std)
Mr_212=ufloat(Mr_212_mean,Mr_212_std)
Mr_300=ufloat(Mr_300_mean,Mr_300_std)

frecs = ['081','212','300']

# Configuración del gráfico
x = [1/3,2/3,1]  # Posiciones de las barras
width = 0.25  # Ancho de las barras

fig2, (ax1,ax2) = plt.subplots(nrows=2,figsize=(7, 6),constrained_layout=True,sharex=True)

bar1 = ax1.bar(x[0],Hc_081_mean , width, yerr=Hc_081_std, capsize=7, color='tab:blue', label=f'{Hc_081:.1f} kA/m')
bar2 = ax1.bar(x[1],Hc_212_mean , width, yerr=Hc_212_std, capsize=7, color='tab:orange', label=f'{Hc_212:.1f} kA/m')
bar3 = ax1.bar(x[2],Hc_300_mean , width, yerr=Hc_300_std, capsize=7, color='tab:green', label=f'{Hc_300:.1f} kA/m')
bar4 = ax2.bar(x[0],Mr_081_mean , width, yerr=Mr_081_std, capsize=7, color='tab:blue', label=f'{Mr_081:.0f} A/m')
bar5 = ax2.bar(x[1],Mr_212_mean , width, yerr=Mr_212_std, capsize=7, color='tab:orange', label=f'{Mr_212:.0f} A/m')
bar6 = ax2.bar(x[2],Mr_300_mean , width, yerr=Mr_300_std, capsize=7, color='tab:green', label=f'{Mr_300:.0f} A/m')

# ax2.set_ylim(80,120)
for a in[ax1,ax2]:
    a.set_xticks(x)
    a.grid(axis='y', linestyle='--', alpha=0.7)
    a.set_xticklabels(frecs)
    a.legend(ncol=1,bbox_to_anchor=(1,1))
ax2.legend(ncol=1,bbox_to_anchor=(1,1))

ax1.set_ylabel('Hc (kA/m)', fontsize=12)
ax1.set_title('Coercitivo vs Frecuencia',loc='left', fontsize=13)
ax2.set_ylabel('Mr (A/m)', fontsize=12)
ax2.set_title('Remanencia vs Frecuencia',loc='left', fontsize=13)
ax2.set_xlabel('f (kHz)', fontsize=12)
plt.suptitle('NE250331@citrato')
#plt.savefig('comparativa_SAR_100_97.png',dpi=300)
plt.show()
#% Ciclos
fig, ax = plt.subplots(nrows=1,figsize=(7, 6),constrained_layout=True)
for c in ciclos_NE_081:
    _,_,_,H_kAm,M_Am,_ = lector_ciclos(c)
    ax.plot(H_kAm,M_Am,c='tab:blue',label='081')

for c in ciclos_NE_212:
    _,_,_,H_kAm,M_Am,_ = lector_ciclos(c)
    ax.plot(H_kAm,M_Am,c='tab:orange',label='212')

for c in ciclos_NE_300:
    _,_,_,H_kAm,M_Am,_ = lector_ciclos(c)
    ax.plot(H_kAm,M_Am,c='tab:green',label='300')

ax.grid()
ax.set_xlabel('H (kA/m)')
ax.set_ylabel('M (A/m)')
ax.set_title('Comparativa ciclos de histéresis\nNE250331@citrato')
ax.legend(ncol=3)
ax.set_xlim(0,60e3)
ax.set_ylim(0,)
plt.show()
###############################################
#%% Importo NFlores
# Tabla de resultados
res_NF_081 = glob(os.path.join('..','250409_NF250331@citrato_concentrado','081','**','*resultados*'),recursive=True)
res_NF_212 = glob(os.path.join('..','250409_NF250331@citrato_concentrado','212','**','*resultados*'),recursive=True)
res_NF_265 = glob(os.path.join('..','250409_NF250331@citrato_concentrado','265','**','*resultados*'),recursive=True)
for r in [res_NF_081,res_NF_212,res_NF_265]:
    r.sort()
# ciclos
ciclos_NF_081 = glob(os.path.join('..','250409_NF250331@citrato_concentrado','081','**','*ciclo_promedio*'),recursive=True)
ciclos_NF_212 = glob(os.path.join('..','250409_NF250331@citrato_concentrado','212','**','*ciclo_promedio*'),recursive=True)
ciclos_NF_265 = glob(os.path.join('..','250409_NF250331@citrato_concentrado','265','**','*ciclo_promedio*'),recursive=True)
for c in [ciclos_NF_081,ciclos_NF_212,ciclos_NF_265]:
    c.sort()
#% Coercitivo y remanencia
Coercitivo_081,Remanencia_081 = [],[]
Coercitivo_212,Remanencia_212 = [],[]
Coercitivo_265,Remanencia_265 = [],[]

for r1 in res_NF_081:
  _, _, _,_,Mr, Hc, H_max, M_max, xi_M_0, frec_fund, mag_fund , dphi_fem, SAR, tau, _= lector_resultados(r1)
  Coercitivo_081.append(Hc)
  Remanencia_081.append(Mr)

for r2 in res_NF_212:
  _, _, _,_,Mr, Hc, H_max, M_max, xi_M_0, frec_fund, mag_fund , dphi_fem, SAR, tau, _= lector_resultados(r2)
  Coercitivo_212.append(Hc)
  Remanencia_212.append(Mr)
for r3 in res_NF_265:
  _, _, _,_,Mr, Hc, H_max, M_max, xi_M_0, frec_fund, mag_fund , dphi_fem, SAR, tau, _= lector_resultados(r3)
  Coercitivo_265.append(Hc)
  Remanencia_265.append(Mr)

Hc_081 = np.concatenate(Coercitivo_081)
Hc_212 = np.concatenate(Coercitivo_212)
Hc_265 = np.concatenate(Coercitivo_265)
Hc_081_mean,Hc_081_std = np.mean(Hc_081),np.std(Hc_081)
Hc_212_mean,Hc_212_std = np.mean(Hc_212),np.std(Hc_212)
Hc_265_mean,Hc_265_std = np.mean(Hc_265),np.std(Hc_265)

Mr_081 = np.concatenate(Remanencia_081)
Mr_212 = np.concatenate(Remanencia_212)
Mr_265 = np.concatenate(Remanencia_265)
Mr_081_mean,Mr_081_std = np.mean(Mr_081),np.std(Mr_081)
Mr_212_mean,Mr_212_std = np.mean(Mr_212),np.std(Mr_212)
Mr_265_mean,Mr_265_std = np.mean(Mr_265),np.std(Mr_265)

Hc_081=ufloat(Hc_081_mean,Hc_081_std)
Hc_212=ufloat(Hc_212_mean,Hc_212_std)
Hc_265=ufloat(Hc_265_mean,Hc_265_std)
Mr_081=ufloat(Mr_081_mean,Mr_081_std)
Mr_212=ufloat(Mr_212_mean,Mr_212_std)
Mr_265=ufloat(Mr_265_mean,Mr_265_std)

frecs = ['081','212','265']

# Configuración del gráfico
x = [1/3,2/3,1]  # Posiciones de las barras
width = 0.25  # Ancho de las barras

fig2, (ax1,ax2) = plt.subplots(nrows=2,figsize=(7, 6),constrained_layout=True,sharex=True)

bar1 = ax1.bar(x[0],Hc_081_mean , width, yerr=Hc_081_std, capsize=7, color='tab:blue', label=f'{Hc_081:.1f} kA/m')
bar2 = ax1.bar(x[1],Hc_212_mean , width, yerr=Hc_212_std, capsize=7, color='tab:orange', label=f'{Hc_212:.1f} kA/m')
bar3 = ax1.bar(x[2],Hc_265_mean , width, yerr=Hc_265_std, capsize=7, color='tab:green', label=f'{Hc_265:.1f} kA/m')
bar4 = ax2.bar(x[0],Mr_081_mean , width, yerr=Mr_081_std, capsize=7, color='tab:blue', label=f'{Mr_081:.0f} A/m')
bar5 = ax2.bar(x[1],Mr_212_mean , width, yerr=Mr_212_std, capsize=7, color='tab:orange', label=f'{Mr_212:.0f} A/m')
bar6 = ax2.bar(x[2],Mr_265_mean , width, yerr=Mr_265_std, capsize=7, color='tab:green', label=f'{Mr_265:.0f} A/m')

#ax2.set_ylim(80,120)
for a in[ax1,ax2]:
    a.set_xticks(x)
    a.grid(axis='y', linestyle='--', alpha=0.7)
    a.set_xticklabels(frecs)
    a.legend(ncol=1,bbox_to_anchor=(1,1))
ax2.legend(ncol=1,bbox_to_anchor=(1,1))

ax1.set_ylabel('Hc (kA/m)', fontsize=12)
ax1.set_title('Coercitivo vs Frecuencia',loc='left', fontsize=13)
ax2.set_ylabel('Mr (A/m)', fontsize=12)
ax2.set_title('Remanencia vs Frecuencia',loc='left', fontsize=13)
ax2.set_xlabel('f (kHz)', fontsize=12)
plt.suptitle('NE250331@citrato')
#plt.savefig('comparativa_SAR_100_97.png',dpi=300)
plt.show()

#% Ciclos
fig, ax = plt.subplots(nrows=1,figsize=(7, 6),constrained_layout=True)
for c in ciclos_NF_081:
    _,_,_,H_kAm,M_Am,_ = lector_ciclos(c)
    ax.plot(H_kAm,M_Am,c='tab:blue',label='081')

for c in ciclos_NF_212:
    _,_,_,H_kAm,M_Am,_ = lector_ciclos(c)
    ax.plot(H_kAm,M_Am,c='tab:orange',label='212')

for c in ciclos_NF_265:
    _,_,_,H_kAm,M_Am,_ = lector_ciclos(c)
    ax.plot(H_kAm,M_Am,c='tab:green',label='265')

ax.grid()
ax.set_xlabel('H (kA/m)')
ax.set_ylabel('M (A/m)')
ax.set_title('Comparativa ciclos de histéresis\nNE250331@citrato')
ax.legend(ncol=3)
ax.set_xlim(0,60e3)
ax.set_ylim(0,)
plt.show()
# %% Comparo ciclos por frecuencia

fig, (ax1,ax2,ax3) = plt.subplots(nrows=3,ncols=1,figsize=(6, 12),sharex=True,sharey=True,constrained_layout=True)


_,_,_,H_NE_081_kAm,M_NE_081_Am,_ = lector_ciclos(ciclos_NE_081[0])
_,_,_,H_NF_081_kAm,M_NF_081_Am,_ = lector_ciclos(ciclos_NF_081[0])

ax1.plot(H_NE_081_kAm,M_NE_081_Am,label='NE')
ax1.plot(H_NF_081_kAm,M_NF_081_Am,label='NF')

_,_,_,H_NE_212_kAm,M_NE_212_Am,_ = lector_ciclos(ciclos_NE_212[1])
_,_,_,H_NF_212_kAm,M_NF_212_Am,_ = lector_ciclos(ciclos_NF_212[0])

ax2.plot(H_NE_212_kAm,M_NE_212_Am,label='NE')
ax2.plot(H_NF_212_kAm,M_NF_212_Am,label='NF')

_,_,_,H_NE_300_kAm,M_NE_300_Am,_ = lector_ciclos(ciclos_NE_300[0])
_,_,_,H_NF_265_kAm,M_NF_265_Am,_ = lector_ciclos(ciclos_NF_265[0])

ax3.plot(H_NE_300_kAm,M_NE_300_Am,label='NE')
ax3.plot(H_NF_265_kAm,M_NF_265_Am,label='NF')


# for c in ciclos_NF_212:
#     _,_,_,H_kAm,M_Am,_ = lector_ciclos(c)
#     ax.plot(H_kAm,M_Am,c='tab:orange',label='212')

# for c in ciclos_NF_265:
#     _,_,_,H_kAm,M_Am,_ = lector_ciclos(c)
#     ax.plot(H_kAm,M_Am,c='tab:green',label='265')

for a in [ax1,ax2,ax3]:
    a.grid()
    a.set_ylabel('M (A/m)')
    a.legend()
ax3.set_xlabel('H (kA/m)')
ax1.set_title('081kHz',loc='left')
ax2.set_title('212 kHz',loc='left')
ax3.set_title('265/300 kHz',loc='left')


plt.show()


