import os
import argparse
import pandas as pd
import geopandas as gpd
import numpy as np
import f90nml              # Paket, um namelists einfach lesen zu können
import csv
from scipy.io import FortranFile
from numpy.random import default_rng
from matplotlib.path import Path
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point, Polygon
import matplotlib.patches as patches
from datetime import datetime
import math as m
import itertools as iter

# Funktion, die die Fläche von Polygonen berechnet. (Für die Benutzung der Trakjektorienstartdichte nötig.)
# Auf Basis der Shoelace-Formel. Übernommen aus Stackoverflow.
# (https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates)
def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

############ Standardwerte #############
dt = 300                    # alle 5 Minuten
strichdicke = 0.5           # Strichdicke für die KLAM_21 kompatible Ausgabe
lintyp = 0                  # Linienstil für die KLAM_21 kompatible Ausgabe
farbe = [180, 170]          # Farben der Trajektorien [vorwärts,rückwärts] für die KLAM_21 kompatible Ausgabe
farbverlauf = True          # Die Trajektorien werden in der KLAM_21 Ausgabedatei in mehrere Linien mit unterschiedlichen Farben geteilt, alle 30 Minuten
out_format = ['KLAM']       # Ausgabe erfolgt standardmaäßig als pl-Datei
out_file = 'KLATRAout_'     # Name der Ausgabedateien
out_ext = ".out"            # Endung der Ausgabedateien
wind_feld = 'q'             # Standardmäßig wird das gemittelte Windfeld verwendet
traj_dichte = None          # nötig als Standard, um ohne zusätzliche Abfrage für jede Gitterzelle Dinge zu berechnen.
p = 2                       # Powerfaktor für inverse distance weighting
zeitschritt = 1             # falls ein fester Zeitschritt gewünscht ist
ws_th = 0.2                 # wind speed treshold for trajectories, trajectories are only cut below that treshold
th_no = 5                   # number of consecutive time steps with slower wind speeds to drop rest of trajectory
########################################
now=datetime.now()        # Abfrage der Anfangszeit, um die Dauer zu ermitteln
print(now)                # Ausgabe der Anfangszeit, falls man es vergessen hat.
now1=now

# Um das Programm aus der Kommandozeile ausführen zu können, muss irgendwie die Inputdatei übergeben werden.
# Das funktioniert mit diesem Modul.
argParser = argparse.ArgumentParser()
argParser.add_argument("-f", "--file", help="input namelist complete path", required=True)  # Namelist mit allen Informationen, um dieses Programm auszuführen.

input_file = argParser.parse_args().file        # die Inputnamelist 
in_nml = f90nml.read(input_file)                # Einlesen der Inputnamelist

directory = in_nml["input"]["directory"]        # Arbeitsverzeichnis
os.chdir(directory)

# Speichern der in der Namelist angegebenen Variablen unter hier im Programm genutzten Namen
in_file = in_nml["input"]["in_file"]              # Eingangsdatei für KLAM_21

# Umwandeln der gewünschten Start- und Endzeiten in Listen, falls es Einzelwerte sind
start_ein_list = in_nml["trajektorien"]["start"]
if type(start_ein_list) != list:
  start_ein_list = [start_ein_list]
end_ein_list = in_nml["trajektorien"]["end"]
if type(end_ein_list) != list:
  end_ein_list = [end_ein_list]

# Überschreiben der Standardwerte, falls Eintrag vorhanden
if "windfeld" in in_nml["trajektorien"]:
  wind_feld = in_nml["trajektorien"]["windfeld"]
if "traj_start_dichte" in in_nml["trajektorien"]:
  traj_dichte = in_nml["trajektorien"]["traj_start_dichte"] 
if "dt" in in_nml["output"]:
  dt = in_nml["output"]["dt"]                   
if "format" in in_nml["output"]:
  out_format = in_nml["output"]["format"]
  if type(out_format) != list:
    out_format = [out_format]
if "out_file" in in_nml["output"]:
  out_file = in_nml["output"]["out_file"]
if "farbverlauf_mit_zeit" in in_nml["output"]:
  farbverlauf = in_nml["output"]["farbverlauf_mit_zeit"]
if "strichdicke" in in_nml["output"]:
  strichdicke = in_nml["output"]["strichdicke"]
if "farbe" in in_nml["output"]:
  farbe = in_nml["output"]["farbe"]
if "lintyp" in in_nml["output"]:
  lintyp = in_nml["output"]["lintyp"]
if "out_ext" in in_nml["output"]:
  out_ext = in_nml["output"]["out_ext"]
if "zeitschritt" in in_nml["trajektorien"]:
  zeitschritt = in_nml["trajektorien"]["zeitschritt"]
if type(in_nml["trajektorien"]["x_koord"]) != list:
  in_nml["trajektorien"]["x_koord"] = [in_nml["trajektorien"]["x_koord"]]
if type(in_nml["trajektorien"]["y_koord"]) != list:
  in_nml["trajektorien"]["y_koord"] = [in_nml["trajektorien"]["y_koord"]]
if type(in_nml["trajektorien"]["landuse"]) != list:
  in_nml["trajektorien"]["landuse"] = [in_nml["trajektorien"]["landuse"]]
if type(traj_dichte) != list:
  traj_dichte = [traj_dichte]
if "Ueberwaermungsgebiete" in in_nml["input"]:
  warm = in_nml["input"]["Ueberwaermungsgebiete"]
if "Quellgebiete" in in_nml["input"]:
  quelle = in_nml["input"]["Quellgebiete"]
if "windspeed_treshold" in in_nml["Trajektorien"]:
  ws_th = in_nml["Trajektorien"]["windspeed_treshold"]
if "treshold" in in_nml["Trajektorien"]:
  th_no = in_nml["Trajektorien"]["treshold"]


# Einlesen der KLAM_21 Steuerdatei und darin enthaltener Parameter
nml = f90nml.read(directory+in_file)  # KLAM_21 Steuerdatei
dx = int(nml["grid"]["dx"])           # Rastergröße
xlu = nml["grid"]["xrmap"]       # x SW-Ecke Modellgebiet
ylu = nml["grid"]["yrmap"]       # y SW-Ecke Modellgebiet
nx = nml["grid"]["nx"]           # Anzahl Spalten
ny = nml["grid"]["ny"]           # Anzahl Zeilen
iozeit = pd.DataFrame(nml["output"]["iozeit"], columns=["zt"])

# Falls keine Ausgabezeit zu SImulationsbeginn vorliegt, werden künstlich diese Daten erzeugt.
# Da anfangs Windstille vorherrscht und noch keine Kaltluftschicht ausgebildet ist, werden alle Werte 0 gesetzt.
if 0 not in iozeit:
  # Falls keine Dateien für den Zeitpunkt 0 vorhanden sind, werden welche für alle benötigten Parameter geschrieben.
  if "u"+wind_feld+"000000."+nml["output"]["xtension"] not in os.listdir(nml["output"]["resdir"]):
    # write 0-Windfelder
    uq000000 = pd.DataFrame(0, index=np.arange(ny), columns=np.arange(nx))
    with open(nml["output"]["resdir"]+"/u"+wind_feld+f'{iozeit.loc[1,"zt"]:06}'+"."+nml["output"]["xtension"], 'r', encoding="ISO-8859-1") as f:
      head = [next(f) for _ in range(8)]
      f.close()
    fn = open(nml["output"]["resdir"]+"/u"+wind_feld+"000000."+nml["output"]["xtension"], 'w', encoding="ISO-8859-1")
    fn.writelines(head)
    uq000000.to_csv(fn, header=False, index=False, sep=" ")
    fn.close()
    fn = open(nml["output"]["resdir"]+"/v"+wind_feld+"000000."+nml["output"]["xtension"], 'w', encoding="ISO-8859-1")
    fn.writelines(head)
    uq000000.to_csv(fn, header=False, index=False, sep=" ")
    fn.close()
    with open(nml["output"]["resdir"]+"/Hx"+f'{iozeit.loc[1,"zt"]:06}'+"."+nml["output"]["xtension"], 'r', encoding="ISO-8859-1") as f:
      head = [next(f) for _ in range(8)]
      f.close()
    fn = open(nml["output"]["resdir"]+"/Hx"+"000000."+nml["output"]["xtension"], 'w', encoding="ISO-8859-1")
    fn.writelines(head)
    uq000000.to_csv(fn, header=False, index=False, sep=" ")
    fn.close()
    with open(nml["output"]["resdir"]+"/Ex"+f'{iozeit.loc[1,"zt"]:06}'+"."+nml["output"]["xtension"], 'r', encoding="ISO-8859-1") as f:
      head = [next(f) for _ in range(9)]
      f.close()
    fn = open(nml["output"]["resdir"]+"/Ex"+"000000."+nml["output"]["xtension"], 'w', encoding="ISO-8859-1")
    fn.writelines(head)
    np.savetxt(fn, uq000000, fmt="%4d")
    fn.close()
    print("Dateien für Zeitpunkt 0 geschrieben.")
  iozeit.loc[len(iozeit),"zt"] = 0
  iozeit = iozeit.loc[np.argsort(iozeit.zt)].reset_index(drop=True)

# Einlesen der Zeitschrittdatei von KLAM_21, wenn gewünscht und vorhanden.
if in_nml["trajektorien"]["zs"] == 'klam':
  if "dt_file" in nml["perform"]:
    f = FortranFile(nml["perform"]["dt_file"], "r")
    time = []
    while True:
        try:
            time.extend(f.read_reals(dtype=np.float32))
        except:
            break
    f.close()
    times = np.array(time)
    times = np.column_stack((times, times.cumsum()))
  else:
    print("Es sollen die Zeitschritte von KLAM_21 genutzt werden, allerdings liegt keine Zeitschrittdatei vor. \nDas Programm wird jetzt geschlossen.")
    exit()
    
# Generieren von ähnlichen Daten, falls ein fester Zeitschritt gewünscht ist.
else:
  times = np.array([zeitschritt]*round(max(iozeit.zt)/zeitschritt))
  times = np.column_stack((times, times.cumsum()))
  
# Einlesen aller KLAM_21 Ergebnisdateien
ut_list = []; vt_list = []; hx_list = []; ex_list = []
for zeit in iozeit.zt:
  ut_name = nml["output"]["resdir"]+"/u"+wind_feld+f'{int(zeit):06}'+"."+nml["output"]["xtension"]
  vt_name = nml["output"]["resdir"]+"/v"+wind_feld+f'{int(zeit):06}'+"."+nml["output"]["xtension"]
  hxt_name = nml["output"]["resdir"]+"/Hx"+f'{int(zeit):06}'+"."+nml["output"]["xtension"]
  ext_name = nml["output"]["resdir"]+"/Ex"+f'{int(zeit):06}'+"."+nml["output"]["xtension"]

  # Einlesen der Windfelddaten
  ut = np.array(pd.read_csv(ut_name, skiprows=8, delim_whitespace=True, header=None, encoding="ISO-8859-1"))
  vt = np.array(pd.read_csv(vt_name, skiprows=8, delim_whitespace=True, header=None, encoding="ISO-8859-1"))
  hxt = np.array(pd.read_csv(hxt_name, skiprows=8, delim_whitespace=True, header=None, encoding="ISO-8859-1"))
  ext = np.array(pd.read_fwf(ext_name, skiprows=9, widths=[5]*nx, header=None, encoding="ISO-8859-1"))

  ut_list.append(ut)
  
  vt_list.append(vt)
  
  hx_list.append(hxt)
  
  ex_list.append(ext)

# Speichern aller Windfelddateien in einem DataFrame, um leicht auf den richtigen Parameter und Zeitpunkt zugreifen zu können
wind_df = pd.DataFrame({"ut":ut_list, "vt":vt_list, "hx":hx_list, "ex":ex_list}, index=iozeit.zt)

# Einlesen der Landnutzungsdatei
Landuse = pd.read_csv(nml["landuse"]["fn_file"], skiprows=6, encoding="ISO-8859-1", delim_whitespace=True, header=None)
if in_nml["trajektorien"]["art"] in ['Landuse', 'Landuse_Bereich', 'Landuse_Poly']:
  if "lu_file" in in_nml["trajektorien"]:
    if in_nml["trajektorien"]["lu_file"].endswith(".geojson"):
      Landuse_test = gpd.read_file(directory+in_nml["trajektorien"]["lu_file"])     # GeoJSON Landuse benötigt, um sinnvoll 
    else: 
      print("Für die Landnutzungsklassen basierte Startpunktberechnung wird eine GeoJSON Datei der Landnutzungsklassen benötigt.  \nDas Programm wird jetzt geschlossen.")
      exit()
  else: 
    print("Für die Landnutzungsklassen basierte Startpunktberechnung wird eine GeoJSON Datei der Landnutzungsklassen benötigt.  \nDas Programm wird jetzt geschlossen.")
    exit()

if "Ueberwaermungsgebiete" in in_nml["input"]:
  umriss = gpd.read_file(warm)
  name_warm = in_nml["input"]["name_waerme"]
if "Quellgebiete" in in_nml["input"]:
  quellgebiet = gpd.read_file(quelle)
  name_quell = in_nml["input"]["name_quell"]

# Loop über die gewünschten Trajektorienberechnungs Start- und Endzeitpunkte
for end_ind in range(0, len(end_ein_list)):
  end_ein = [end_ein_list[end_ind]]
  start_ein = [start_ein_list[end_ind]]
  
  # Ausgabe im Terminal, für welche Start- und Endzeit nun die Berechnung beginnt
  print("Startzeit: "+str(start_ein[0])+", Endzeit: "+str(end_ein[0]))
  
  # Wenn bestimmte Landuseklasse gewünscht:
  # Da hier ggf. über mehrere Landnutzungsklassen geloopt werden muss, befindet sich die ganze Trajektorienberechnung in diesem if.
  if in_nml["trajektorien"]["art"] in ['Landuse', 'Landuse_Bereich', 'Landuse_Poly']:
    lu_start_list = in_nml["trajektorien"]["landuse"]
    
    for li in range(0, len(lu_start_list)):
      lu_start = lu_start_list[li]
      # Generieren des Ausgabedateinamens
      out_file_name = directory+in_nml["output"]["out_dir"]+out_file+str(start_ein[0])+"_"+str(end_ein[0])+"_LU"+str(lu_start)
      print("Landuse = "+str(lu_start))
  
      if traj_dichte != None:
        if in_nml["trajektorien"]["art"] == 'Landuse_Poly': 
          # Bestimmung des gewünschten Polygons.
          x_eingabe = in_nml["trajektorien"]["x_koord"]
          y_eingabe = in_nml["trajektorien"]["y_koord"]
          koord = list(zip(x_eingabe,y_eingabe))
          poly = Path(koord, closed=True)
          
          # Alle im Untersuchungsgebiet liegenden Rasterzellenmittelpunkte werden bestimmt und dann bestimmt, welche davon sich im Polygon befinden.
          alle_punkte =  list(iter.product([(x+0.5)*dx+xlu for x in range(0, nx)], [(y+0.5)*dx+ylu for y in range(0, ny)]))
          punkte_innen = pd.DataFrame([alle_punkte[i] for i in np.where(poly.contains_points(alle_punkte, radius=-0.5))[0]], columns=["x", "y"])

          # x1 und y1 sind die relativen Koordinaten für die Berechnungsstartpunkte der Trajektorien
          x1=[]; y1=[]
          for i in range(0,len(Landuse_test.loc[Landuse_test.DN==lu_start,])):
            lu_poly = Landuse_test.loc[Landuse_test.DN==lu_start,].reset_index().loc[i,"geometry"]
            lu_path = Path(list(lu_poly.exterior.coords), closed=True)
            if poly.intersects_path(lu_path):
              nstart = lu_poly.area/10000*traj_dichte[li]
              n = m.ceil(nstart)
              punkte_innen2 = [punkte_innen.loc[i,] for i in np.where(lu_path.contains_points(punkte_innen, radius=-0.5))[0]]
              if len(punkte_innen2) > 0:
                punkte_lu_poly = [alle_punkte[i] for i in np.where(lu_path.contains_points(alle_punkte, radius=-0.05))[0]]
                if len(punkte_lu_poly) > 0:
                  n = m.ceil(nstart*len(punkte_innen2)/len(punkte_lu_poly))
                rng = default_rng(seed=51)                # seed setzen, um immer gleiche Punkte zufällig zu erhalten (bei gleichem Eingabezeugs)    
                ind = list(rng.choice(len(punkte_innen2), size=n, replace=False))
                x1 = x1+[punkte_innen2[xi][0]-xlu for xi in ind]; y1 = y1+[punkte_innen2[xi][1]-ylu for xi in ind]
          del(alle_punkte)
          ind_del = []
          for el in range(0,len(x1)):
            if Landuse.loc[ny-y1[el]/dx-0.5,x1[el]/dx-0.5] != lu_start:
              ind_del.append(el)
          
          for i in sorted(ind_del, reverse=True):
              del(x1[i], y1[i])

        ##### Landuse in Teilgebiet (durch Rechteck definiert)
        elif in_nml["trajektorien"]["art"] == 'Landuse_Bereich':
          x_eingabe = in_nml["trajektorien"]["x_koord"]           # Eckpunkte eines Rechtecks. (Zwei Endpunkte einer Diagonalen reichen.)
          y_eingabe = in_nml["trajektorien"]["y_koord"]           # Koordinaten ohne xlu/ylu zu subtrahieren.
          y_ein = [y_eingabe[0],y_eingabe[0],y_eingabe[1],y_eingabe[1]]
          koord = list(zip(2*x_eingabe,y_ein))
          poly = Path(koord, closed=True)
  
          alle_punkte =  list(iter.product([(x+0.5)*dx+xlu for x in range(0, nx)], [(y+0.5)*dx+ylu for y in range(0, ny)]))
          punkte_innen = pd.DataFrame([alle_punkte[i] for i in np.where(poly.contains_points(alle_punkte, radius=-0.5))[0]], columns=["x", "y"])

          x1=[]; y1=[]
          for i in range(0,len(Landuse_test.loc[Landuse_test.DN==lu_start,])):
            lu_poly = Landuse_test.loc[Landuse_test.DN==lu_start,].reset_index().loc[i,"geometry"]
            lu_path = Path(list(lu_poly.exterior.coords), closed=True)
            if poly.intersects_path(lu_path):
              nstart = lu_poly.area/10000*traj_dichte[li]
              n = m.ceil(nstart)
              punkte_innen2 = [punkte_innen.loc[i,] for i in np.where(lu_path.contains_points(punkte_innen, radius=-0.5))[0]]
              if len(punkte_innen2) > 0:
                punkte_lu_poly = [alle_punkte[i] for i in np.where(lu_path.contains_points(alle_punkte, radius=-0.05))[0]]
                if len(punkte_lu_poly) > 0:
                  n = m.ceil(nstart*len(punkte_innen2)/len(punkte_lu_poly))
                rng = default_rng(seed=51)                # seed setzen, um immer gleiche Punkte zufällig zu erhalten (bei gleichem Eingabezeugs)    
                ind = list(rng.choice(len(punkte_innen2), size=n, replace=False))
                x1 = x1+[punkte_innen2[xi][0]-xlu for xi in ind]; y1 = y1+[punkte_innen2[xi][1]-ylu for xi in ind]
          del(alle_punkte)
          ind_del = []
          for el in range(0,len(x1)):
            if Landuse.loc[ny-y1[el]/dx-0.5,x1[el]/dx-0.5] != lu_start:
              ind_del.append(el)
          
          for i in sorted(ind_del, reverse=True):
              del(x1[i], y1[i])

        #### Landuse im gesamten Modellgebiet
        elif in_nml["trajektorien"]["art"] == 'Landuse':
  
          alle_punkte =  list(iter.product([(x+0.5)*dx+xlu for x in range(0, nx)], [(y+0.5)*dx+ylu for y in range(0, ny)]))
          x1=[]; y1=[]
          for i in range(0,len(Landuse_test.loc[Landuse_test.DN==lu_start,])):
            lu_poly = Landuse_test.loc[Landuse_test.DN==lu_start,].reset_index().loc[i,"geometry"]
            lu_path = Path(list(lu_poly.exterior.coords), closed=True)
            nstart = lu_poly.area/10000*traj_dichte[li]
            n = m.ceil(nstart)
            punkte_lu_poly = [alle_punkte[i] for i in np.where(lu_path.contains_points(alle_punkte, radius=-0.05))[0]]
            if len(punkte_lu_poly) > 0:
              rng = default_rng(seed=51)                # seed setzen, um immer gleiche Punkte zufällig zu erhalten (bei gleichem Eingabezeugs)    
              ind = list(rng.choice(len(punkte_lu_poly), size=n, replace=False))
              x1 = x1+[punkte_lu_poly[xi][0]-xlu for xi in ind]; y1 = y1+[punkte_lu_poly[xi][1]-ylu for xi in ind]
          del(alle_punkte)
          ind_del = []
          for el in range(0,len(x1)):
            if Landuse.loc[ny-y1[el]/dx-0.5,x1[el]/dx-0.5] != lu_start:
              ind_del.append(el)
          
          for i in sorted(ind_del, reverse=True):
              del(x1[i], y1[i])
    
      ##### Variante, wo alle Rasterzellen mit bestimmten Landuse rausgeschrieben werden   
      else:
        landuse_koords = Landuse[Landuse==lu_start].stack().index.tolist()
        x1 = [(i+0.5)*dx-xlu for i in [x[1] for x in landuse_koords]]
        y1 = [(i+0.5)*dx-ylu for i in [x[0] for x in landuse_koords]]
      
      xy = pd.DataFrame({"x":x1,"y":y1})
      xy.drop_duplicates(inplace=True)
      x1 = xy.x; y1 = xy.y
      Startzeit = [start_ein[0]]*len(x1)
      Endzeit = [end_ein[0]]*len(x1)

      print("Die Startpunktberechnung hat so lange gedauert: ", datetime.now()-now)
      if len(x1) == 0:
        print("Mit diesen Parametern werden keine Trajektorien berechnet.")
      else:
        if len(x1) == 1:
          print("Es wird jetzt "+str(len(x1))+" Trajektorie berechnet.")
        else:
          print("Es werden jetzt "+str(len(x1))+" Trajektorien berechnet.")

        now = datetime.now()
  
        traj_list=list()                    # Liste, in der alle in einem Durchlauf berechneten Trajektorien als Dataframes abgespeichert werden.
        for p_traj in range(0,len(x1)):
          x = x1[p_traj]
          y = y1[p_traj]
          Zeit = Startzeit[p_traj]

          kf = x/dx                                     # x-Index mit Nachkommastelle
          lf = y/dx                                     # y-Index mit Nachkommastelle
          k = int(kf)                                   # x-Index
          l = int(lf)                                   # y-Index
  
          zeitxy_list = [[Zeit,x,y,Landuse.loc[ny-1-l,k]]]
          wind_list = []
  
          t1, t2 = iozeit.loc[(iozeit["zt"]<=Zeit)][-1:].zt.item(),iozeit.loc[(iozeit["zt"]>Zeit)][:1].zt.item()
  
          # Für Vorwärtstrajektorien
          if Startzeit[p_traj] < Endzeit[p_traj]:
            while Zeit < Endzeit[p_traj]:                     # berechnet so lange Koordinaten, bis die gewünschte Endzeit überschritten wurde.
              mgv = 0                                         # Indikator, ob das Modellgebiet verlassen wurde
  
              if Zeit > t2:                                   # falls die Windfeldausgabedateien nicht mehr am besten sind für den Zeitpunkt
                t1, t2 = iozeit.loc[(iozeit["zt"]<=Zeit)][-1:].zt.item(),iozeit.loc[(iozeit["zt"]>Zeit)][:1].zt.item()
  
              kf = x/dx                                     # x-Index mit Nachkommastelle
              lf = y/dx                                     # y-Index mit Nachkommastelle
              k = int(kf)                                   # x-Index
              l = int(lf)                                   # y-Index
              xdec = kf-k                                   # x Nachkommastelle (für Quadranten)
              ydec = lf-l                                   # y Nachkommastelle (für Quadranten)
  
              # Abfrage des Quadranten der Gitterzelle, in der der aktuelle Punkt liegt.
              # Zuweisung der Indizes der Gitterzellen für die räumliche Interpolation.
              if xdec >= 0.5:
                if ydec >= 0.5: indices = pd.DataFrame({"x":[k, k, k+1, k+1], "y":[l, l+1, l+1, l]}) # Q1
                else: indices = pd.DataFrame({"x":[k, k, k+1, k+1], "y":[l, l-1, l-1, l]}) # Q4
              else:
                if ydec >= 0.5: indices = pd.DataFrame({"x":[k, k, k-1, k-1], "y":[l, l+1, l+1, l]}) # Q2
                else: indices = pd.DataFrame({"x":[k, k, k-1, k-1], "y":[l, l-1, l-1, l]}) # Q3
  
              # Abfrage, ob räumlich interpoliert werden soll, und Ausgabe der (interpolierten) Windfelddaten als v1, v2, u1 und u2.
              # 1 ist jeweils der frühere Ausgabezeitpunkt, 2 der spätere Ausgabezeitpunkt.
              if x < 0 or y < 0  or x > nml["grid"]["nx"]*dx or y > nml["grid"]["ny"]*dx : # Punkt außerhalb Modellgebiet
                #print("Hier kommen wir nicht mehr weiter, der Punkt liegt außerhalb des Modellgebiets.")
                mgv = 1
                break
              elif min(indices.x) < 0 or min(indices.y) < 0 or max(indices.x) > nx or max(indices.y) > ny:
                # Punkt liegt in Rasterfeld am Rand, für räumliche Interpolation wäre außerhalb des Modellgebiets liegendes Rasterfeld nötig
                # ->  es wird nicht räumlich interpoliert
                #print("Es wurde nicht räumlich interpoliert, da der Punkt am Rand des Modellgebiets liegt.")
                v1 = wind_df.loc[t1,"vt"][ny-1-l,k]
                u1 = wind_df.loc[t1,"ut"][ny-1-l,k]
                v2 = wind_df.loc[t2,"vt"][ny-1-l,k]
                u2 = wind_df.loc[t2,"ut"][ny-1-l,k]
                hx1 = wind_df.loc[t1,"hx"][ny-1-l,k]
                hx2 = wind_df.loc[t2,"hx"][ny-1-l,k]
                ex1 = wind_df.loc[t1,"ex"][ny-1-l,k]
                ex2 = wind_df.loc[t2,"ex"][ny-1-l,k]
              elif (x == (k+0.5)*dx) & (y == (l+0.5)*dx):
                #print("Es wurde nicht räumlich interpoliert, da der Punkt genau auf dem Mittelpunkt einer Rasterzelle liegt.")
                v1 = wind_df.loc[t1,"vt"][ny-1-l,k]
                u1 = wind_df.loc[t1,"ut"][ny-1-l,k]
                v2 = wind_df.loc[t2,"vt"][ny-1-l,k]
                u2 = wind_df.loc[t2,"ut"][ny-1-l,k]
                hx1 = wind_df.loc[t1,"hx"][ny-1-l,k]
                hx2 = wind_df.loc[t2,"hx"][ny-1-l,k]
                ex1 = wind_df.loc[t1,"ex"][ny-1-l,k]
                ex2 = wind_df.loc[t2,"ex"][ny-1-l,k]
              else:
                # räumliche Interpolation von 4 Rasterfeldern (welche vier wird durch Quadranten in if-Abfrage (Zeile 256 ff.) bestimmt)
                indices["x_koord"] = (indices.x+0.5)*dx
                indices["y_koord"] = (indices.y+0.5)*dx
                indices["Entfernung"] = ((x-indices.x_koord)**2+(y-indices.y_koord)**2)**0.5
                indices["inverse_distance"] = indices.Entfernung**-p
                indices["gewicht"] = indices.inverse_distance/sum(indices.inverse_distance)
                # räumliche Interpolation
                v1 = 0
                u1 = 0
                v2 = 0
                u2 = 0
                hx1 = 0; hx2 = 0; ex1 = 0; ex2 = 0
                for i in range(0,len(indices)):
                  v1 = v1 + wind_df.loc[t1,"vt"][ny-1-indices.y[i],indices.x[i]] * indices.gewicht[i]
                  u1 = u1 + wind_df.loc[t1,"ut"][ny-1-indices.y[i],indices.x[i]] * indices.gewicht[i]
                  hx1 = hx1 + wind_df.loc[t1,"hx"][ny-1-indices.y[i],indices.x[i]] * indices.gewicht[i]
                  ex1 = ex1 + wind_df.loc[t1,"ex"][ny-1-indices.y[i],indices.x[i]] * indices.gewicht[i]
                  v2 = v2 + wind_df.loc[t2,"vt"][ny-1-indices.y[i],indices.x[i]] * indices.gewicht[i]
                  u2 = u2 + wind_df.loc[t2,"ut"][ny-1-indices.y[i],indices.x[i]] * indices.gewicht[i]
                  hx2 = hx2 + wind_df.loc[t2,"hx"][ny-1-indices.y[i],indices.x[i]] * indices.gewicht[i]
                  ex2 = ex2 + wind_df.loc[t2,"ex"][ny-1-indices.y[i],indices.x[i]] * indices.gewicht[i]
  
              # Heraussuchen und Mitteln der vier Zeitschritte, die um Zeit herum genutzt wurden.
              Zeitschritt = times[np.abs(times[:,1]-Zeit).argsort()[:2],0].min()
  
              g1 = (t2-Zeit)/(t2-t1)                                    # Gewicht für Windfeld vom Zeitpunkt 1
              g2 = (Zeit-t1)/(t2-t1)                                    # Gewicht für Zeitpunkt 2
  
              v = v1*g1+v2*g2                                           # Berechnung der räumlich und zeitlich interpolierten
              u = u1*g1+u2*g2                                           # Windvektoren für die aktuelle Position.
              hx = hx1*g1+hx2*g2
              ex = ex1*g1+ex2*g2
  
              lu = Landuse.loc[ny-1-l,k]
              x = x + Zeitschritt*u/100                                 # Berechnung der neuen x-Position (/100, da die Windfelddaten in cm/s ausgegeben)
              y = y + Zeitschritt*v/100                                 # Berechnung der neuen y-Position
              Zeit = Zeit + Zeitschritt
              wind_list.append([u, v, Zeitschritt, g1, g2, t1, t2, ex, hx])
              zeitxy_list.append([Zeit, x, y, lu])
  
            wind_list.append([np.nan]*9)
            trajektorie = pd.DataFrame(np.column_stack((zeitxy_list, wind_list)), columns=["Zeit", "x", "y", "LU", "u", "v", "Zeitschritt", "g1", "g2", "t1", "t2", "ex", "hx"])
            trajektorie["dt"] = round(trajektorie["Zeit"]-Startzeit[p_traj])
            trajektorie["ws"] = ((trajektorie["u"]/100)**2+(trajektorie["v"]/100)**2)**0.5
            if mgv == 1:
              trajektorie.fillna(-55.55, inplace=True)
            traj_list.append(trajektorie)                               # Wenn alle Punkte einer Trajektorie berechnet sind, wird das gesamte Dataframe dieser
                                                                        # Trajektorie in die Liste angehängt.
  
          # Rückwärtstrajektorienberechnung
          elif Startzeit[p_traj] > Endzeit[p_traj]:
            while Zeit > Endzeit[p_traj]:                               # berechnet so lange Koordinaten, bis die gewünschte Endzeit unterschritten wurde.
              mgv = 0
  
              if Zeit < t1:                                             # falls die Windfeldausgabedateien nicht mehr am besten sind für den Zeitpunkt
                t1, t2 = iozeit.loc[(iozeit["zt"]<=Zeit)][-1:].zt.item(),iozeit.loc[(iozeit["zt"]>Zeit)][:1].zt.item()
  
              kf = x/dx   # x-Index mit Nachkommastelle
              lf = y/dx   # y-Index mit Nachkommastelle
              k = int(kf) # x-Index
              l = int(lf) # y-Index
              xdec = kf-k # x Nachkommastelle (für Quadranten)
              ydec = lf-l # y Nachkommastelle (für Quadranten)
  
              # Abfrage des Quadranten der Gitterzelle, in der der aktuelle Punkt liegt.
              # Zuweisung der Indizes der Gitterzellen für die räumliche Interpolation.
              if xdec >= 0.5:
                if ydec >= 0.5: indices = pd.DataFrame({"x":[k, k, k+1, k+1], "y":[l, l+1, l+1, l]}) # Q1
                else: indices = pd.DataFrame({"x":[k, k, k+1, k+1], "y":[l, l-1, l-1, l]}) # Q4
              else:
                if ydec >= 0.5: indices = pd.DataFrame({"x":[k, k, k-1, k-1], "y":[l, l+1, l+1, l]}) # Q2
                else: indices = pd.DataFrame({"x":[k, k, k-1, k-1], "y":[l, l-1, l-1, l]}) # Q3
  
              # Abfrage, ob räumlich interpoliert werden soll, und Ausgabe der (interpolierten) Windfelddaten als v1, v2, u1 und u2.
              # 1 ist jeweils der frühere Ausgabezeitpunkt, 2 der spätere Ausgabezeitpunkt.
              if x < 0 or y < 0  or x > nml["grid"]["nx"]*dx or y > nml["grid"]["ny"]*dx : # Punkt außerhalb Modellgebiet
                #print("Hier kommen wir nicht mehr weiter, der Punkt liegt außerhalb des Modellgebiets.")
                mgv = 1
                break
              elif min(indices.x) < 0 or min(indices.y) < 0 or max(indices.x) > nx or max(indices.y) > ny:
                # Punkt liegt in Rasterfeld am Rand, für räumliche Interpolation wäre außerhalb des Modellgebiets liegendes Rasterfeld nötig
                # ->  es wird nicht räumlich interpoliert
                v1 = wind_df.loc[t1,"vt"][ny-1-l,k]
                u1 = wind_df.loc[t1,"ut"][ny-1-l,k]
                v2 = wind_df.loc[t2,"vt"][ny-1-l,k]
                u2 = wind_df.loc[t2,"ut"][ny-1-l,k]
                hx1 = wind_df.loc[t1,"hx"][ny-1-l,k]
                hx2 = wind_df.loc[t2,"hx"][ny-1-l,k]
                ex1 = wind_df.loc[t1,"ex"][ny-1-l,k]
                ex2 = wind_df.loc[t2,"ex"][ny-1-l,k]
              elif (x == (k+0.5)*dx) & (y == (l+0.5)*dx):
                # Es wurde nicht räumlich interpoliert, da der Punkt genau auf dem Mittelpunkt einer Rasterzelle liegt.
                v1 = wind_df.loc[t1,"vt"][ny-1-l,k]
                u1 = wind_df.loc[t1,"ut"][ny-1-l,k]
                v2 = wind_df.loc[t2,"vt"][ny-1-l,k]
                u2 = wind_df.loc[t2,"ut"][ny-1-l,k]
                hx1 = wind_df.loc[t1,"hx"][ny-1-l,k]
                hx2 = wind_df.loc[t2,"hx"][ny-1-l,k]
                ex1 = wind_df.loc[t1,"ex"][ny-1-l,k]
                ex2 = wind_df.loc[t2,"ex"][ny-1-l,k]
              else:
                # räumliche Interpolation von 4 Rasterfeldern (welche vier wird durch Quadranten in if-Abfrage (Zeile 354 ff.) bestimmt)
                indices["x_koord"] = (indices.x+0.5)*dx
                indices["y_koord"] = (indices.y+0.5)*dx
                indices["Entfernung"] = ((x-indices.x_koord)**2+(y-indices.y_koord)**2)**0.5
                indices["inverse_distance"] = indices.Entfernung**-p
                indices["gewicht"] = indices.inverse_distance/sum(indices.inverse_distance)
                # räumliche Interpolation
                v1 = 0
                u1 = 0
                v2 = 0
                u2 = 0
                hx1 = 0; hx2 = 0; ex1 = 0; ex2 = 0
                for i in range(0,len(indices)):
                  v1 = v1 + wind_df.loc[t1,"vt"][ny-1-indices.y[i],indices.x[i]] * indices.gewicht[i]
                  u1 = u1 + wind_df.loc[t1,"ut"][ny-1-indices.y[i],indices.x[i]] * indices.gewicht[i]
                  hx1 = hx1 + wind_df.loc[t1,"hx"][ny-1-indices.y[i],indices.x[i]] * indices.gewicht[i]
                  ex1 = ex1 + wind_df.loc[t1,"ex"][ny-1-indices.y[i],indices.x[i]] * indices.gewicht[i]
                  v2 = v2 + wind_df.loc[t2,"vt"][ny-1-indices.y[i],indices.x[i]] * indices.gewicht[i]
                  u2 = u2 + wind_df.loc[t2,"ut"][ny-1-indices.y[i],indices.x[i]] * indices.gewicht[i]
                  hx2 = hx2 + wind_df.loc[t2,"hx"][ny-1-indices.y[i],indices.x[i]] * indices.gewicht[i]
                  ex2 = ex2 + wind_df.loc[t2,"ex"][ny-1-indices.y[i],indices.x[i]] * indices.gewicht[i]
              # Heraussuchen und Mitteln der vier Zeitschritte, die um Zeit herum genutzt wurden.
              Zeitschritt = times[np.abs(times[:,1]-Zeit).argsort()[:2],0].min()
  
              g1 = (t2-Zeit)/(t2-t1)                                    # Gewicht für Windfeld vom Zeitpunkt 1
              g2 = (Zeit-t1)/(t2-t1)                                    # Gewicht für Windfeld vom Zeitpunkt 2
  
              v = v1*g1+v2*g2                                           # Berechnung der räumlich und zeitlich interpolierten
              u = u1*g1+u2*g2                                           # Windvektoren für die aktuelle Position.
              hx = hx1*g1+hx2*g2
              ex = ex1*g1+ex2*g2
  
              lu = Landuse.loc[ny-1-l,k]
              x = x - Zeitschritt*u/100                                 # Berechnung der neuen x-Position (/100, da die Windfelddaten in cm/s ausgegeben)
              y = y - Zeitschritt*v/100                                 # Berechnung der neuen y-Position
              Zeit = Zeit - Zeitschritt
              wind_list.append([u, v, Zeitschritt, g1, g2, t1, t2, ex, hx])
              zeitxy_list.append([Zeit, x, y, lu])
  
            wind_list.append([np.nan]*9)
            trajektorie = pd.DataFrame(np.column_stack((zeitxy_list, wind_list)), columns=["Zeit", "x", "y", "LU", "u", "v", "Zeitschritt", "g1", "g2", "t1", "t2", "ex", "hx"])
            trajektorie["dt"] = round(Startzeit[p_traj]-trajektorie["Zeit"])
            trajektorie["ws"] = ((trajektorie["u"]/100)**2+(trajektorie["v"]/100)**2)**0.5
            if mgv == 1:
              trajektorie.fillna(-55.55, inplace=True)
            traj_list.append(trajektorie)                               # Wenn alle Punkte einer Trajektorie berechnet sind, wird das gesamte Dataframe dieser
                                                                        # Trajektorie in die Liste angehängt.
  
        print("Die Trajektorienberechnung hat so lange gedauert: ", datetime.now()-now)
        now = datetime.now()
        #################################################################################
        # Speichern der Trajektorien

        if "KLAM" in out_format:
          for i in range(0, len(traj_list)):
            trajek = traj_list[i].copy()
            trajek["lahm"] = trajek.ws < ws_th
            trajek["summe"] = trajek.dt #["lahm"].cumsum()
            trajek["summe"] = trajek["summe"].sub(trajek["summe"].mask(trajek.lahm).ffill().fillna(0)).astype(int)
            trajek["lahmer"] = trajek.ws.diff() < 0
            trajek["summe1"] = trajek.summe
            trajek["summe1"] = trajek["summe1"].sub(trajek["summe1"].mask(trajek.lahmer).ffill().fillna(0)).astype(int)
            if max(trajek.summe1) >= th_no:
              ind_drop = trajek.loc[trajek.summe1>=th_no,"summe1"].index[0]
              trajekt = trajek.loc[:ind_drop,:"ws"]
              trajekt.loc[ind_drop,["u","v","ws","ex","hx","Zeitschritt"]] = [-1111,-1111,-11.11,-111.1,-111.1,-11.11]
            else: 
              trajekt = trajek.loc[:,:"ws"]
    
            test = pd.DataFrame(trajekt.loc[trajekt.index==0])
            if Startzeit[i] < Endzeit[i]:
              for dti in range(dt+int(test.loc[0,"Zeit"]),int(test.loc[0,"Zeit"])+int(max(trajekt.dt)),dt):
                test = pd.concat([test, trajekt.loc[np.argsort(np.argsort(round(abs(trajekt['Zeit']-dti))))==0]])
            else:
              for dti in range(int(test.loc[0,"Zeit"])-dt,int(test.loc[0,"Zeit"])-int(max(trajekt.dt)),-dt):
                test = pd.concat([test, trajekt.loc[np.argsort(np.argsort(round(abs(trajekt['Zeit']-dti))))==0]])
            test = pd.concat([test,  trajekt.loc[trajekt.index==max(trajekt.index)]], ignore_index=True)
            test = test.drop_duplicates()
            if "Ueberwaermungsgebiete" in in_nml["input"]:
              for q in range(len(test)):
                geometry = [Point(test.loc[q,"x"]+xlu, test.loc[q,"y"]+ylu)]
                stbz = gpd.GeoDataFrame(geometry=geometry, crs=in_nml["input"]["koord_system"])
                bezirk = gpd.sjoin(umriss, stbz, "inner", "contains")
                if len(bezirk[name_warm].values) > 0:        
                    test.loc[q,"Überwärmungsgebiet"] = bezirk[name_warm].values[0]
                else:
                    test.loc[q,"Überwärmungsgebiet"] = "ob"
            else: 
              test["Überwärmungsgebiet"] = "nc"
            if "Quellgebiete" in in_nml["input"]:
              geometry = [Point(test.loc[test.Zeit==min(test.Zeit),"x"]+xlu, test.loc[test.Zeit==min(test.Zeit),"y"]+ylu)]
              quell = gpd.GeoDataFrame(geometry=geometry, crs=in_nml["input"]["koord_system"])
              gebiet = gpd.sjoin(quellgebiet, quell, "right", "contains")
              if len(gebiet[name_quell].values) > 0:
                test["Quellgebiet"] = gebiet[name_quell].values[0]
              else:
                test["Quellgebiet"] = "ob"
            else:
              test["Quellgebiet"] = "nc"
            if in_nml["output"]["Farbverlauf_mit_Zeit"] == True:
              zeiten = pd.DataFrame({"zeit":[1800*n for n in range(1,int(iozeit.zt[len(iozeit)-1]/1800+1))], 
              "vw_farbe":list(range(farbe[0],farbe[0]+10))+list(range(farbe[0],farbe[0]+6)), "rw_farbe":list(range(farbe[1]+9,farbe[1]-1, -1))+list(range(farbe[1]+9,farbe[1]+3,-1))})
              if Startzeit[i] < Endzeit[i]:
                zeiten = zeiten[(zeiten.zeit >= min(test.Zeit)) & (zeiten.zeit <= max(test.Zeit)+1800)]   # + 1800, um auch Zeiten zwischen der Endzeit und der nächsten halben Stunde zu berücksichtigen
                zeiten.reset_index(inplace=True, drop=True)
                for n in range(0, len(zeiten.zeit)):
                  if n == 0:
                    test1 = test[test.Zeit < zeiten.zeit[n]]
                    test1 = pd.concat([test1,test.loc[np.argsort(np.argsort(round(abs(test['Zeit']-zeiten.zeit[n]))))==0]])
                  elif n == len(zeiten.zeit)-1:
                    test1 = test.loc[np.argsort(np.argsort(round(abs(test['Zeit']-zeiten.zeit[n-1]))))==0]
                    test1 = pd.concat([test1,test[test.Zeit > zeiten.zeit[n-1]]])
                  else:
                    test1 = test.loc[np.argsort(np.argsort(round(abs(test['Zeit']-zeiten.zeit[n-1]))))==0]
                    test1 = pd.concat([test1,test[(test.Zeit > zeiten.zeit[n-1]) & (test.Zeit < zeiten.zeit[n])]])
                    test1 = pd.concat([test1,test.loc[np.argsort(np.argsort(round(abs(test['Zeit']-zeiten.zeit[n]))))==0]])
                  test1.drop_duplicates(inplace=True)
                  test1["*lintyp"] = lintyp
                  test1["s"] = strichdicke
                  test1["x"] = test1.x+xlu
                  test1["y"] = test1.y+ylu
                  test1["Nr"] = i+1
                  test1["u"] = test1.u/100
                  test1["v"] = test1.v/100
    #              test1["ws"] = (test1.u**2+test1.v**2)**0.5
                  test1["ex"] = test1.ex/10
                  test1["hx"] = test1.hx/10
                  test1 = test1.round({"x":2,"y":2, "u":2, "v":2, "ws":2, "Zeitschritt":2, "Zeit":2, "hx":2, "ex":2})
                  test1["LU"] = test1.LU.astype(int)
                  test1.loc[:,["Quellgebiet","Überwärmungsgebiet"]] = test1.loc[:,["Quellgebiet","Überwärmungsgebiet"]].fillna("ob")
                  test1.fillna(-99.99,inplace=True)
                  if Startzeit[i] < Endzeit[i]:
                    test1["icolor"] = zeiten.vw_farbe[n]
                  else:
                    test1["icolor"] = zeiten.rw_farbe[n]
                  if (i == 0) & (n == 0):
                    f = open(out_file_name+out_ext, 'w')
                    writer = csv.writer(f)
                    writer.writerow(["*"+nml["output"]["commres"]])
                    writer.writerow(["*Koordinatensystem: "+in_nml["input"]["koord_system"]])
                    f.writelines(["*u und v in m/s"+"\n", "*ex in kJ/m²"+"\n", "*hx in m\n"])
                    test1.loc[:,["*lintyp","icolor","x","y","s","Nr","Zeit","u","v","ws","ex","hx", "Zeitschritt", "LU","Überwärmungsgebiet","Quellgebiet"]].to_csv(f,header=True, index=False, sep=" ")
                    f.close()
                  elif (i == len(traj_list)-1) & (n == len(zeiten.zeit)-1):
                    f = open(out_file_name+out_ext, 'a')
                    writer = csv.writer(f)
                    writer.writerow([99])
                    test1.loc[:,["*lintyp","icolor","x","y","s","Nr","Zeit","u","v","ws","ex","hx", "Zeitschritt", "LU","Überwärmungsgebiet","Quellgebiet"]].to_csv(f, header=False, sep=" ", index=False)
                    writer.writerow([99])
                    writer.writerow([999])
                    f.close()
                  else:
                    f = open(out_file_name+out_ext, 'a')
                    writer = csv.writer(f)
                    writer.writerow([99])
                    test1.loc[:,["*lintyp","icolor","x","y","s","Nr","Zeit","u","v","ws","ex","hx", "Zeitschritt", "LU","Überwärmungsgebiet","Quellgebiet"]].to_csv(f, header=False, sep=" ", index=False)
                    f.close()
              else:
                zeiten = zeiten[(zeiten.zeit <= max(test.Zeit)+1800) & (zeiten.zeit >= min(test.Zeit))]   # + 1800, um auch Zeiten zwischen der Startzeit und der nächsten halben Stunde zu berücksichtigen
                zeiten.reset_index(inplace=True, drop=True)
                for n in range(len(zeiten.zeit)-1,-1,-1):
                  if n == 0:
                    test1 = test.loc[np.argsort(np.argsort(round(abs(test['Zeit']-zeiten.zeit[n]))))==0]
                    test1 = pd.concat([test1,test[test.Zeit < zeiten.zeit[n]]])
                  elif n == len(zeiten.zeit)-1:
                    test1 = test[test.Zeit > zeiten.zeit[n-1]]
                    test1 = pd.concat([test1,test.loc[np.argsort(np.argsort(round(abs(test['Zeit']-zeiten.zeit[n-1]))))==0]])
                  else:
                    test1 = test.loc[np.argsort(np.argsort(round(abs(test['Zeit']-zeiten.zeit[n]))))==0]
                    test1 = pd.concat([test1,test[(test.Zeit > zeiten.zeit[n-1]) & (test.Zeit < zeiten.zeit[n])]])
                    test1 = pd.concat([test1,test.loc[np.argsort(np.argsort(round(abs(test['Zeit']-zeiten.zeit[n-1]))))==0]])
                  test1.drop_duplicates(inplace=True)
                  test1["*lintyp"] = lintyp
                  test1["s"] = strichdicke
                  test1["x"] = test1.x+xlu
                  test1["y"] = test1.y+ylu
                  test1["Nr"] = i+1
                  test1["u"] = test1.u/100
                  test1["v"] = test1.v/100
    #              test1["ws"] = (test1.u**2+test1.v**2)**0.5
                  test1["ex"] = test1.ex/10
                  test1["hx"] = test1.hx/10
                  test1 = test1.round({"x":2,"y":2, "u":2, "v":2,"ws":2, "Zeitschritt":2, "Zeit":2, "hx":2, "ex":2})
                  test1["LU"] = test1.LU.astype(int)
                  test1.loc[:,["Quellgebiet","Überwärmungsgebiet"]] = test1.loc[:,["Quellgebiet","Überwärmungsgebiet"]].fillna("ob")
                  test1.fillna(-99.99,inplace=True)
                  if Startzeit[i] < Endzeit[i]:
                    test1["icolor"] = zeiten.vw_farbe[n]
                  else:
                    test1["icolor"] = zeiten.rw_farbe[n]
                  if (i == 0) & (n == len(zeiten.zeit)-1):
                    f = open(out_file_name+out_ext, 'w')
                    writer = csv.writer(f)
                    writer.writerow(["*"+nml["output"]["commres"]])
                    writer.writerow(["*Koordinatensystem: "+in_nml["input"]["koord_system"]])
                    f.writelines(["*u und v in m/s"+"\n", "*ex in kJ/m²"+"\n", "*hx in m\n"])
                    test1.loc[:,["*lintyp","icolor","x","y","s","Nr","Zeit","u","v","ws","ex","hx", "Zeitschritt", "LU","Überwärmungsgebiet","Quellgebiet"]].to_csv(f,header=True, index=False, sep=" ")
                    f.close()
                  elif (i == len(traj_list)-1) & (n == 0):
                    f = open(out_file_name+out_ext, 'a')
                    writer = csv.writer(f)
                    writer.writerow([99])
                    test1.loc[:,["*lintyp","icolor","x","y","s","Nr","Zeit","u","v","ws","ex","hx", "Zeitschritt", "LU","Überwärmungsgebiet","Quellgebiet"]].to_csv(f, header=False, sep=" ", index=False)
                    writer.writerow([99])
                    writer.writerow([999])
                    f.close()
                  else:
                    f = open(out_file_name+out_ext, 'a')
                    writer = csv.writer(f)
                    writer.writerow([99])
                    test1.loc[:,["*lintyp","icolor","x","y","s","Nr","Zeit","u","v","ws","ex","hx", "Zeitschritt", "LU","Überwärmungsgebiet","Quellgebiet"]].to_csv(f, header=False, sep=" ", index=False)
                    f.close()
            
            else:
              if test.Zeit.iloc[0] > test.Zeit.iloc[-1]:
                test["icolor"] = farbe[0]                # vorwärts ist blau
              else: 
                test["icolor"] = farbe[1]               # rückwärts ist rot
              test["x"] = test.x + xlu
              test["y"] = test.y + ylu
              test["*lintyp"] = lintyp
              test["s"] = strichdicke
              test["Nr"] = i+1
              test["u"] = test.u/100
              test["v"] = test.v/100
    #          test["ws"] = (test.u**2+test.v**2)**0.5
              test["ex"] = test.ex/10
              test["hx"] = test.hx/10
              test = test.round({"x":2,"y":2, "u":2, "v":2, "ws":2, "Zeitschritt":2, "Zeit":2, "hx":2, "ex":2})
              test["LU"] = test.LU.astype(int)
              test.loc[:,["Quellgebiet","Überwärmungsgebiet"]] = test.loc[:,["Quellgebiet","Überwärmungsgebiet"]].fillna("ob")
              test.fillna(-99.99,inplace=True)
              if i == 0:
                f = open(out_file_name+out_ext, 'w')
                writer = csv.writer(f)
                writer.writerow(["*"+nml["output"]["commres"]])
                writer.writerow(["*Koordinatensystem: "+in_nml["input"]["koord_system"]])
                f.writelines(["*u und v in m/s"+"\n", "*ex in kJ/m²"+"\n", "*hx in m\n"])
                test.loc[:,["*lintyp","icolor","x","y","s","Nr","Zeit","u","v","ws","ex","hx", "Zeitschritt", "LU","Überwärmungsgebiet","Quellgebiet"]].to_csv(f,header=True, index=False, sep=" ")
                f.close()
              elif i == len(traj_list)-1:
                f = open(out_file_name+out_ext, 'a')
                writer = csv.writer(f)
                writer.writerow([99])
                test.loc[:,["*lintyp","icolor","x","y","s","Nr","Zeit","u","v","ws","ex","hx", "Zeitschritt", "LU","Überwärmungsgebiet","Quellgebiet"]].to_csv(f, header=False, sep=" ", index=False)
                writer.writerow([99])
                writer.writerow([999])
                f.close()
              else:
                f = open(out_file_name+out_ext, 'a')
                writer = csv.writer(f)
                writer.writerow([99])
                test.loc[:,["*lintyp","icolor","x","y","s","Nr","Zeit","u","v","ws","ex","hx", "Zeitschritt", "LU","Überwärmungsgebiet","Quellgebiet"]].to_csv(f, header=False, sep=" ", index=False)
                f.close()
          print("Die Trajektorien wurden als pl_datei gespeichert.")
        
        if "geojson" in out_format:
          lines_df = pd.DataFrame({"geometry":[], "Nr":[], "x_start":[], "y_start":[], "t_start":[], "t_end":[], "Überwärmungsgebiet":[], "Quellgebiet":[]}) #
          points_df = pd.DataFrame()
          geom = []
          for i in range(0, len(traj_list)):
            trajek = traj_list[i].copy()
            trajek["lahm"] = trajek.ws < ws_th
            trajek["summe"] = trajek.dt # (2c) #["lahm"].cumsum()
            trajek["summe"] = trajek["summe"].sub(trajek["summe"].mask(trajek.lahm).ffill().fillna(0)).astype(int)
            trajek["lahmer"] = trajek.ws.diff() < 0
            trajek["summe1"] = trajek.summe #.dt (2b)
            trajek["summe1"] = trajek["summe1"].sub(trajek["summe1"].mask(trajek.lahmer).ffill().fillna(0)).astype(int)
            if max(trajek.summe1) >= th_no:
              ind_drop = trajek.loc[trajek.summe1>=th_no,"summe1"].index[0]
              trajekt = trajek.loc[:ind_drop,:"ws"]
              trajekt.loc[ind_drop,["u","v","ws","ex","hx","Zeitschritt"]] = [-1111,-1111,-11.11,-111.1,-111.1,-11.11]
            else: 
              trajekt = trajek.loc[:,:"ws"]
    
            test = pd.DataFrame(trajekt.loc[trajekt.index==0])
            if Startzeit[i] < Endzeit[i]:
              for dti in range(dt+int(test.loc[0,"Zeit"]),int(test.loc[0,"Zeit"])+int(max(trajekt.dt)),dt):
                test = pd.concat([test, trajekt.loc[np.argsort(np.argsort(round(abs(trajekt['Zeit']-dti))))==0]])
            else:
              for dti in range(int(test.loc[0,"Zeit"])-dt,int(test.loc[0,"Zeit"])-int(max(trajekt.dt)),-dt):
                test = pd.concat([test, trajekt.loc[np.argsort(np.argsort(round(abs(trajekt['Zeit']-dti))))==0]])
            test = pd.concat([test,  trajekt.loc[trajekt.index==max(trajekt.index)]], ignore_index=True)
            test = test.drop_duplicates()
            if "Ueberwaermungsgebiete" in in_nml["input"]:
              for q in range(len(test)):
                geometry = [Point(test.loc[q,"x"]+xlu, test.loc[q,"y"]+ylu)]
                stbz = gpd.GeoDataFrame(geometry=geometry, crs=in_nml["input"]["koord_system"])
                bezirk = gpd.sjoin(umriss, stbz, "inner", "contains")
                if len(bezirk[name_warm].values) > 0:        
                    test.loc[q,"Überwärmungsgebiet"] = bezirk[name_warm].values[0]
                else:
                    test.loc[q,"Überwärmungsgebiet"] = "ob"
            else: 
              test["Überwärmungsgebiet"] = "nc"
            if "Quellgebiete" in in_nml["input"]:
              geometry = [Point(test.loc[test.Zeit==min(test.Zeit),"x"]+xlu, test.loc[test.Zeit==min(test.Zeit),"y"]+ylu)]
              quell = gpd.GeoDataFrame(geometry=geometry, crs=in_nml["input"]["koord_system"])
              gebiet = gpd.sjoin(quellgebiet, quell, "right", "contains")
              if len(gebiet[name_quell].values) > 0:
                test["Quellgebiet"] = gebiet[name_quell].values[0]
              else:
                test["Quellgebiet"] = "ob"
            else:
              test["Quellgebiet"] = "nc"
            line = list(zip(round(test.x+xlu,2), round(test.y+ylu,2)))
            geom.append(LineString(line))
            lines_df.loc[i,"Nr":] = [i+1, test.x[0], test.y[0], test.Zeit[0], round(test.Zeit[max(test.index)],2), test.Überwärmungsgebiet[0], test.Quellgebiet[0]] #
            test["Nr"] = i+1
            points_df = pd.concat([points_df,test], ignore_index=True)
          lines_gdf = gpd.GeoDataFrame(lines_df, crs=in_nml["input"]["koord_system"], geometry = geom)
          # with open(directory+in_nml["output"]["out_dir"]+out_file+str(start_ein)+"_"+str(end_ein)+"_LU"+str(lu_start)+"_LS.geojson", "w") as file:
          #   file.write(lines_gdf.to_json())
          lines_gdf.to_file(out_file_name+"_LS.geojson")
          points_df["x"] = points_df.x+xlu
          points_df["y"] = points_df.y+ylu
          points_df["u"] = points_df.u/100
          points_df["v"] = points_df.v/100
    #      points_df["ws"] = (points_df.u**2+points_df.v**2)**0.5
          points_df["ex"] = points_df.ex/10
          points_df["hx"] = points_df.hx/10
          points_df = points_df.round(2)
          points_df["geometry"] = points_df[["x", "y"]].apply(Point, axis=1)
          points_df.loc[:,["Quellgebiet","Überwärmungsgebiet"]] = points_df.loc[:,["Quellgebiet","Überwärmungsgebiet"]].fillna("ob")
          points_df.fillna(-99.99, inplace=True)
          points_gdf = gpd.GeoDataFrame(points_df[["Nr", "x", "y", "Zeit","u", "v","ws", "ex", "hx", "Zeitschritt", "LU", "Überwärmungsgebiet", "Quellgebiet", "geometry"]], crs=in_nml["input"]["koord_system"], geometry="geometry")
          points_gdf.to_file(out_file_name+"_P.geojson")
          # with open(directory+in_nml["output"]["out_dir"]+out_file+str(start_ein)+"_"+str(end_ein)+"_LU"+str(lu_start)+"_P.geojson", "w") as file:
          #   file.write(points_gdf.to_json())
          print("Die Trajektorien wurden im geojson-Format gespeichert.")
  
        print("Dauer des Speicherns = s", datetime.now()-now)
    # 
  else:
    ##### Polygon
    if in_nml["trajektorien"]["art"] == 'Polygon':
      x_eingabe = in_nml["trajektorien"]["x_koord"]
      y_eingabe = in_nml["trajektorien"]["y_koord"]
      koord = list(zip(x_eingabe,y_eingabe))
      poly = Path(koord, closed=True)
      
      alle_punkte =  list(iter.product([(x+0.5)*dx+xlu for x in range(0, nx)], [(y+0.5)*dx+ylu for y in range(0, ny)]))
    
      x1 = []; y1 = []
      if traj_dichte != None:  
        if PolyArea(x_eingabe, y_eingabe)/10000*traj_dichte[0] <= 1:
          x1.append(x_eingabe[round((len(x_eingabe)-1)/2)]-xlu); y1.append(y_eingabe[round((len(y_eingabe)-1)/2)]-ylu)
        else: 
          n = m.ceil(PolyArea(x_eingabe, y_eingabe)/10000*traj_dichte[0])
          punkte_innen = [alle_punkte[i] for i in np.where(poly.contains_points(alle_punkte, radius=-0.5))[0]]
          rng = default_rng(seed=51)                # seed setzen, um immer gleiche Punkte zufällig zu erhalten (bei gleichem Eingabezeugs)    
          ind = list(rng.choice(len(punkte_innen), size=n, replace=False))
          x1 = x1+[punkte_innen[xi][0]-xlu for xi in ind]; y1 = y1+[punkte_innen[xi][1]-ylu for xi in ind]
      else:
        punkte_innen = [alle_punkte[i] for i in np.where(poly.contains_points(alle_punkte, radius=-0.5))[0]]
        x1 = [punkte_innen[xi][0]-xlu for xi in range(0,len(punkte_innen))]
        y1 = [punkte_innen[xi][1]-ylu for xi in range(0,len(punkte_innen))]
      del(alle_punkte)
      
    
    ##### Einzelkoordinaten
    elif in_nml["trajektorien"]["art"] == 'Einzel':
      x1 = [xk-xlu for xk in in_nml["trajektorien"]["x_koord"]]                                         
      y1 = [yk-ylu for yk in in_nml["trajektorien"]["y_koord"]]
      
     
    ##### Rechteck (gefüllt)
    elif in_nml["trajektorien"]["art"] == "Rechteck":
      x_eingabe = in_nml["trajektorien"]["x_koord"]
      y_eingabe = in_nml["trajektorien"]["y_koord"]
      xind_ein = list(range(min([m.floor((x-xlu)/dx) for x in x_eingabe]), max([m.ceil((x-xlu)/dx) for x in x_eingabe])))
      yind_ein =  list(range(min([m.floor((y-ylu)/dx) for y in y_eingabe]), max([m.ceil((y-ylu)/dx) for y in y_eingabe])))
      
      koord = list(iter.product([(x+0.5)*dx+xlu for x in xind_ein], [(y+0.5)*dx+ylu for y in yind_ein]))
      if traj_dichte != None:
        x1 = []; y1 = []
        n = m.ceil(abs((x_eingabe[0]-x_eingabe[1])*(y_eingabe[0]-y_eingabe[1]))/10000*traj_dichte)
        rng = default_rng(seed=51)                # seed setzen, um immer gleiche Punkte zufällig zu erhalten (bei gleichem Eingabezeugs)    
        ind = list(rng.choice(len(koord), size=n, replace=False))
        x1 = x1+[koord[xi][0]-xlu for xi in ind]; y1 = y1+[koord[xi][1]-ylu for xi in ind]
        
      else:
        x1 = [k[0]-xlu for k in koord]; y1 = [k[1]-ylu for k in koord]
  
    #### Rechteck (random gefüllt)
    elif in_nml["trajektorien"]["art"] == "Rechteck_r":
      x_eingabe = in_nml["trajektorien"]["x_koord"]
      y_eingabe = in_nml["trajektorien"]["y_koord"]
      if traj_dichte != None:
        n = m.ceil(abs((x_eingabe[0]-x_eingabe[1])*(y_eingabe[0]-y_eingabe[1]))/10000*traj_dichte)
        rng = default_rng(seed=51)                # seed setzen, um immer gleiche Punkte zufällig zu erhalten (bei gleichem Eingabezeugs)    
        x1 = list(rng.integers(min(x_eingabe)-xlu,max(x_eingabe)-xlu, size=n))
        rng = default_rng(seed=51)                # seed setzen, um immer gleiche Punkte zufällig zu erhalten (bei gleichem Eingabezeugs)    
        y1 = list(rng.integers(min(y_eingabe)-ylu,max(y_eingabe)-ylu, size=n))
        
      else:
        print("Bitte traj_start_dichte in der Steuerdatei angeben oder art='Rechteck' statt 'Rechteck_r'.")
        print("Das Programm wird jetzt geschlossen.")
        exit()
    
    ###### Rechteckrand
    elif in_nml["trajektorien"]["art"] == "Rechteckrand":
      x_eingabe = in_nml["trajektorien"]["x_koord"]
      y_eingabe = in_nml["trajektorien"]["y_koord"]
      x1 = [(x+0.5)*dx for x in xind_ein]
      x1 = x1+[(xind_ein[-1]+0.5)*dx]*len(yind_ein)+[(xind_ein[0]+0.5)*dx]*len(yind_ein)+[(x+0.5)*dx for x in xind_ein]
      y1 = [(yind_ein[0]+0.5)*dx]*len(xind_ein)
      y1 = y1+[(y+0.5)*dx for y in yind_ein]*2+[(yind_ein[-1]+0.5)*dx]*len(xind_ein)
      
        
    else:
      print("Keine gültige Eingabe in Trajektorien art. Programm wird geschlossen.")
      exit()
    
    if (in_nml["trajektorien"]["art"] not in ['Einzel', 'Rechteckrand']) and traj_dichte == None:
      print("Es sollen jetzt "+str(len(x1))+" Trajektorien berechnet werden. Soll wirklich keine Trajektorienstartdichte in der Startdatei angegeben werden?")
      print("Wenn hier eine Trajektorienstartdichte angegeben wird, wird diese in einer neuen Startdatei (.changed) gespeichert.")
      while True:
        abbruch = input("Tippe j, um das Programm unverändert laufen zu lassen, t, um eine Trajektorienstartdichte anzugeben, oder q, um abzubrechen: ")
        if abbruch not in ["j", "t", "q"]:
          print("Das ist keine valide Eingabe. Gib bitte j, t oder n ein: ")
        else: 
          if abbruch == "q":
            print("Programm wird geschlossen.")
            exit()
          elif abbruch == "t":
            while True:
              try:
                traj_dichte = float(input("Pro Hektar sollen so viele Trajektorien starten (bitte Zahl eingeben): "))
                break
              except: 
                print("Bitte eine Zahl aus Ziffern eingeben, wenn es eine Dezimalzahl ist, mit \".\" als Trennzeichen.")
            in_nml["trajektorien"]["traj_start_dichte"] = traj_dichte
            f90nml.write(in_nml, input_file+".changed")
          else:
            print("Ok, dann fahren wir so fort.")
        
      xy = pd.DataFrame({"x":x1,"y":y1})
      xy.drop_duplicates(inplace=True)
      x1 = xy.x; y1 = xy.y
      Startzeit = [start_ein[0]]*len(x1)
      Endzeit = [end_ein[0]]*len(x1)

      print("Die Startpunktberechnung hat so lange gedauert: ", datetime.now()-now)
      if len(x1) == 0:
        print("Mit diesen Parametern werden keine Trajektorien berechnet.")
      else:
        if len(x1) == 1:
          print("Es wird jetzt "+str(len(x1))+" Trajektorie berechnet.")
        else:
          print("Es werden jetzt "+str(len(x1))+" Trajektorien berechnet.")
    
    now = datetime.now()
    
    traj_list=list()                    # Liste, in der alle in einem Durchlauf berechneten Trajektorien als Dataframes abgespeichert werden.
    for p_traj in range(0,len(x1)):
      x = x1[p_traj]
      y = y1[p_traj]
      Zeit = Startzeit[p_traj]
      #Zeitschritt = min(i for i in times.loc[(times['Summe'] >= Startzeit) & (times['Summe'] <= Endzeit),0] if i > 0)
    
      #trajektorie = pd.DataFrame({"x":[x],"y":[y],"Zeit":[Zeit], "u":[np.nan], "v":[np.nan], "dt":[0], "Zeitschritt":[np.nan], "hx":[np.nan], "ex":[np.nan]}, index=[1])
      
      kf = x/dx                                     # x-Index mit Nachkommastelle
      lf = y/dx                                     # y-Index mit Nachkommastelle
      k = int(kf)                                   # x-Index
      l = int(lf)                                   # y-Index
  
      zeitxy_list = [[Zeit,x,y,Landuse.loc[ny-1-l,k]]]
      wind_list = []
      
      t1, t2 = iozeit.loc[(iozeit["zt"]<=Zeit)][-1:].zt.item(),iozeit.loc[(iozeit["zt"]>Zeit)][:1].zt.item()
  
      # Für Vorwärtstrajektorien
      if Startzeit[p_traj] < Endzeit[p_traj]:
        while Zeit < Endzeit[p_traj]:                     # berechnet so lange Koordinaten, bis die gewünschte Endzeit überschritten wurde.
          mgv = 0                                         # Indikator, ob das Modellgebiet verlassen wurde
          
          if Zeit > t2:                                   # falls die Windfeldausgabedateien nicht mehr am besten sind für den Zeitpunkt
            t1, t2 = iozeit.loc[(iozeit["zt"]<=Zeit)][-1:].zt.item(),iozeit.loc[(iozeit["zt"]>Zeit)][:1].zt.item()
    
          kf = x/dx                                     # x-Index mit Nachkommastelle
          lf = y/dx                                     # y-Index mit Nachkommastelle
          k = int(kf)                                   # x-Index
          l = int(lf)                                   # y-Index
          xdec = kf-k                                   # x Nachkommastelle (für Quadranten)
          ydec = lf-l                                   # y Nachkommastelle (für Quadranten)
    
          # Abfrage des Quadranten der Gitterzelle, in der der aktuelle Punkt liegt.
          # Zuweisung der Indizes der Gitterzellen für die räumliche Interpolation.
          if xdec >= 0.5:
            if ydec >= 0.5: indices = pd.DataFrame({"x":[k, k, k+1, k+1], "y":[l, l+1, l+1, l]}) # Q1
            else: indices = pd.DataFrame({"x":[k, k, k+1, k+1], "y":[l, l-1, l-1, l]}) # Q4
          else:
            if ydec >= 0.5: indices = pd.DataFrame({"x":[k, k, k-1, k-1], "y":[l, l+1, l+1, l]}) # Q2
            else: indices = pd.DataFrame({"x":[k, k, k-1, k-1], "y":[l, l-1, l-1, l]}) # Q3
    
          # Abfrage, ob räumlich interpoliert werden soll, und Ausgabe der (interpolierten) Windfelddaten als v1, v2, u1 und u2.
          # 1 ist jeweils der frühere Ausgabezeitpunkt, 2 der spätere Ausgabezeitpunkt.
          if x < 0 or y < 0  or x > nml["grid"]["nx"]*dx or y > nml["grid"]["ny"]*dx : # Punkt außerhalb Modellgebiet
            #print("Hier kommen wir nicht mehr weiter, der Punkt liegt außerhalb des Modellgebiets.")
            mgv = 1
            break
          elif min(indices.x) < 0 or min(indices.y) < 0 or max(indices.x) > nx or max(indices.y) > ny:
            # Punkt liegt in Rasterfeld am Rand, für räumliche Interpolation wäre außerhalb des Modellgebiets liegendes Rasterfeld nötig
            # ->  es wird nicht räumlich interpoliert
            #print("Es wurde nicht räumlich interpoliert, da der Punkt am Rand des Modellgebiets liegt.")
            v1 = wind_df.loc[t1,"vt"][ny-1-l,k]
            u1 = wind_df.loc[t1,"ut"][ny-1-l,k]
            v2 = wind_df.loc[t2,"vt"][ny-1-l,k]
            u2 = wind_df.loc[t2,"ut"][ny-1-l,k]
            hx1 = wind_df.loc[t1,"hx"][ny-1-l,k]
            hx2 = wind_df.loc[t2,"hx"][ny-1-l,k]
            ex1 = wind_df.loc[t1,"ex"][ny-1-l,k]
            ex2 = wind_df.loc[t2,"ex"][ny-1-l,k]
          elif (x == (k+0.5)*dx) & (y == (l+0.5)*dx):
            #print("Es wurde nicht räumlich interpoliert, da der Punkt genau auf dem Mittelpunkt einer Rasterzelle liegt.")
            v1 = wind_df.loc[t1,"vt"][ny-1-l,k]
            u1 = wind_df.loc[t1,"ut"][ny-1-l,k]
            v2 = wind_df.loc[t2,"vt"][ny-1-l,k]
            u2 = wind_df.loc[t2,"ut"][ny-1-l,k]
            hx1 = wind_df.loc[t1,"hx"][ny-1-l,k]
            hx2 = wind_df.loc[t2,"hx"][ny-1-l,k]
            ex1 = wind_df.loc[t1,"ex"][ny-1-l,k]
            ex2 = wind_df.loc[t2,"ex"][ny-1-l,k]
          else:
            # räumliche Interpolation von 4 Rasterfeldern (welche vier wird durch Quadranten in if-Abfrage (Zeile 256 ff.) bestimmt)
            indices["x_koord"] = (indices.x+0.5)*dx
            indices["y_koord"] = (indices.y+0.5)*dx
            indices["Entfernung"] = ((x-indices.x_koord)**2+(y-indices.y_koord)**2)**0.5
            indices["inverse_distance"] = indices.Entfernung**-p
            indices["gewicht"] = indices.inverse_distance/sum(indices.inverse_distance)
            # räumliche Interpolation
            v1 = 0
            u1 = 0
            v2 = 0
            u2 = 0
            hx1 = 0; hx2 = 0; ex1 = 0; ex2 = 0
            for i in range(0,len(indices)):
              v1 = v1 + wind_df.loc[t1,"vt"][ny-1-indices.y[i],indices.x[i]] * indices.gewicht[i]
              u1 = u1 + wind_df.loc[t1,"ut"][ny-1-indices.y[i],indices.x[i]] * indices.gewicht[i]
              hx1 = hx1 + wind_df.loc[t1,"hx"][ny-1-indices.y[i],indices.x[i]] * indices.gewicht[i]
              ex1 = ex1 + wind_df.loc[t1,"ex"][ny-1-indices.y[i],indices.x[i]] * indices.gewicht[i]
              v2 = v2 + wind_df.loc[t2,"vt"][ny-1-indices.y[i],indices.x[i]] * indices.gewicht[i]
              u2 = u2 + wind_df.loc[t2,"ut"][ny-1-indices.y[i],indices.x[i]] * indices.gewicht[i]
              hx2 = hx2 + wind_df.loc[t2,"hx"][ny-1-indices.y[i],indices.x[i]] * indices.gewicht[i]
              ex2 = ex2 + wind_df.loc[t2,"ex"][ny-1-indices.y[i],indices.x[i]] * indices.gewicht[i]
    
          # Heraussuchen und Mitteln der vier Zeitschritte, die um Zeit herum genutzt wurden.
          Zeitschritt = times[np.abs(times[:,1]-Zeit).argsort()[:2],0].min()
    
          g1 = (t2-Zeit)/(t2-t1)                                    # Gewicht für Windfeld vom Zeitpunkt 1
          g2 = (Zeit-t1)/(t2-t1)                                    # Gewicht für Zeitpunkt 2
    
          v = v1*g1+v2*g2                                           # Berechnung der räumlich und zeitlich interpolierten
          u = u1*g1+u2*g2                                           # Windvektoren für die aktuelle Position.
          hx = hx1*g1+hx2*g2
          ex = ex1*g1+ex2*g2
  
          lu = Landuse.loc[ny-1-l,k]
          x = x + Zeitschritt*u/100                                 # Berechnung der neuen x-Position (/100, da die Windfelddaten in cm/s ausgegeben)
          y = y + Zeitschritt*v/100                                 # Berechnung der neuen y-Position
          Zeit = Zeit + Zeitschritt
          wind_list.append([u, v, Zeitschritt, g1, g2, t1, t2, ex, hx]) 
          zeitxy_list.append([Zeit, x, y, lu])
    
        wind_list.append([np.nan]*9)
        trajektorie = pd.DataFrame(np.column_stack((zeitxy_list, wind_list)), columns=["Zeit", "x", "y", "LU", "u", "v", "Zeitschritt", "g1", "g2", "t1", "t2", "ex", "hx"])
        trajektorie["dt"] = round(trajektorie["Zeit"]-Startzeit[p_traj])
        trajektorie["ws"] = ((trajektorie["u"]/100)**2+(trajektorie["v"]/100)**2)**0.5
        if mgv == 1:
          trajektorie.fillna(-55.55, inplace=True)
        traj_list.append(trajektorie)                               # Wenn alle Punkte einer Trajektorie berechnet sind, wird das gesamte Dataframe dieser 
                                                                    # Trajektorie in die Liste angehängt.
                                                                    
      # Rückwärtstrajektorienberechnung                                                              
      elif Startzeit[p_traj] > Endzeit[p_traj]:
        while Zeit > Endzeit[p_traj]:                               # berechnet so lange Koordinaten, bis die gewünschte Endzeit unterschritten wurde.
          mgv = 0
    
          if Zeit < t1:                                             # falls die Windfeldausgabedateien nicht mehr am besten sind für den Zeitpunkt
            t1, t2 = iozeit.loc[(iozeit["zt"]<=Zeit)][-1:].zt.item(),iozeit.loc[(iozeit["zt"]>Zeit)][:1].zt.item()
    
          kf = x/dx   # x-Index mit Nachkommastelle
          lf = y/dx   # y-Index mit Nachkommastelle
          k = int(kf) # x-Index
          l = int(lf) # y-Index
          xdec = kf-k # x Nachkommastelle (für Quadranten)
          ydec = lf-l # y Nachkommastelle (für Quadranten)
    
          # Abfrage des Quadranten der Gitterzelle, in der der aktuelle Punkt liegt.
          # Zuweisung der Indizes der Gitterzellen für die räumliche Interpolation.
          if xdec >= 0.5:
            if ydec >= 0.5: indices = pd.DataFrame({"x":[k, k, k+1, k+1], "y":[l, l+1, l+1, l]}) # Q1
            else: indices = pd.DataFrame({"x":[k, k, k+1, k+1], "y":[l, l-1, l-1, l]}) # Q4
          else:
            if ydec >= 0.5: indices = pd.DataFrame({"x":[k, k, k-1, k-1], "y":[l, l+1, l+1, l]}) # Q2
            else: indices = pd.DataFrame({"x":[k, k, k-1, k-1], "y":[l, l-1, l-1, l]}) # Q3
    
          # Abfrage, ob räumlich interpoliert werden soll, und Ausgabe der (interpolierten) Windfelddaten als v1, v2, u1 und u2.
          # 1 ist jeweils der frühere Ausgabezeitpunkt, 2 der spätere Ausgabezeitpunkt.
          if x < 0 or y < 0  or x > nml["grid"]["nx"]*dx or y > nml["grid"]["ny"]*dx : # Punkt außerhalb Modellgebiet
            #print("Hier kommen wir nicht mehr weiter, der Punkt liegt außerhalb des Modellgebiets.")
            mgv = 1
            break
          elif min(indices.x) < 0 or min(indices.y) < 0 or max(indices.x) > nx or max(indices.y) > ny:
            # Punkt liegt in Rasterfeld am Rand, für räumliche Interpolation wäre außerhalb des Modellgebiets liegendes Rasterfeld nötig
            # ->  es wird nicht räumlich interpoliert
            v1 = wind_df.loc[t1,"vt"][ny-1-l,k]
            u1 = wind_df.loc[t1,"ut"][ny-1-l,k]
            v2 = wind_df.loc[t2,"vt"][ny-1-l,k]
            u2 = wind_df.loc[t2,"ut"][ny-1-l,k]
            hx1 = wind_df.loc[t1,"hx"][ny-1-l,k]
            hx2 = wind_df.loc[t2,"hx"][ny-1-l,k]
            ex1 = wind_df.loc[t1,"ex"][ny-1-l,k]
            ex2 = wind_df.loc[t2,"ex"][ny-1-l,k]
          elif (x == (k+0.5)*dx) & (y == (l+0.5)*dx):
            # Es wurde nicht räumlich interpoliert, da der Punkt genau auf dem Mittelpunkt einer Rasterzelle liegt.
            v1 = wind_df.loc[t1,"vt"][ny-1-l,k]
            u1 = wind_df.loc[t1,"ut"][ny-1-l,k]
            v2 = wind_df.loc[t2,"vt"][ny-1-l,k]
            u2 = wind_df.loc[t2,"ut"][ny-1-l,k]
            hx1 = wind_df.loc[t1,"hx"][ny-1-l,k]
            hx2 = wind_df.loc[t2,"hx"][ny-1-l,k]
            ex1 = wind_df.loc[t1,"ex"][ny-1-l,k]
            ex2 = wind_df.loc[t2,"ex"][ny-1-l,k]
          else:
            # räumliche Interpolation von 4 Rasterfeldern (welche vier wird durch Quadranten in if-Abfrage (Zeile 354 ff.) bestimmt)
            indices["x_koord"] = (indices.x+0.5)*dx
            indices["y_koord"] = (indices.y+0.5)*dx
            indices["Entfernung"] = ((x-indices.x_koord)**2+(y-indices.y_koord)**2)**0.5
            indices["inverse_distance"] = indices.Entfernung**-p
            indices["gewicht"] = indices.inverse_distance/sum(indices.inverse_distance)
            # räumliche Interpolation
            v1 = 0
            u1 = 0
            v2 = 0
            u2 = 0
            hx1 = 0; hx2 = 0; ex1 = 0; ex2 = 0
            for i in range(0,len(indices)):
              v1 = v1 + wind_df.loc[t1,"vt"][ny-1-indices.y[i],indices.x[i]] * indices.gewicht[i]
              u1 = u1 + wind_df.loc[t1,"ut"][ny-1-indices.y[i],indices.x[i]] * indices.gewicht[i]
              hx1 = hx1 + wind_df.loc[t1,"hx"][ny-1-indices.y[i],indices.x[i]] * indices.gewicht[i]
              ex1 = ex1 + wind_df.loc[t1,"ex"][ny-1-indices.y[i],indices.x[i]] * indices.gewicht[i]
              v2 = v2 + wind_df.loc[t2,"vt"][ny-1-indices.y[i],indices.x[i]] * indices.gewicht[i]
              u2 = u2 + wind_df.loc[t2,"ut"][ny-1-indices.y[i],indices.x[i]] * indices.gewicht[i]
              hx2 = hx2 + wind_df.loc[t2,"hx"][ny-1-indices.y[i],indices.x[i]] * indices.gewicht[i]
              ex2 = ex2 + wind_df.loc[t2,"ex"][ny-1-indices.y[i],indices.x[i]] * indices.gewicht[i]
          # Heraussuchen und Mitteln der vier Zeitschritte, die um Zeit herum genutzt wurden.
          Zeitschritt = times[np.abs(times[:,1]-Zeit).argsort()[:2],0].min()
    
          g1 = (t2-Zeit)/(t2-t1)                                    # Gewicht für Windfeld vom Zeitpunkt 1
          g2 = (Zeit-t1)/(t2-t1)                                    # Gewicht für Windfeld vom Zeitpunkt 2
    
          v = v1*g1+v2*g2                                           # Berechnung der räumlich und zeitlich interpolierten
          u = u1*g1+u2*g2                                           # Windvektoren für die aktuelle Position.
          hx = hx1*g1+hx2*g2
          ex = ex1*g1+ex2*g2
          
          lu = Landuse.loc[ny-1-l,k]
          x = x - Zeitschritt*u/100                                 # Berechnung der neuen x-Position (/100, da die Windfelddaten in cm/s ausgegeben)
          y = y - Zeitschritt*v/100                                 # Berechnung der neuen y-Position
          Zeit = Zeit - Zeitschritt
          wind_list.append([u, v, Zeitschritt, g1, g2, t1, t2, ex, hx]) 
          zeitxy_list.append([Zeit, x, y, lu])
    
        wind_list.append([np.nan]*9)
        trajektorie = pd.DataFrame(np.column_stack((zeitxy_list, wind_list)), columns=["Zeit", "x", "y", "LU", "u", "v", "Zeitschritt", "g1", "g2", "t1", "t2", "ex", "hx"])
        trajektorie["dt"] = round(Startzeit[p_traj]-trajektorie["Zeit"])
        trajektorie["ws"] = ((trajektorie["u"]/100)**2+(trajektorie["v"]/100)**2)**0.5
        if mgv == 1:
          trajektorie.fillna(-55.55, inplace=True)
        traj_list.append(trajektorie)                               # Wenn alle Punkte einer Trajektorie berechnet sind, wird das gesamte Dataframe dieser 
                                                                    # Trajektorie in die Liste angehängt.
    
    print("Die Trajektorienberechnung hat so lange gedauert: ", datetime.now()-now)
    now = datetime.now()
    #################################################################################
    # Generieren des Trajektoriendateinamens
    out_file_name = directory+in_nml["output"]["out_dir"]+out_file+str(start_ein[0])+"_"+str(end_ein[0])

    if "KLAM" in out_format:
      for i in range(0, len(traj_list)):
        trajek = traj_list[i].copy()
        trajek["lahm"] = trajek.ws < ws_th
        trajek["summe"] = trajek.dt #["lahm"].cumsum()
        trajek["summe"] = trajek["summe"].sub(trajek["summe"].mask(trajek.lahm).ffill().fillna(0)).astype(int)
        trajek["lahmer"] = trajek.ws.diff() < 0
        trajek["summe1"] = trajek.summe
        trajek["summe1"] = trajek["summe1"].sub(trajek["summe1"].mask(trajek.lahmer).ffill().fillna(0)).astype(int)
        if max(trajek.summe1) >= th_no:
          ind_drop = trajek.loc[trajek.summe1>=th_no,"summe1"].index[0]
          trajekt = trajek.loc[:ind_drop,:"ws"]
          trajekt.loc[ind_drop,["u","v","ws","ex","hx","Zeitschritt"]] = [-1111,-1111,-11.11,-111.1,-111.1,-11.11]
        else: 
          trajekt = trajek.loc[:,:"ws"]

        test = pd.DataFrame(trajekt.loc[trajekt.index==0])
        if Startzeit[i] < Endzeit[i]:
          for dti in range(dt+int(test.loc[0,"Zeit"]),int(test.loc[0,"Zeit"])+int(max(trajekt.dt)),dt):
            test = pd.concat([test, trajekt.loc[np.argsort(np.argsort(round(abs(trajekt['Zeit']-dti))))==0]])
        else:
          for dti in range(int(test.loc[0,"Zeit"])-dt,int(test.loc[0,"Zeit"])-int(max(trajekt.dt)),-dt):
            test = pd.concat([test, trajekt.loc[np.argsort(np.argsort(round(abs(trajekt['Zeit']-dti))))==0]])
        test = pd.concat([test,  trajekt.loc[trajekt.index==max(trajekt.index)]], ignore_index=True)
        test = test.drop_duplicates()
        if "Ueberwaermungsgebiete" in in_nml["input"]:
          for q in range(len(test)):
            geometry = [Point(test.loc[q,"x"]+xlu, test.loc[q,"y"]+ylu)]
            stbz = gpd.GeoDataFrame(geometry=geometry, crs=in_nml["input"]["koord_system"])
            bezirk = gpd.sjoin(umriss, stbz, "inner", "contains")
            if len(bezirk[name_warm].values) > 0:        
                test.loc[q,"Überwärmungsgebiet"] = bezirk[name_warm].values[0]
            else:
                test.loc[q,"Überwärmungsgebiet"] = "ob"
        else: 
          test["Überwärmungsgebiet"] = "nc"
        if "Quellgebiete" in in_nml["input"]:
          geometry = [Point(test.loc[test.Zeit==min(test.Zeit),"x"]+xlu, test.loc[test.Zeit==min(test.Zeit),"y"]+ylu)]
          quell = gpd.GeoDataFrame(geometry=geometry, crs=in_nml["input"]["koord_system"])
          gebiet = gpd.sjoin(quellgebiet, quell, "right", "contains")
          if len(gebiet[name_quell].values) > 0:
            test["Quellgebiet"] = gebiet[name_quell].values[0]
          else:
            test["Quellgebiet"] = "ob"
        else:
          test["Quellgebiet"] = "nc"
        if in_nml["output"]["Farbverlauf_mit_Zeit"] == True:
          zeiten = pd.DataFrame({"zeit":[1800*n for n in range(1,int(iozeit.zt[len(iozeit)-1]/1800+1))], 
          "vw_farbe":list(range(farbe[0],farbe[0]+10))+list(range(farbe[0],farbe[0]+6)), "rw_farbe":list(range(farbe[1]+9,farbe[1]-1, -1))+list(range(farbe[1]+9,farbe[1]+3,-1))})
          if Startzeit[i] < Endzeit[i]:
            zeiten = zeiten[(zeiten.zeit >= min(test.Zeit)) & (zeiten.zeit <= max(test.Zeit)+1800)]   # + 1800, um auch Zeiten zwischen der Endzeit und der nächsten halben Stunde zu berücksichtigen
            zeiten.reset_index(inplace=True, drop=True)
            for n in range(0, len(zeiten.zeit)):
              if n == 0:
                test1 = test[test.Zeit < zeiten.zeit[n]]
                test1 = pd.concat([test1,test.loc[np.argsort(np.argsort(round(abs(test['Zeit']-zeiten.zeit[n]))))==0]])
              elif n == len(zeiten.zeit)-1:
                test1 = test.loc[np.argsort(np.argsort(round(abs(test['Zeit']-zeiten.zeit[n-1]))))==0]
                test1 = pd.concat([test1,test[test.Zeit > zeiten.zeit[n-1]]])
              else:
                test1 = test.loc[np.argsort(np.argsort(round(abs(test['Zeit']-zeiten.zeit[n-1]))))==0]
                test1 = pd.concat([test1,test[(test.Zeit > zeiten.zeit[n-1]) & (test.Zeit < zeiten.zeit[n])]])
                test1 = pd.concat([test1,test.loc[np.argsort(np.argsort(round(abs(test['Zeit']-zeiten.zeit[n]))))==0]])
              test1.drop_duplicates(inplace=True)
              test1["*lintyp"] = lintyp
              test1["s"] = strichdicke
              test1["x"] = test1.x+xlu
              test1["y"] = test1.y+ylu
              test1["Nr"] = i+1
              test1["u"] = test1.u/100
              test1["v"] = test1.v/100
#              test1["ws"] = (test1.u**2+test1.v**2)**0.5
              test1["ex"] = test1.ex/10
              test1["hx"] = test1.hx/10
              test1 = test1.round({"x":2,"y":2, "u":2, "v":2, "ws":2, "Zeitschritt":2, "Zeit":2, "hx":2, "ex":2})
              test1["LU"] = test1.LU.astype(int)
              test1.loc[:,["Quellgebiet","Überwärmungsgebiet"]] = test1.loc[:,["Quellgebiet","Überwärmungsgebiet"]].fillna("ob")
              test1.fillna(-99.99,inplace=True)
              if Startzeit[i] < Endzeit[i]:
                test1["icolor"] = zeiten.vw_farbe[n]
              else:
                test1["icolor"] = zeiten.rw_farbe[n]
              if (i == 0) & (n == 0):
                f = open(out_file_name+out_ext, 'w')
                writer = csv.writer(f)
                writer.writerow(["*"+nml["output"]["commres"]])
                writer.writerow(["*Koordinatensystem: "+in_nml["input"]["koord_system"]])
                f.writelines(["*u und v in m/s"+"\n", "*ex in kJ/m²"+"\n", "*hx in m\n"])
                test1.loc[:,["*lintyp","icolor","x","y","s","Nr","Zeit","u","v","ws","ex","hx", "Zeitschritt", "LU","Überwärmungsgebiet","Quellgebiet"]].to_csv(f,header=True, index=False, sep=" ")
                f.close()
              elif (i == len(traj_list)-1) & (n == len(zeiten.zeit)-1):
                f = open(out_file_name+out_ext, 'a')
                writer = csv.writer(f)
                writer.writerow([99])
                test1.loc[:,["*lintyp","icolor","x","y","s","Nr","Zeit","u","v","ws","ex","hx", "Zeitschritt", "LU","Überwärmungsgebiet","Quellgebiet"]].to_csv(f, header=False, sep=" ", index=False)
                writer.writerow([99])
                writer.writerow([999])
                f.close()
              else:
                f = open(out_file_name+out_ext, 'a')
                writer = csv.writer(f)
                writer.writerow([99])
                test1.loc[:,["*lintyp","icolor","x","y","s","Nr","Zeit","u","v","ws","ex","hx", "Zeitschritt", "LU","Überwärmungsgebiet","Quellgebiet"]].to_csv(f, header=False, sep=" ", index=False)
                f.close()
          else:
            zeiten = zeiten[(zeiten.zeit <= max(test.Zeit)+1800) & (zeiten.zeit >= min(test.Zeit))]   # + 1800, um auch Zeiten zwischen der Startzeit und der nächsten halben Stunde zu berücksichtigen
            zeiten.reset_index(inplace=True, drop=True)
            for n in range(len(zeiten.zeit)-1,-1,-1):
              if n == 0:
                test1 = test.loc[np.argsort(np.argsort(round(abs(test['Zeit']-zeiten.zeit[n]))))==0]
                test1 = pd.concat([test1,test[test.Zeit < zeiten.zeit[n]]])
              elif n == len(zeiten.zeit)-1:
                test1 = test[test.Zeit > zeiten.zeit[n-1]]
                test1 = pd.concat([test1,test.loc[np.argsort(np.argsort(round(abs(test['Zeit']-zeiten.zeit[n-1]))))==0]])
              else:
                test1 = test.loc[np.argsort(np.argsort(round(abs(test['Zeit']-zeiten.zeit[n]))))==0]
                test1 = pd.concat([test1,test[(test.Zeit > zeiten.zeit[n-1]) & (test.Zeit < zeiten.zeit[n])]])
                test1 = pd.concat([test1,test.loc[np.argsort(np.argsort(round(abs(test['Zeit']-zeiten.zeit[n-1]))))==0]])
              test1.drop_duplicates(inplace=True)
              test1["*lintyp"] = lintyp
              test1["s"] = strichdicke
              test1["x"] = test1.x+xlu
              test1["y"] = test1.y+ylu
              test1["Nr"] = i+1
              test1["u"] = test1.u/100
              test1["v"] = test1.v/100
#              test1["ws"] = (test1.u**2+test1.v**2)**0.5
              test1["ex"] = test1.ex/10
              test1["hx"] = test1.hx/10
              test1 = test1.round({"x":2,"y":2, "u":2, "v":2,"ws":2, "Zeitschritt":2, "Zeit":2, "hx":2, "ex":2})
              test1["LU"] = test1.LU.astype(int)
              test1.loc[:,["Quellgebiet","Überwärmungsgebiet"]] = test1.loc[:,["Quellgebiet","Überwärmungsgebiet"]].fillna("ob")
              test1.fillna(-99.99,inplace=True)
              if Startzeit[i] < Endzeit[i]:
                test1["icolor"] = zeiten.vw_farbe[n]
              else:
                test1["icolor"] = zeiten.rw_farbe[n]
              if (i == 0) & (n == len(zeiten.zeit)-1):
                f = open(out_file_name+out_ext, 'w')
                writer = csv.writer(f)
                writer.writerow(["*"+nml["output"]["commres"]])
                writer.writerow(["*Koordinatensystem: "+in_nml["input"]["koord_system"]])
                f.writelines(["*u und v in m/s"+"\n", "*ex in kJ/m²"+"\n", "*hx in m\n"])
                test1.loc[:,["*lintyp","icolor","x","y","s","Nr","Zeit","u","v","ws","ex","hx", "Zeitschritt", "LU","Überwärmungsgebiet","Quellgebiet"]].to_csv(f,header=True, index=False, sep=" ")
                f.close()
              elif (i == len(traj_list)-1) & (n == 0):
                f = open(out_file_name+out_ext, 'a')
                writer = csv.writer(f)
                writer.writerow([99])
                test1.loc[:,["*lintyp","icolor","x","y","s","Nr","Zeit","u","v","ws","ex","hx", "Zeitschritt", "LU","Überwärmungsgebiet","Quellgebiet"]].to_csv(f, header=False, sep=" ", index=False)
                writer.writerow([99])
                writer.writerow([999])
                f.close()
              else:
                f = open(out_file_name+out_ext, 'a')
                writer = csv.writer(f)
                writer.writerow([99])
                test1.loc[:,["*lintyp","icolor","x","y","s","Nr","Zeit","u","v","ws","ex","hx", "Zeitschritt", "LU","Überwärmungsgebiet","Quellgebiet"]].to_csv(f, header=False, sep=" ", index=False)
                f.close()
        
        else:
          if test.Zeit.iloc[0] > test.Zeit.iloc[-1]:
            test["icolor"] = farbe[0]                # vorwärts ist blau
          else: 
            test["icolor"] = farbe[1]               # rückwärts ist rot
          test["x"] = test.x + xlu
          test["y"] = test.y + ylu
          test["*lintyp"] = lintyp
          test["s"] = strichdicke
          test["Nr"] = i+1
          test["u"] = test.u/100
          test["v"] = test.v/100
#          test["ws"] = (test.u**2+test.v**2)**0.5
          test["ex"] = test.ex/10
          test["hx"] = test.hx/10
          test = test.round({"x":2,"y":2, "u":2, "v":2, "ws":2, "Zeitschritt":2, "Zeit":2, "hx":2, "ex":2})
          test["LU"] = test.LU.astype(int)
          test.loc[:,["Quellgebiet","Überwärmungsgebiet"]] = test.loc[:,["Quellgebiet","Überwärmungsgebiet"]].fillna("ob")
          test.fillna(-99.99)
          if i == 0:
            f = open(out_file_name+out_ext, 'w')
            writer = csv.writer(f)
            writer.writerow(["*"+nml["output"]["commres"]])
            writer.writerow(["*Koordinatensystem: "+in_nml["input"]["koord_system"]])
            f.writelines(["*u und v in m/s"+"\n", "*ex in kJ/m²"+"\n", "*hx in m\n"])
            test.loc[:,["*lintyp","icolor","x","y","s","Nr","Zeit","u","v","ws","ex","hx", "Zeitschritt", "LU","Überwärmungsgebiet","Quellgebiet"]].to_csv(f,header=True, index=False, sep=" ")
            f.close()
          elif i == len(traj_list)-1:
            f = open(out_file_name+out_ext, 'a')
            writer = csv.writer(f)
            writer.writerow([99])
            test.loc[:,["*lintyp","icolor","x","y","s","Nr","Zeit","u","v","ws","ex","hx", "Zeitschritt", "LU","Überwärmungsgebiet","Quellgebiet"]].to_csv(f, header=False, sep=" ", index=False)
            writer.writerow([99])
            writer.writerow([999])
            f.close()
          else:
            f = open(out_file_name+out_ext, 'a')
            writer = csv.writer(f)
            writer.writerow([99])
            test.loc[:,["*lintyp","icolor","x","y","s","Nr","Zeit","u","v","ws","ex","hx", "Zeitschritt", "LU","Überwärmungsgebiet","Quellgebiet"]].to_csv(f, header=False, sep=" ", index=False)
            f.close()
      print("Die Trajektorien wurden als pl_datei gespeichert.")
    
    if "geojson" in out_format:
      lines_df = pd.DataFrame({"geometry":[], "Nr":[], "x_start":[], "y_start":[], "t_start":[], "t_end":[], "Überwärmungsgebiet":[], "Quellgebiet":[]}) #
      points_df = pd.DataFrame()
      geom = []
      for i in range(0, len(traj_list)):
        trajek = traj_list[i].copy()
        trajek["lahm"] = trajek.ws < ws_th
        trajek["summe"] = trajek.dt #["lahm"].cumsum()
        trajek["summe"] = trajek["summe"].sub(trajek["summe"].mask(trajek.lahm).ffill().fillna(0)).astype(int)
        trajek["lahmer"] = trajek.ws.diff() < 0
        trajek["summe1"] = trajek.summe
        trajek["summe1"] = trajek["summe1"].sub(trajek["summe1"].mask(trajek.lahmer).ffill().fillna(0)).astype(int)
        if max(trajek.summe1) >= th_no:
          ind_drop = trajek.loc[trajek.summe1>=th_no,"summe1"].index[0]
          trajekt = trajek.loc[:ind_drop,:"ws"]
          trajekt.loc[ind_drop,["u","v","ws","ex","hx","Zeitschritt"]] = [-1111,-1111,-11.11,-111.1,-111.1,-11.11]
        else: 
          trajekt = trajek.loc[:,:"ws"]

        test = pd.DataFrame(trajekt.loc[trajekt.index==0])
        if Startzeit[i] < Endzeit[i]:
          for dti in range(dt+int(test.loc[0,"Zeit"]),int(test.loc[0,"Zeit"])+int(max(trajekt.dt)),dt):
            test = pd.concat([test, trajekt.loc[np.argsort(np.argsort(round(abs(trajekt['Zeit']-dti))))==0]])
        else:
          for dti in range(int(test.loc[0,"Zeit"])-dt,int(test.loc[0,"Zeit"])-int(max(trajekt.dt)),-dt):
            test = pd.concat([test, trajekt.loc[np.argsort(np.argsort(round(abs(trajekt['Zeit']-dti))))==0]])
        test = pd.concat([test,  trajekt.loc[trajekt.index==max(trajekt.index)]], ignore_index=True)
        test = test.drop_duplicates()
        if "Ueberwaermungsgebiete" in in_nml["input"]:
          for q in range(len(test)):
            geometry = [Point(test.loc[q,"x"]+xlu, test.loc[q,"y"]+ylu)]
            stbz = gpd.GeoDataFrame(geometry=geometry, crs=in_nml["input"]["koord_system"])
            bezirk = gpd.sjoin(umriss, stbz, "inner", "contains")
            if len(bezirk[name_warm].values) > 0:        
                test.loc[q,"Überwärmungsgebiet"] = bezirk[name_warm].values[0]
            else:
                test.loc[q,"Überwärmungsgebiet"] = "ob"
        else: 
          test["Überwärmungsgebiet"] = "nc"
        if "Quellgebiete" in in_nml["input"]:
          geometry = [Point(test.loc[test.Zeit==min(test.Zeit),"x"]+xlu, test.loc[test.Zeit==min(test.Zeit),"y"]+ylu)]
          quell = gpd.GeoDataFrame(geometry=geometry, crs=in_nml["input"]["koord_system"])
          gebiet = gpd.sjoin(quellgebiet, quell, "right", "contains")
          if len(gebiet[name_quell].values) > 0:
            test["Quellgebiet"] = gebiet[name_quell].values[0]
          else:
            test["Quellgebiet"] = "ob"
        else:
          test["Quellgebiet"] = "nc"
        line = list(zip(round(test.x+xlu,2), round(test.y+ylu,2)))
        geom.append(LineString(line))
        lines_df.loc[i,"Nr":] = [i+1, test.x[0], test.y[0], test.Zeit[0], round(test.Zeit[max(test.index)],2), test.Überwärmungsgebiet[0], test.Quellgebiet[0]] #
        test["Nr"] = i+1
        points_df = pd.concat([points_df,test], ignore_index=True)
      lines_gdf = gpd.GeoDataFrame(lines_df, crs=in_nml["input"]["koord_system"], geometry = geom)
      # with open(directory+in_nml["output"]["out_dir"]+out_file+str(start_ein)+"_"+str(end_ein)+"_LU"+str(lu_start)+"_LS.geojson", "w") as file:
      #   file.write(lines_gdf.to_json())
      lines_gdf.to_file(out_file_name+"_LS.geojson")
      points_df["x"] = points_df.x+xlu
      points_df["y"] = points_df.y+ylu
      points_df["u"] = points_df.u/100
      points_df["v"] = points_df.v/100
#      points_df["ws"] = (points_df.u**2+points_df.v**2)**0.5
      points_df["ex"] = points_df.ex/10
      points_df["hx"] = points_df.hx/10
      points_df = points_df.round(2)
      points_df["geometry"] = points_df[["x", "y"]].apply(Point, axis=1)
      points_df.loc[:,["Quellgebiet","Überwärmungsgebiet"]] = points_df.loc[:,["Quellgebiet","Überwärmungsgebiet"]].fillna("ob")
      points_df.fillna(-99.99, inplace=True)
      points_gdf = gpd.GeoDataFrame(points_df[["Nr", "x", "y", "Zeit","u", "v","ws", "ex", "hx", "Zeitschritt", "LU", "Überwärmungsgebiet", "Quellgebiet", "geometry"]], crs=in_nml["input"]["koord_system"], geometry="geometry")
      points_gdf.to_file(out_file_name+"_P.geojson")
      # with open(directory+in_nml["output"]["out_dir"]+out_file+str(start_ein)+"_"+str(end_ein)+"_LU"+str(lu_start)+"_P.geojson", "w") as file:
      #   file.write(points_gdf.to_json())
      print("Die Trajektorien wurden im geojson-Format gespeichert.")
    
    print("Dauer des Speicherns = s", datetime.now()-now)
print("Gesamtdauer: ",datetime.now()-now1)
