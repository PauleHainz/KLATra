import os
import argparse
import pandas as pd
import geopandas as gpd
import numpy as np
import f90nml              # package for reading fortran namelists
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

########## functions #########################
# function to calculate polygon area for trajectory start density
# using the Shoelace-formula. From Stackoverflow.
# (https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates)
def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
  
def random_points_landuse(poly=None):
    # Find all points in model area
    alle_punkte =  list(iter.product([(x+0.5)*dx+xlu for x in range(0, nx)], [(y+0.5)*dx+ylu for y in range(0, ny)]))
    x1=[]; y1=[]
    for i in range(0,len(Landuse_test.loc[Landuse_test.DN==lu_start,])):
      lu_poly = Landuse_test.loc[Landuse_test.DN==lu_start,].reset_index().loc[i,"geometry"]
      lu_path = Path(list(lu_poly.exterior.coords), closed=True)
      if poly == None:
        nstart = lu_poly.area/10000*traj_dichte[li]
        n = m.ceil(nstart)
        punkte_lu_poly = [alle_punkte[i] for i in np.where(lu_path.contains_points(alle_punkte, radius=-0.05))[0]]
        if len(punkte_lu_poly) > 0:
          rng = default_rng(seed=51)               # set seed to get same random starting points for same parameters (same starting density, landuse class)    
          ind = list(rng.choice(len(punkte_lu_poly), size=n, replace=False))
          x1 = x1+[punkte_lu_poly[xi][0]-xlu for xi in ind]; y1 = y1+[punkte_lu_poly[xi][1]-ylu for xi in ind]
      else:
        punkte_innen = pd.DataFrame([alle_punkte[i] for i in np.where(poly.contains_points(alle_punkte, radius=-0.5))[0]], columns=["x", "y"])
        if poly.intersects_path(lu_path):
          nstart = lu_poly.area/10000*traj_dichte[li]
          n = m.ceil(nstart)
          punkte_innen2 = [punkte_innen.loc[i,] for i in np.where(lu_path.contains_points(punkte_innen, radius=-0.5))[0]]
          if len(punkte_innen2) > 0:
            punkte_lu_poly = [alle_punkte[i] for i in np.where(lu_path.contains_points(alle_punkte, radius=-0.05))[0]]
            if len(punkte_lu_poly) > 0:
              n = m.ceil(nstart*len(punkte_innen2)/len(punkte_lu_poly))
            rng = default_rng(seed=51)               # set seed to get same random starting points for same parameters (same rectangle, starting density, landuse class)
            ind = list(rng.choice(len(punkte_innen2), size=n, replace=False))
            x1 = x1+[punkte_innen2[xi][0]-xlu for xi in ind]; y1 = y1+[punkte_innen2[xi][1]-ylu for xi in ind]

    del(alle_punkte)
    ind_del = []
    for el in range(0,len(x1)):
      if Landuse.loc[ny-y1[el]/dx-0.5,x1[el]/dx-0.5] != lu_start:
        ind_del.append(el)
    for i in sorted(ind_del, reverse=True):
        del(x1[i], y1[i])

    return x1, y1

############ standard values #############
dt = 300                    # every 5 minutes
strichdicke = 0.5           # line width for KLAM_21 compatible output
lintyp = 0                  # line style for KLAM_21 compatible output
farbe = [180, 170]          # trajectories colors [forwards,backwards] for KLAM_21 compatible output
farbverlauf = True          # trajectories are divided in multiple segments with different colors every 30 minutes in the KLAM_21 compatible output
out_format = ['KLAM']       # KLAM_21 compatible output is standard output format
out_file = 'KLATRAout_'     # name of output files
out_ext = ".out"            # extension of output files
wind_feld = 'q'             # normally the mean wind field of the cold air layer (q) is used
traj_dichte = None          # needed as standard to calculate trajectories for each raster point if no trajectory start density is given
p = 2                       # powerfactor for inverse distance weighting
zeitschritt = 1             # if fixed time step is used, it is by standard 1 second
ws_th = 0.2                 # wind speed treshold for trajectories, trajectories are only cut below that treshold
th_time = 300                 # time in seconds with slower wind speeds than ws_th and slowing wind speeds to drop rest of trajectory (after this time)
########################################
now=datetime.now()        # starting time to calculate running time
print(now)                # print starting time in console
now1=now

# Get input file from command line
argParser = argparse.ArgumentParser()
argParser.add_argument("-f", "--file", help="complete path of the input namelist", required=True)  # namelist with all needed information to use this script

input_file = argParser.parse_args().file        # input file (fortran namelist) 
in_nml = f90nml.read(input_file)                # read input file

directory1 = os.getcwd()

directory = in_nml["input"]["directory"]        # get working directory
os.chdir(directory)                             # change to working dierectory

# save parameters from input file with in program used names
in_file = in_nml["input"]["in_file"]              # KLAM_21 input file

# change start and end times for trajectory calculation to list, if only one value
start_ein_list = in_nml["trajektorien"]["start"]
if type(start_ein_list) != list:
  start_ein_list = [start_ein_list]
end_ein_list = in_nml["trajektorien"]["end"]
if type(end_ein_list) != list:
  end_ein_list = [end_ein_list]

# overwrite standard values, if parameter in input file
if "windfeld" in in_nml["trajektorien"]:
  wind_feld = in_nml["trajektorien"]["windfeld"]
if "traj_start_dichte" in in_nml["trajektorien"]:
  traj_dichte = in_nml["trajektorien"]["traj_start_dichte"] 
  if traj_dichte != None and type(traj_dichte) != list:
    traj_dichte = [traj_dichte]
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
if "Ueberwaermungsgebiete" in in_nml["input"]:
  warm = in_nml["input"]["Ueberwaermungsgebiete"]
if "Quellgebiete" in in_nml["input"]:
  quelle = in_nml["input"]["Quellgebiete"]
if "windspeed_treshold" in in_nml["Trajektorien"]:
  ws_th = in_nml["Trajektorien"]["windspeed_treshold"]
if "treshold" in in_nml["Trajektorien"]:
  th_time = in_nml["Trajektorien"]["treshold"]


# read KLAM_21 input file and parameter from it
nml = f90nml.read(directory+in_file)  # KLAM_21 input file
dx = int(nml["grid"]["dx"])           # raster size
xlu = nml["grid"]["xrmap"]       # x SW-corner of model area
ylu = nml["grid"]["yrmap"]       # y SW-corner of model area
nx = nml["grid"]["nx"]           # number of columns
ny = nml["grid"]["ny"]           # number of rows
iozeit = pd.DataFrame(nml["output"]["iozeit"], columns=["zt"])   # times at which KLAM_21 saved simulation output

# If there was no data saved for simulation start (time 0) and no files were already written, this data is written to files
# As there is no wind at simulation start and no cold air layer all values in these files are 0.
if 0 not in iozeit:
  # if there are no files for time 0, for all needed parameters one will be written
  if "u"+wind_feld+"000000."+nml["output"]["xtension"] not in os.listdir(nml["output"]["resdir"]):
    uq000000 = pd.DataFrame(0, index=np.arange(ny), columns=np.arange(nx))
    # read head of simulation output to create same format
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
    print("Files for simulation time 0 written.")
  iozeit.loc[len(iozeit),"zt"] = 0
  iozeit = iozeit.loc[np.argsort(iozeit.zt)].reset_index(drop=True)

# Read KLAM_21 time step file if wanted and existing.
# save each used time step and create dataframe with timesteps and cumulated time
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
    print("KLAM_21 time steps should be used, but there is no dt_file. \nProgram exits now.")
    exit()
    
# Generate similar time step data, if fixed time step.
else:
  times = np.array([zeitschritt]*round(max(iozeit.zt)/zeitschritt))
  times = np.column_stack((times, times.cumsum()))
  
# read KLAM_21 output files
ut_list = []; vt_list = []; hx_list = []; ex_list = []
# create file name and read file, files are then appended to list of same variable
for zeit in iozeit.zt:
  ut_name = nml["output"]["resdir"]+"/u"+wind_feld+f'{int(zeit):06}'+"."+nml["output"]["xtension"]
  vt_name = nml["output"]["resdir"]+"/v"+wind_feld+f'{int(zeit):06}'+"."+nml["output"]["xtension"]
  hxt_name = nml["output"]["resdir"]+"/Hx"+f'{int(zeit):06}'+"."+nml["output"]["xtension"]
  ext_name = nml["output"]["resdir"]+"/Ex"+f'{int(zeit):06}'+"."+nml["output"]["xtension"]

  # read wind field and cold air layer data
  ut = np.array(pd.read_csv(ut_name, skiprows=8, delim_whitespace=True, header=None, encoding="ISO-8859-1"))
  vt = np.array(pd.read_csv(vt_name, skiprows=8, delim_whitespace=True, header=None, encoding="ISO-8859-1"))
  hxt = np.array(pd.read_csv(hxt_name, skiprows=8, delim_whitespace=True, header=None, encoding="ISO-8859-1"))
  ext = np.array(pd.read_fwf(ext_name, skiprows=9, widths=[5]*nx, header=None, encoding="ISO-8859-1"))

  ut_list.append(ut)
  vt_list.append(vt)
  hx_list.append(hxt)
  ex_list.append(ext)

# save all KLAM_21 output data in one DataFrame, to easily get data for the right time and parameter
wind_df = pd.DataFrame({"ut":ut_list, "vt":vt_list, "hx":hx_list, "ex":ex_list}, index=iozeit.zt)

# read Landuse file
Landuse = pd.read_csv(nml["landuse"]["fn_file"], skiprows=6, encoding="ISO-8859-1", delim_whitespace=True, header=None)

# if landuse based trajectory start point calculation wanted, read polygonized landuse file
if in_nml["trajektorien"]["art"] in ['Landuse', 'Landuse_Bereich', 'Landuse_Poly']:
  if "lu_file" in in_nml["trajektorien"]:
    if in_nml["trajektorien"]["lu_file"].endswith(".geojson"):
      Landuse_test = gpd.read_file(directory+in_nml["trajektorien"]["lu_file"])     # polygonized (saved as GeoJSON) Landuse needed to start calculation in right landuse class 
    else: 
      print("For landuse based calculation of starting points a geoJSON file of polygonized landuse data is needed.  \nProgram exits now.")
      exit()
  else: 
    print("For landuse based calculation of starting points a geoJSON file of polygonized landuse data is needed.  \nProgram exits now.")
    exit()
  if len(in_nml["trajektorien"]["landuse"]) > 1:
    if len(traj_dichte) == 1:
      print("There is only one trajectory starting density. It will be used for all landuse classes.")
      traj_dichte = traj_dichte * len(in_nml["trajektorien"]["landuse"])
    elif len(traj_dichte) != len(in_nml["trajektorien"]["landuse"]):
      print("""The number of land use classes and trajectory starting densities is not the same. 
      Please correct that or write only one starting density which will then be used for all landuse classes. \n
      Program exits now.""")
      exit()

# if target and source areas are wanted for a spatial join, those files are read
if "Ueberwaermungsgebiete" in in_nml["input"]:
  umriss = gpd.read_file(warm)
  name_warm = in_nml["input"]["name_waerme"]
if "Quellgebiete" in in_nml["input"]:
  quellgebiet = gpd.read_file(quelle)
  name_quell = in_nml["input"]["name_quell"]

# Loop over start and ending times for trajectory calculation
for end_ind in range(0, len(end_ein_list)):
  end_ein = [end_ein_list[end_ind]]
  start_ein = [start_ein_list[end_ind]]
  
  # information in console for which start and end time calculation starts
  print("start time: "+str(start_ein[0])+" s, end time: "+str(end_ein[0])+" s")
  
  # if landuse based calculation wanted a loop over all wanted landuse classes and start densities is needed, 
  # therefore all of the calculation and output is inside this if
  if in_nml["trajektorien"]["art"] in ['Landuse', 'Landuse_Bereich', 'Landuse_Poly']:
    lu_start_list = in_nml["trajektorien"]["landuse"]
    
    for li in range(0, len(lu_start_list)):
      lu_start = lu_start_list[li]
      # generate output file name
      out_file_name = directory+in_nml["output"]["out_dir"]+out_file+str(start_ein[0])+"_"+str(end_ein[0])+"_LU"+str(lu_start)
      print("Landuse = "+str(lu_start))
  
      if traj_dichte != None:
        if in_nml["trajektorien"]["art"] == 'Landuse_Poly': 
          # define polygone from input file coordinates
          x_eingabe = in_nml["trajektorien"]["x_koord"]
          y_eingabe = in_nml["trajektorien"]["y_koord"]
          koord = list(zip(x_eingabe,y_eingabe))
          poly = Path(koord)                 # create path from input coordinates ("wanted polygon")
          
          x1, y1 = random_points_landuse(poly)      

        ##### Landuse in part of the area (defined by rectangle)
        elif in_nml["trajektorien"]["art"] == 'Landuse_Bereich':
          x_eingabe = in_nml["trajektorien"]["x_koord"]           # diagonal corners of a rectangle
          y_eingabe = in_nml["trajektorien"]["y_koord"]           # absolute coordinates (don´t substract xlu or ylu)
          y_ein = [y_eingabe[0],y_eingabe[0],y_eingabe[1],y_eingabe[1]]
          koord = list(zip(2*x_eingabe,y_ein))
          poly = Path(koord)
  
          x1, y1 = random_points_landuse(poly)      

        #### Landuse in whole model area
        elif in_nml["trajektorien"]["art"] == 'Landuse':
  
          x1, y1 = random_points_landuse(poly)      
    
      ##### every cell with specific landuse class   
      else:
        landuse_koords = Landuse[Landuse==lu_start].stack().index.tolist()
        x1 = [(i+0.5)*dx-xlu for i in [x[1] for x in landuse_koords]]
        y1 = [(i+0.5)*dx-ylu for i in [x[0] for x in landuse_koords]]
      
      xy = pd.DataFrame({"x":x1,"y":y1})
      xy.drop_duplicates(inplace=True)
      x1 = xy.x; y1 = xy.y
      Startzeit = [start_ein[0]]*len(x1)
      Endzeit = [end_ein[0]]*len(x1)

      print("Calculation of starting points took so long: ", datetime.now()-now)
      if len(x1) == 0:
        print("With these parameters no trajectories are calculated.")
      else:
        if len(x1) == 1:
          print(str(len(x1))+" trajectory will be calculated.")
        else:
          print(str(len(x1))+" trajectories will be calculated.")

        now = datetime.now()
  
        traj_list=list()                    # list, in which all trajectories dataframes will be saved
        for p_traj in range(0,len(x1)):
          x = x1[p_traj]
          y = y1[p_traj]
          Zeit = Startzeit[p_traj]

          kf = x/dx                                     # x-index with decimal places
          lf = y/dx                                     # y-index with decimal places
          k = int(kf)                                   # x-index
          l = int(lf)                                   # y-index
  
          zeitxy_list = [[Zeit,x,y,Landuse.loc[ny-1-l,k]]]
          wind_list = []
          
          # KLAM_21 output times between which the trajectory calculation starts, used for temporal interpolation
          # if t1 or t2 is exceeded whilst calculatińg, these times are adjusted so that the trajectory time lays again between two output times (or is equal to one of them)
          t1, t2 = iozeit.loc[(iozeit["zt"]<=Zeit)][-1:].zt.item(),iozeit.loc[(iozeit["zt"]>Zeit)][:1].zt.item()
  
          # forward trajectory calculation
          if Startzeit[p_traj] < Endzeit[p_traj]:
            while Zeit < Endzeit[p_traj]:                     # calculates coordinates until end time is exceeded
              outside_model_area = False                                         # indicator, whether trajectory left the model area
  
              if Zeit > t2:                                   # if time used for temporal interpolation is exceeded, adjust t1 and t2
                t1, t2 = iozeit.loc[(iozeit["zt"]<=Zeit)][-1:].zt.item(),iozeit.loc[(iozeit["zt"]>Zeit)][:1].zt.item()
  
              kf = x/dx                                     # x-index with decimal places
              lf = y/dx                                     # y-index with decimal places
              k = int(kf)                                   # x-index
              l = int(lf)                                   # y-index
              xdec = kf-k                                   # x decimal place (to calculate quadrant)
              ydec = lf-l                                   # y decimal place (to calculate quadrant)
  
              # find quadrant of the raster cell where trajectory is "currently" and get raster cells indices for spatial interpolation
              if xdec >= 0.5:
                if ydec >= 0.5: indices = pd.DataFrame({"x":[k, k, k+1, k+1], "y":[l, l+1, l+1, l]}) # quadrant 1
                else: indices = pd.DataFrame({"x":[k, k, k+1, k+1], "y":[l, l-1, l-1, l]}) # quadrant 4
              else:
                if ydec >= 0.5: indices = pd.DataFrame({"x":[k, k, k-1, k-1], "y":[l, l+1, l+1, l]}) # quadrant 2
                else: indices = pd.DataFrame({"x":[k, k, k-1, k-1], "y":[l, l-1, l-1, l]}) # quadrant 3
  
              # spatial interpolation, if useful. Save spatially interpolated values for both times (earlier iozeit as 1, later as 2)
              if x < 0 or y < 0  or x > nml["grid"]["nx"]*dx or y > nml["grid"]["ny"]*dx : # point is located outside model region
                outside_model_area = True      # indicator for trajectory leaving model area
                break
              elif min(indices.x) < 0 or min(indices.y) < 0 or max(indices.x) > nx or max(indices.y) > ny:
                # point lies at the edge of the model area.
                # no spatial interpolation is used.
                v1 = wind_df.loc[t1,"vt"][ny-1-l,k]
                u1 = wind_df.loc[t1,"ut"][ny-1-l,k]
                v2 = wind_df.loc[t2,"vt"][ny-1-l,k]
                u2 = wind_df.loc[t2,"ut"][ny-1-l,k]
                hx1 = wind_df.loc[t1,"hx"][ny-1-l,k]
                hx2 = wind_df.loc[t2,"hx"][ny-1-l,k]
                ex1 = wind_df.loc[t1,"ex"][ny-1-l,k]
                ex2 = wind_df.loc[t2,"ex"][ny-1-l,k]
              elif (x == (k+0.5)*dx) & (y == (l+0.5)*dx):
                # no interpoolation as trajectory is located in raster cells middle
                v1 = wind_df.loc[t1,"vt"][ny-1-l,k]
                u1 = wind_df.loc[t1,"ut"][ny-1-l,k]
                v2 = wind_df.loc[t2,"vt"][ny-1-l,k]
                u2 = wind_df.loc[t2,"ut"][ny-1-l,k]
                hx1 = wind_df.loc[t1,"hx"][ny-1-l,k]
                hx2 = wind_df.loc[t2,"hx"][ny-1-l,k]
                ex1 = wind_df.loc[t1,"ex"][ny-1-l,k]
                ex2 = wind_df.loc[t2,"ex"][ny-1-l,k]
              else:
                # spatial interpolation of 4 raster cells
                # calculation of inverse distance weights
                indices["x_koord"] = (indices.x+0.5)*dx
                indices["y_koord"] = (indices.y+0.5)*dx
                indices["Entfernung"] = ((x-indices.x_koord)**2+(y-indices.y_koord)**2)**0.5
                indices["inverse_distance"] = indices.Entfernung**-p
                indices["gewicht"] = indices.inverse_distance/sum(indices.inverse_distance)
                # spatial Interpolation
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
  
              # calculate mean of timesteps used by KLAM_21 that are closest in simualtion time
              # if fixed time step calculating mean of same time steps -> doesn't change time step
              Zeitschritt = times[np.abs(times[:,1]-Zeit).argsort()[:2],0].min()
  
              g1 = (t2-Zeit)/(t2-t1)                                    # weight for data from iozeit 1
              g2 = (Zeit-t1)/(t2-t1)                                    # weight for data from iozeit 2
  
              v = v1*g1+v2*g2                                           # calculate spatially and temporally interpolated
              u = u1*g1+u2*g2                                           # values of the wind field and cold air layer
              hx = hx1*g1+hx2*g2                                        # for current position
              ex = ex1*g1+ex2*g2
  
              lu = Landuse.loc[ny-1-l,k]
              x = x + Zeitschritt*u/100                                 # calculate new x-Position (divided by 100, because KLAM_21 in cm/s)
              y = y + Zeitschritt*v/100                                 # calculate new y-Position
              Zeit = Zeit + Zeitschritt
              wind_list.append([u, v, Zeitschritt, g1, g2, t1, t2, ex, hx])
              zeitxy_list.append([Zeit, x, y, lu])
  
            wind_list.append([np.nan]*9)
            trajektorie = pd.DataFrame(np.column_stack((zeitxy_list, wind_list)), columns=["Zeit", "x", "y", "LU", "u", "v", "Zeitschritt", "g1", "g2", "t1", "t2", "ex", "hx"])
            trajektorie["dt"] = round(trajektorie["Zeit"]-Startzeit[p_traj])
            trajektorie["ws"] = ((trajektorie["u"]/100)**2+(trajektorie["v"]/100)**2)**0.5
            if outside_model_area:
              trajektorie.fillna(-55.55, inplace=True)
            traj_list.append(trajektorie)                               # When all points of a trajectory are calculated, the whole dataframe is appended to this list
  
          # backward trajectory calculation
          elif Startzeit[p_traj] > Endzeit[p_traj]:
            while Zeit > Endzeit[p_traj]:                               # calculates coordinates until time is less than end time 
              outside_model_area = False
  
              if Zeit < t1:                                             # if time is less than t1, t1 will become t2 and a new t1 is found
                t1, t2 = iozeit.loc[(iozeit["zt"]<=Zeit)][-1:].zt.item(),iozeit.loc[(iozeit["zt"]>Zeit)][:1].zt.item()
  
              kf = x/dx   # x-index with decimal places
              lf = y/dx   # y-index with decimal places
              k = int(kf) # x-Index
              l = int(lf) # y-Index
              xdec = kf-k # x decimal place (for quadrants)
              ydec = lf-l # y decimal place (for quadrants)
  
              # find quadrant of the raster cell where trajectory is "currently" and get raster cells indices for spatial interpolation
              if xdec >= 0.5:
                if ydec >= 0.5: indices = pd.DataFrame({"x":[k, k, k+1, k+1], "y":[l, l+1, l+1, l]}) # Q1
                else: indices = pd.DataFrame({"x":[k, k, k+1, k+1], "y":[l, l-1, l-1, l]}) # Q4
              else:
                if ydec >= 0.5: indices = pd.DataFrame({"x":[k, k, k-1, k-1], "y":[l, l+1, l+1, l]}) # Q2
                else: indices = pd.DataFrame({"x":[k, k, k-1, k-1], "y":[l, l-1, l-1, l]}) # Q3
  
              # spatial interpolation, if useful. Save spatially interpolated values for both times (earlier iozeit as 1, later as 2)
              if x < 0 or y < 0  or x > nml["grid"]["nx"]*dx or y > nml["grid"]["ny"]*dx : # point is located outside model region
                outside_model_area = True      # indicator for trajectory leaving model area
                break
              elif min(indices.x) < 0 or min(indices.y) < 0 or max(indices.x) > nx or max(indices.y) > ny:
                # point lies at the edge of the model area.
                # no spatial interpolation is used.
                v1 = wind_df.loc[t1,"vt"][ny-1-l,k]
                u1 = wind_df.loc[t1,"ut"][ny-1-l,k]
                v2 = wind_df.loc[t2,"vt"][ny-1-l,k]
                u2 = wind_df.loc[t2,"ut"][ny-1-l,k]
                hx1 = wind_df.loc[t1,"hx"][ny-1-l,k]
                hx2 = wind_df.loc[t2,"hx"][ny-1-l,k]
                ex1 = wind_df.loc[t1,"ex"][ny-1-l,k]
                ex2 = wind_df.loc[t2,"ex"][ny-1-l,k]
              elif (x == (k+0.5)*dx) & (y == (l+0.5)*dx):
                # no interpoolation as trajectory is located in raster cells middle
                v1 = wind_df.loc[t1,"vt"][ny-1-l,k]
                u1 = wind_df.loc[t1,"ut"][ny-1-l,k]
                v2 = wind_df.loc[t2,"vt"][ny-1-l,k]
                u2 = wind_df.loc[t2,"ut"][ny-1-l,k]
                hx1 = wind_df.loc[t1,"hx"][ny-1-l,k]
                hx2 = wind_df.loc[t2,"hx"][ny-1-l,k]
                ex1 = wind_df.loc[t1,"ex"][ny-1-l,k]
                ex2 = wind_df.loc[t2,"ex"][ny-1-l,k]
              else:
                # spatial interpolation of 4 raster cells
                # calculation of inverse distance weights
                indices["x_koord"] = (indices.x+0.5)*dx
                indices["y_koord"] = (indices.y+0.5)*dx
                indices["Entfernung"] = ((x-indices.x_koord)**2+(y-indices.y_koord)**2)**0.5
                indices["inverse_distance"] = indices.Entfernung**-p
                indices["gewicht"] = indices.inverse_distance/sum(indices.inverse_distance)
                # spatial Interpolation
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

              # calculate mean of timesteps used by KLAM_21 that are closest in simualtion time
              # if fixed time step calculating mean of same time steps -> doesn't change time step
              Zeitschritt = times[np.abs(times[:,1]-Zeit).argsort()[:2],0].min()
  
              g1 = (t2-Zeit)/(t2-t1)                                    # weight for data from iozeit 1
              g2 = (Zeit-t1)/(t2-t1)                                    # weight for data from iozeit 2
  
              v = v1*g1+v2*g2                                           # calculate spatially and temporally interpolated
              u = u1*g1+u2*g2                                           # values of the wind field and cold air layer
              hx = hx1*g1+hx2*g2                                        # for current position
              ex = ex1*g1+ex2*g2
  
              lu = Landuse.loc[ny-1-l,k]
              x = x - Zeitschritt*u/100                                 # calculate new x-Position (divided by 100, because KLAM_21 in cm/s)
              y = y - Zeitschritt*v/100                                 # calculate new y-Position
              Zeit = Zeit - Zeitschritt
              wind_list.append([u, v, Zeitschritt, g1, g2, t1, t2, ex, hx])
              zeitxy_list.append([Zeit, x, y, lu])
  
            wind_list.append([np.nan]*9)
            trajektorie = pd.DataFrame(np.column_stack((zeitxy_list, wind_list)), columns=["Zeit", "x", "y", "LU", "u", "v", "Zeitschritt", "g1", "g2", "t1", "t2", "ex", "hx"])
            trajektorie["dt"] = round(Startzeit[p_traj]-trajektorie["Zeit"])
            trajektorie["ws"] = ((trajektorie["u"]/100)**2+(trajektorie["v"]/100)**2)**0.5
            if outside_model_area:
              trajektorie.fillna(-55.55, inplace=True)
            traj_list.append(trajektorie)                                # When all points of a trajectory are calculated, the whole dataframe is appended to this list
            
        print("Trajectory calculation took so long: ", datetime.now()-now)
        now = datetime.now()
        #################################################################################
        # Saving the trajectories

        if "KLAM" in out_format:
          for i in range(0, len(traj_list)):   # loop over all calculated trajectories
            # get data frame from traj_list
            trajek = traj_list[i].copy()
            trajek["lahm"] = trajek.ws < ws_th      # find times where wind speed is smaller than the wind speed treshold
            trajek["summe"] = trajek.dt             
            # cumulative sum of time steps where wind speed is less than the treshold
            trajek["summe"] = trajek["summe"].sub(trajek["summe"].mask(trajek.lahm).ffill().fillna(0)).astype(int) 
            trajek["lahmer"] = trajek.ws.diff() < 0    # find decreasing wind speeds
            trajek["summe1"] = trajek.summe
            # cumulative sum of time steps where wind speed is less than treshold and decreasing
            trajek["summe1"] = trajek["summe1"].sub(trajek["summe1"].mask(trajek.lahmer).ffill().fillna(0)).astype(int)
            # cut trajectory if th_time is met or exceeded by cumulative sum of time steps where decreasing wind speed is less than treshold
            if max(trajek.summe1) >= th_time:
              ind_drop = trajek.loc[trajek.summe1>=th_time,"summe1"].index[0]
              trajekt = trajek.loc[:ind_drop,:"ws"]
              trajekt.loc[ind_drop,["u","v","ws","ex","hx","Zeitschritt"]] = [-1111,-1111,-11.11,-111.1,-111.1,-11.11]
            else: 
              trajekt = trajek.loc[:,:"ws"]
            # create data frame with trajectory starting point, values around every dt seconds and last calculated position (or last position before trajectory was cut 
            # because of wind speed)
            test = pd.DataFrame(trajekt.loc[trajekt.index==0])
            if Startzeit[i] < Endzeit[i]:
              for dti in range(dt+int(test.loc[0,"Zeit"]),int(test.loc[0,"Zeit"])+int(max(trajekt.dt)),dt):
                test = pd.concat([test, trajekt.loc[np.argsort(np.argsort(round(abs(trajekt['Zeit']-dti))))==0]])
            else:
              for dti in range(int(test.loc[0,"Zeit"])-dt,int(test.loc[0,"Zeit"])-int(max(trajekt.dt)),-dt):
                test = pd.concat([test, trajekt.loc[np.argsort(np.argsort(round(abs(trajekt['Zeit']-dti))))==0]])
            test = pd.concat([test,  trajekt.loc[trajekt.index==max(trajekt.index)]], ignore_index=True)
            test = test.drop_duplicates()                # if there are duplicate rows the duplicate is dropped.
            
            # spatial join of warm target areas with every output position, if wanted
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
            
            # spatial join of cold air production areas with temporally first position of trajectory (written for every output position)
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
            
            # if trajectories are divided in multiple segments with different colors every 30 minutes in the KLAM_21 compatible output, this happens here 
            if in_nml["output"]["Farbverlauf_mit_Zeit"] == True:
              zeiten = pd.DataFrame({"zeit":[1800*n for n in range(1,int(iozeit.zt[len(iozeit)-1]/1800+1))], 
              "vw_farbe":list(range(farbe[0],farbe[0]+10))+list(range(farbe[0],farbe[0]+6)), 
              "rw_farbe":list(range(farbe[1]+9,farbe[1]-1, -1))+list(range(farbe[1]+9,farbe[1]+3,-1))}) # data frame with all output times and backwards and forwards colours
              if Startzeit[i] < Endzeit[i]:                                                             # if forward trajectory
                zeiten = zeiten[(zeiten.zeit >= min(test.Zeit)) & (zeiten.zeit <= max(test.Zeit)+1800)] # + 1800 to account for positions between ending time and next half hour
                zeiten.reset_index(inplace=True, drop=True)                                             # take only rows for needed times from previous created dataframe
                for n in range(0, len(zeiten.zeit)):                                                    
                  # cut trajectory into 30 minute intervals and print those as different lines to KLAM_21 compatible file
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
                  # add more information as columns
                  test1["*lintyp"] = lintyp       # line style
                  test1["s"] = strichdicke        # line width
                  test1["x"] = test1.x+xlu        # convert to absolute coordinates
                  test1["y"] = test1.y+ylu        # convert to absolute coordinates
                  test1["Nr"] = i+1               # enumerate to know which parts belong together 
                  test1["u"] = test1.u/100        # convert to m s⁻¹
                  test1["v"] = test1.v/100        # convert to m s⁻¹
                  test1["ex"] = test1.ex/10       # convert to kJ/m²
                  test1["hx"] = test1.hx/10       # convert to m
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
                    writer.writerow(["*coordinate system: "+in_nml["input"]["koord_system"]])
                    f.writelines(["*u and v in m/s"+"\n", "*ex in kJ/m²"+"\n", "*hx in m\n"])
                    test1.loc[:,["*lintyp","icolor","x","y","s","Nr","Zeit","u","v","ws","ex","hx", "Zeitschritt", "LU","Überwärmungsgebiet","Quellgebiet"]].to_csv(f,header=True, index=False, sep=" ")
                    f.close()
                  elif (i == len(traj_list)-1) & (n == len(zeiten.zeit)-1):                 # if last part of last trajectory print also 999 to show end of file 
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
              else:                   # same for backward trajectory
                zeiten = zeiten[(zeiten.zeit <= max(test.Zeit)+1800) & (zeiten.zeit >= min(test.Zeit))]   # + 1800 to account for positions between ending time and next half hour
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
                  test1["*lintyp"] = lintyp       # line style
                  test1["s"] = strichdicke        # line width
                  test1["x"] = test1.x+xlu        # convert to absolute coordinates
                  test1["y"] = test1.y+ylu        # convert to absolute coordinates
                  test1["Nr"] = i+1               # enumerate to know which parts belong together 
                  test1["u"] = test1.u/100        # convert to m s⁻¹
                  test1["v"] = test1.v/100        # convert to m s⁻¹
                  test1["ex"] = test1.ex/10       # convert to kJ/m²
                  test1["hx"] = test1.hx/10       # convert to m
                  test1 = test1.round({"x":2,"y":2, "u":2, "v":2,"ws":2, "Zeitschritt":2, "Zeit":2, "hx":2, "ex":2})
                  test1["LU"] = test1.LU.astype(int)
                  test1.loc[:,["Quellgebiet","Überwärmungsgebiet"]] = test1.loc[:,["Quellgebiet","Überwärmungsgebiet"]].fillna("ob")
                  test1.fillna(-99.99,inplace=True)                     # fill na values
                  if Startzeit[i] < Endzeit[i]:
                    test1["icolor"] = zeiten.vw_farbe[n]
                  else:
                    test1["icolor"] = zeiten.rw_farbe[n]
                  if (i == 0) & (n == len(zeiten.zeit)-1):
                    f = open(out_file_name+out_ext, 'w')
                    writer = csv.writer(f)
                    writer.writerow(["*"+nml["output"]["commres"]])
                    writer.writerow(["*coordinate system: "+in_nml["input"]["koord_system"]])
                    f.writelines(["*u and v in m/s"+"\n", "*ex in kJ/m²"+"\n", "*hx in m\n"])
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
            
            else:                   # if time should not be displayed by colour in KLAM_21 GUI
              if test.Zeit.iloc[0] > test.Zeit.iloc[-1]:
                test["icolor"] = farbe[0]                 # forward colour is normally blue
              else: 
                test["icolor"] = farbe[1]                 # backward colour is normally red
              test["x"] = test.x + xlu       # convert to absolute coordinates
              test["y"] = test.y + ylu       # convert to absolute coordinates
              test["*lintyp"] = lintyp       # line style
              test["s"] = strichdicke        # line width
              test["Nr"] = i+1               # enumerate to know which parts belong together
              test["u"] = test.u/100         # convert to m s⁻¹
              test["v"] = test.v/100         # convert to m s⁻¹
              test["ex"] = test.ex/10        # convert to kJ/m
              test["hx"] = test.hx/10        # convert to m
              test = test.round({"x":2,"y":2, "u":2, "v":2, "ws":2, "Zeitschritt":2, "Zeit":2, "hx":2, "ex":2})
              test["LU"] = test.LU.astype(int)
              test.loc[:,["Quellgebiet","Überwärmungsgebiet"]] = test.loc[:,["Quellgebiet","Überwärmungsgebiet"]].fillna("ob")
              test.fillna(-99.99,inplace=True)
              if i == 0:
                f = open(out_file_name+out_ext, 'w')
                writer = csv.writer(f)
                writer.writerow(["*"+nml["output"]["commres"]])
                writer.writerow(["*coordinate system: "+in_nml["input"]["koord_system"]])
                f.writelines(["*u and v in m/s"+"\n", "*ex in kJ/m²"+"\n", "*hx in m\n"])
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
          print("Trajectories were saved as KLAM_21 compatible files.")
        
        if "geojson" in out_format:                                       # if geojson output wanted
          lines_df = pd.DataFrame({"geometry":[], "Nr":[], "x_start":[], "y_start":[], "t_start":[], "t_end":[], "Überwärmungsgebiet":[], "Quellgebiet":[]}) 
          points_df = pd.DataFrame()                                      # create empty data frames for output as lines and as points
          geom = []
          for i in range(0, len(traj_list)):   # loop over all calculated trajectories
            # get data frame from traj_list
            trajek = traj_list[i].copy()
            trajek["lahm"] = trajek.ws < ws_th      # find times where wind speed is smaller than the wind speed treshold
            trajek["summe"] = trajek.dt             
            # cumulative sum of time steps where wind speed is less than the treshold
            trajek["summe"] = trajek["summe"].sub(trajek["summe"].mask(trajek.lahm).ffill().fillna(0)).astype(int) 
            trajek["lahmer"] = trajek.ws.diff() < 0    # find decreasing wind speeds
            trajek["summe1"] = trajek.summe
            # cumulative sum of time steps where wind speed is less than treshold and decreasing
            trajek["summe1"] = trajek["summe1"].sub(trajek["summe1"].mask(trajek.lahmer).ffill().fillna(0)).astype(int)
            # cut trajectory if th_time is met or exceeded by cumulative sum of time steps where decreasing wind speed is less than treshold
            if max(trajek.summe1) >= th_time:
              ind_drop = trajek.loc[trajek.summe1>=th_time,"summe1"].index[0]
              trajekt = trajek.loc[:ind_drop,:"ws"]
              trajekt.loc[ind_drop,["u","v","ws","ex","hx","Zeitschritt"]] = [-1111,-1111,-11.11,-111.1,-111.1,-11.11]
            else: 
              trajekt = trajek.loc[:,:"ws"]
            # create data frame with trajectory starting point, values around every dt seconds and last calculated position (or last position before trajectory was cut 
            # because of wind speed)
            test = pd.DataFrame(trajekt.loc[trajekt.index==0])
            if Startzeit[i] < Endzeit[i]:
              for dti in range(dt+int(test.loc[0,"Zeit"]),int(test.loc[0,"Zeit"])+int(max(trajekt.dt)),dt):
                test = pd.concat([test, trajekt.loc[np.argsort(np.argsort(round(abs(trajekt['Zeit']-dti))))==0]])
            else:
              for dti in range(int(test.loc[0,"Zeit"])-dt,int(test.loc[0,"Zeit"])-int(max(trajekt.dt)),-dt):
                test = pd.concat([test, trajekt.loc[np.argsort(np.argsort(round(abs(trajekt['Zeit']-dti))))==0]])
            test = pd.concat([test,  trajekt.loc[trajekt.index==max(trajekt.index)]], ignore_index=True)
            test = test.drop_duplicates()                # if there are duplicate rows the duplicate is dropped.
            
            # spatial join of warm target areas with every output position, if wanted
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
            
            # spatial join of cold air production areas with temporally first position of trajectory (written for every output position)
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
          #   file.write(lines_gdf.to_json())                                     # alternative for saving
          lines_gdf.to_file(out_file_name+"_LS.geojson")                          # save trajectories as Linestrings
          points_df["x"] = points_df.x+xlu        # convert to absolute coordinates
          points_df["y"] = points_df.y+ylu        # convert to absolute coordinates
          points_df["u"] = points_df.u/100        # convert to m s⁻¹
          points_df["v"] = points_df.v/100        # convert to m s⁻¹
          points_df["ex"] = points_df.ex/10       # convert to kJ/m
          points_df["hx"] = points_df.hx/10       # convert to m
          points_df["geometry"] = points_df[["x", "y"]].apply(Point, axis=1)
          points_df.loc[:,["Quellgebiet","Überwärmungsgebiet"]] = points_df.loc[:,["Quellgebiet","Überwärmungsgebiet"]].fillna("ob")
          points_df.fillna(-99.99, inplace=True)
          points_df = points_df.round(2)          # round to 2 decimal points
          points_gdf = gpd.GeoDataFrame(points_df[["Nr", "x", "y", "Zeit","u", "v","ws", "ex", "hx", "Zeitschritt", "LU", "Überwärmungsgebiet", "Quellgebiet", "geometry"]], crs=in_nml["input"]["koord_system"], geometry="geometry")
          points_gdf.to_file(out_file_name+"_P.geojson")
          # with open(directory+in_nml["output"]["out_dir"]+out_file+str(start_ein)+"_"+str(end_ein)+"_LU"+str(lu_start)+"_P.geojson", "w") as file:    # alternative way
          #   file.write(points_gdf.to_json())                                                                                                          # to save as geojson
          print("Trajectories saved as geoJSON LineStrings and Points.")
  
        print("Saving took so long: ", datetime.now()-now)
    # 
  else:
    out_file_name = directory+in_nml["output"]["out_dir"]+out_file+str(start_ein[0])+"_"+str(end_ein[0])
    ##### Polygon
    if in_nml["trajektorien"]["art"] == 'Polygon':
      x_eingabe = in_nml["trajektorien"]["x_koord"]
      y_eingabe = in_nml["trajektorien"]["y_koord"]
      koord = list(zip(x_eingabe,y_eingabe))
      poly = Path(koord)
      
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
        print("Please add traj_start_dichte in the input file or change art to='Rechteck' instead of 'Rechteck_r'.")
        print("Program exits now.")
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
      print("No valid value in art in the input file. Program exits now.")
      exit()
      
       
    xy = pd.DataFrame({"x":x1,"y":y1})
    xy.drop_duplicates(inplace=True)
    x1 = xy.x; y1 = xy.y
    Startzeit = [start_ein[0]]*len(x1)
    Endzeit = [end_ein[0]]*len(x1)

    print("Calculation of starting points took so long: ", datetime.now()-now)
    if len(x1) == 0:
      print("With these parameters no trajectories are calculated.")
    else:
      if len(x1) == 1:
        print(str(len(x1))+" trajectory will be calculated.")
      else:
        print(str(len(x1))+" trajectories will be calculated.")
      
      now = datetime.now()
  
      traj_list=list()                    # list, in which all trajectories dataframes will be saved
      for p_traj in range(0,len(x1)):
        x = x1[p_traj]
        y = y1[p_traj]
        Zeit = Startzeit[p_traj]
  
        kf = x/dx                                     # x-index with decimal places
        lf = y/dx                                     # y-index with decimal places
        k = int(kf)                                   # x-index
        l = int(lf)                                   # y-index
  
        zeitxy_list = [[Zeit,x,y,Landuse.loc[ny-1-l,k]]]
        wind_list = []
        
        # KLAM_21 output times between which the trajectory calculation starts, used for temporal interpolation
        # if t1 or t2 is exceeded whilst calculatińg, these times are adjusted so that the trajectory time lays again between two output times (or is equal to one of them)
        t1, t2 = iozeit.loc[(iozeit["zt"]<=Zeit)][-1:].zt.item(),iozeit.loc[(iozeit["zt"]>Zeit)][:1].zt.item()
  
        # forward trajectory calculation
        if Startzeit[p_traj] < Endzeit[p_traj]:
          while Zeit < Endzeit[p_traj]:                     # calculates coordinates until end time is exceeded
            outside_model_area = False                                         # indicator, whether trajectory left the model area
  
            if Zeit > t2:                                   # if time used for temporal interpolation is exceeded, adjust t1 and t2
              t1, t2 = iozeit.loc[(iozeit["zt"]<=Zeit)][-1:].zt.item(),iozeit.loc[(iozeit["zt"]>Zeit)][:1].zt.item()
  
            kf = x/dx                                     # x-index with decimal places
            lf = y/dx                                     # y-index with decimal places
            k = int(kf)                                   # x-index
            l = int(lf)                                   # y-index
            xdec = kf-k                                   # x decimal place (to calculate quadrant)
            ydec = lf-l                                   # y decimal place (to calculate quadrant)
  
            # find quadrant of the raster cell where trajectory is "currently" and get raster cells indices for spatial interpolation
            if xdec >= 0.5:
              if ydec >= 0.5: indices = pd.DataFrame({"x":[k, k, k+1, k+1], "y":[l, l+1, l+1, l]}) # quadrant 1
              else: indices = pd.DataFrame({"x":[k, k, k+1, k+1], "y":[l, l-1, l-1, l]}) # quadrant 4
            else:
              if ydec >= 0.5: indices = pd.DataFrame({"x":[k, k, k-1, k-1], "y":[l, l+1, l+1, l]}) # quadrant 2
              else: indices = pd.DataFrame({"x":[k, k, k-1, k-1], "y":[l, l-1, l-1, l]}) # quadrant 3
  
            # spatial interpolation, if useful. Save spatially interpolated values for both times (earlier iozeit as 1, later as 2)
            if x < 0 or y < 0  or x > nml["grid"]["nx"]*dx or y > nml["grid"]["ny"]*dx : # point is located outside model region
              outside_model_area = True      # indicator for trajectory leaving model area
              break
            elif min(indices.x) < 0 or min(indices.y) < 0 or max(indices.x) > nx or max(indices.y) > ny:
              # point lies at the edge of the model area.
              # no spatial interpolation is used.
              v1 = wind_df.loc[t1,"vt"][ny-1-l,k]
              u1 = wind_df.loc[t1,"ut"][ny-1-l,k]
              v2 = wind_df.loc[t2,"vt"][ny-1-l,k]
              u2 = wind_df.loc[t2,"ut"][ny-1-l,k]
              hx1 = wind_df.loc[t1,"hx"][ny-1-l,k]
              hx2 = wind_df.loc[t2,"hx"][ny-1-l,k]
              ex1 = wind_df.loc[t1,"ex"][ny-1-l,k]
              ex2 = wind_df.loc[t2,"ex"][ny-1-l,k]
            elif (x == (k+0.5)*dx) & (y == (l+0.5)*dx):
              # no interpoolation as trajectory is located in raster cells middle
              v1 = wind_df.loc[t1,"vt"][ny-1-l,k]
              u1 = wind_df.loc[t1,"ut"][ny-1-l,k]
              v2 = wind_df.loc[t2,"vt"][ny-1-l,k]
              u2 = wind_df.loc[t2,"ut"][ny-1-l,k]
              hx1 = wind_df.loc[t1,"hx"][ny-1-l,k]
              hx2 = wind_df.loc[t2,"hx"][ny-1-l,k]
              ex1 = wind_df.loc[t1,"ex"][ny-1-l,k]
              ex2 = wind_df.loc[t2,"ex"][ny-1-l,k]
            else:
              # spatial interpolation of 4 raster cells
              # calculation of inverse distance weights
              indices["x_koord"] = (indices.x+0.5)*dx
              indices["y_koord"] = (indices.y+0.5)*dx
              indices["Entfernung"] = ((x-indices.x_koord)**2+(y-indices.y_koord)**2)**0.5
              indices["inverse_distance"] = indices.Entfernung**-p
              indices["gewicht"] = indices.inverse_distance/sum(indices.inverse_distance)
              # spatial Interpolation
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
  
            # calculate mean of timesteps used by KLAM_21 that are closest in simualtion time
            # if fixed time step calculating mean of same time steps -> doesn't change time step
            Zeitschritt = times[np.abs(times[:,1]-Zeit).argsort()[:2],0].min()
  
            g1 = (t2-Zeit)/(t2-t1)                                    # weight for data from iozeit 1
            g2 = (Zeit-t1)/(t2-t1)                                    # weight for data from iozeit 2
  
            v = v1*g1+v2*g2                                           # calculate spatially and temporally interpolated
            u = u1*g1+u2*g2                                           # values of the wind field and cold air layer
            hx = hx1*g1+hx2*g2                                        # for current position
            ex = ex1*g1+ex2*g2
  
            lu = Landuse.loc[ny-1-l,k]
            x = x + Zeitschritt*u/100                                 # calculate new x-Position (divided by 100, because KLAM_21 in cm/s)
            y = y + Zeitschritt*v/100                                 # calculate new y-Position
            Zeit = Zeit + Zeitschritt
            wind_list.append([u, v, Zeitschritt, g1, g2, t1, t2, ex, hx])
            zeitxy_list.append([Zeit, x, y, lu])
  
          wind_list.append([np.nan]*9)
          trajektorie = pd.DataFrame(np.column_stack((zeitxy_list, wind_list)), columns=["Zeit", "x", "y", "LU", "u", "v", "Zeitschritt", "g1", "g2", "t1", "t2", "ex", "hx"])
          trajektorie["dt"] = round(trajektorie["Zeit"]-Startzeit[p_traj])
          trajektorie["ws"] = ((trajektorie["u"]/100)**2+(trajektorie["v"]/100)**2)**0.5
          if outside_model_area:
            trajektorie.fillna(-55.55, inplace=True)
          traj_list.append(trajektorie)                               # When all points of a trajectory are calculated, the whole dataframe is appended to this list
  
        # backward trajectory calculation
        elif Startzeit[p_traj] > Endzeit[p_traj]:
          while Zeit > Endzeit[p_traj]:                               # calculates coordinates until time is less than end time 
            outside_model_area = False
  
            if Zeit < t1:                                             # if time is less than t1, t1 will become t2 and a new t1 is found
              t1, t2 = iozeit.loc[(iozeit["zt"]<=Zeit)][-1:].zt.item(),iozeit.loc[(iozeit["zt"]>Zeit)][:1].zt.item()
  
            kf = x/dx   # x-index with decimal places
            lf = y/dx   # y-index with decimal places
            k = int(kf) # x-Index
            l = int(lf) # y-Index
            xdec = kf-k # x decimal place (for quadrants)
            ydec = lf-l # y decimal place (for quadrants)
  
            # find quadrant of the raster cell where trajectory is "currently" and get raster cells indices for spatial interpolation
            if xdec >= 0.5:
              if ydec >= 0.5: indices = pd.DataFrame({"x":[k, k, k+1, k+1], "y":[l, l+1, l+1, l]}) # Q1
              else: indices = pd.DataFrame({"x":[k, k, k+1, k+1], "y":[l, l-1, l-1, l]}) # Q4
            else:
              if ydec >= 0.5: indices = pd.DataFrame({"x":[k, k, k-1, k-1], "y":[l, l+1, l+1, l]}) # Q2
              else: indices = pd.DataFrame({"x":[k, k, k-1, k-1], "y":[l, l-1, l-1, l]}) # Q3
  
            # spatial interpolation, if useful. Save spatially interpolated values for both times (earlier iozeit as 1, later as 2)
            if x < 0 or y < 0  or x > nml["grid"]["nx"]*dx or y > nml["grid"]["ny"]*dx : # point is located outside model region
              outside_model_area = True      # indicator for trajectory leaving model area
              break
            elif min(indices.x) < 0 or min(indices.y) < 0 or max(indices.x) > nx or max(indices.y) > ny:
              # point lies at the edge of the model area.
              # no spatial interpolation is used.
              v1 = wind_df.loc[t1,"vt"][ny-1-l,k]
              u1 = wind_df.loc[t1,"ut"][ny-1-l,k]
              v2 = wind_df.loc[t2,"vt"][ny-1-l,k]
              u2 = wind_df.loc[t2,"ut"][ny-1-l,k]
              hx1 = wind_df.loc[t1,"hx"][ny-1-l,k]
              hx2 = wind_df.loc[t2,"hx"][ny-1-l,k]
              ex1 = wind_df.loc[t1,"ex"][ny-1-l,k]
              ex2 = wind_df.loc[t2,"ex"][ny-1-l,k]
            elif (x == (k+0.5)*dx) & (y == (l+0.5)*dx):
              # no interpoolation as trajectory is located in raster cells middle
              v1 = wind_df.loc[t1,"vt"][ny-1-l,k]
              u1 = wind_df.loc[t1,"ut"][ny-1-l,k]
              v2 = wind_df.loc[t2,"vt"][ny-1-l,k]
              u2 = wind_df.loc[t2,"ut"][ny-1-l,k]
              hx1 = wind_df.loc[t1,"hx"][ny-1-l,k]
              hx2 = wind_df.loc[t2,"hx"][ny-1-l,k]
              ex1 = wind_df.loc[t1,"ex"][ny-1-l,k]
              ex2 = wind_df.loc[t2,"ex"][ny-1-l,k]
            else:
              # spatial interpolation of 4 raster cells
              # calculation of inverse distance weights
              indices["x_koord"] = (indices.x+0.5)*dx
              indices["y_koord"] = (indices.y+0.5)*dx
              indices["Entfernung"] = ((x-indices.x_koord)**2+(y-indices.y_koord)**2)**0.5
              indices["inverse_distance"] = indices.Entfernung**-p
              indices["gewicht"] = indices.inverse_distance/sum(indices.inverse_distance)
              # spatial Interpolation
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
  
            # calculate mean of timesteps used by KLAM_21 that are closest in simualtion time
            # if fixed time step calculating mean of same time steps -> doesn't change time step
            Zeitschritt = times[np.abs(times[:,1]-Zeit).argsort()[:2],0].min()
  
            g1 = (t2-Zeit)/(t2-t1)                                    # weight for data from iozeit 1
            g2 = (Zeit-t1)/(t2-t1)                                    # weight for data from iozeit 2
  
            v = v1*g1+v2*g2                                           # calculate spatially and temporally interpolated
            u = u1*g1+u2*g2                                           # values of the wind field and cold air layer
            hx = hx1*g1+hx2*g2                                        # for current position
            ex = ex1*g1+ex2*g2
  
            lu = Landuse.loc[ny-1-l,k]
            x = x - Zeitschritt*u/100                                 # calculate new x-Position (divided by 100, because KLAM_21 output is in cm/s)
            y = y - Zeitschritt*v/100                                 # calculate new y-Position
            Zeit = Zeit - Zeitschritt
            wind_list.append([u, v, Zeitschritt, g1, g2, t1, t2, ex, hx])
            zeitxy_list.append([Zeit, x, y, lu])
  
          wind_list.append([np.nan]*9)
          trajektorie = pd.DataFrame(np.column_stack((zeitxy_list, wind_list)), columns=["Zeit", "x", "y", "LU", "u", "v", "Zeitschritt", "g1", "g2", "t1", "t2", "ex", "hx"])
          trajektorie["dt"] = round(Startzeit[p_traj]-trajektorie["Zeit"])
          trajektorie["ws"] = ((trajektorie["u"]/100)**2+(trajektorie["v"]/100)**2)**0.5
          if outside_model_area:
            trajektorie.fillna(-55.55, inplace=True)
          traj_list.append(trajektorie)                                # When all points of a trajectory are calculated, the whole dataframe is appended to this list
  
      print("Trajectory calculation took so long: ", datetime.now()-now)
      now = datetime.now()
      #################################################################################
      # Saving the trajectories
  
      if "KLAM" in out_format:
        for i in range(0, len(traj_list)):   # loop over all calculated trajectories
          # get data frame from traj_list
          trajek = traj_list[i].copy()
          trajek["lahm"] = trajek.ws < ws_th      # find times where wind speed is smaller than the wind speed treshold
          trajek["summe"] = trajek.dt             
          # cumulative sum of time steps where wind speed is less than the treshold
          trajek["summe"] = trajek["summe"].sub(trajek["summe"].mask(trajek.lahm).ffill().fillna(0)).astype(int) 
          trajek["lahmer"] = trajek.ws.diff() < 0    # find decreasing wind speeds
          trajek["summe1"] = trajek.summe
          # cumulative sum of time steps where wind speed is less than treshold and decreasing
          trajek["summe1"] = trajek["summe1"].sub(trajek["summe1"].mask(trajek.lahmer).ffill().fillna(0)).astype(int)
          # cut trajectory if th_time is met or exceeded by cumulative sum of time steps where decreasing wind speed is less than treshold
          if max(trajek.summe1) >= th_time:
            ind_drop = trajek.loc[trajek.summe1>=th_time,"summe1"].index[0]
            trajekt = trajek.loc[:ind_drop,:"ws"]
            trajekt.loc[ind_drop,["u","v","ws","ex","hx","Zeitschritt"]] = [-1111,-1111,-11.11,-111.1,-111.1,-11.11]
          else: 
            trajekt = trajek.loc[:,:"ws"]
          # create data frame with trajectory starting point, values around every dt seconds and last calculated position (or last position before trajectory was cut 
          # because of wind speed)
          test = pd.DataFrame(trajekt.loc[trajekt.index==0])
          if Startzeit[i] < Endzeit[i]:
            for dti in range(dt+int(test.loc[0,"Zeit"]),int(test.loc[0,"Zeit"])+int(max(trajekt.dt)),dt):
              test = pd.concat([test, trajekt.loc[np.argsort(np.argsort(round(abs(trajekt['Zeit']-dti))))==0]])
          else:
            for dti in range(int(test.loc[0,"Zeit"])-dt,int(test.loc[0,"Zeit"])-int(max(trajekt.dt)),-dt):
              test = pd.concat([test, trajekt.loc[np.argsort(np.argsort(round(abs(trajekt['Zeit']-dti))))==0]])
          test = pd.concat([test,  trajekt.loc[trajekt.index==max(trajekt.index)]], ignore_index=True)
          test = test.drop_duplicates()                # if there are duplicate rows the duplicate is dropped.
          
          # spatial join of warm target areas with every output position, if wanted
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
          
          # spatial join of cold air production areas with temporally first position of trajectory (written for every output position)
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
          
          # if trajectories are divided in multiple segments with different colors every 30 minutes in the KLAM_21 compatible output, this happens here 
          if in_nml["output"]["Farbverlauf_mit_Zeit"] == True:
            zeiten = pd.DataFrame({"zeit":[1800*n for n in range(1,int(iozeit.zt[len(iozeit)-1]/1800+1))], 
            "vw_farbe":list(range(farbe[0],farbe[0]+10))+list(range(farbe[0],farbe[0]+6)), 
            "rw_farbe":list(range(farbe[1]+9,farbe[1]-1, -1))+list(range(farbe[1]+9,farbe[1]+3,-1))}) # data frame with all output times and backwards and forwards colours
            if Startzeit[i] < Endzeit[i]:                                                             # if forward trajectory
              zeiten = zeiten[(zeiten.zeit >= min(test.Zeit)) & (zeiten.zeit <= max(test.Zeit)+1800)] # + 1800 to account for positions between ending time and next half hour
              zeiten.reset_index(inplace=True, drop=True)                                             # take only rows for needed times from previous created dataframe
              for n in range(0, len(zeiten.zeit)):                                                    
                # cut trajectory into 30 minute intervals and print those as different lines to KLAM_21 compatible file
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
                # add more information as columns
                test1["*lintyp"] = lintyp       # line style
                test1["s"] = strichdicke        # line width
                test1["x"] = test1.x+xlu        # convert to absolute coordinates
                test1["y"] = test1.y+ylu        # convert to absolute coordinates
                test1["Nr"] = i+1               # enumerate to know which parts belong together 
                test1["u"] = test1.u/100        # convert to m s⁻¹
                test1["v"] = test1.v/100        # convert to m s⁻¹
                test1["ex"] = test1.ex/10       # convert to kJ/m²
                test1["hx"] = test1.hx/10       # convert to m
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
                  writer.writerow(["*coordinate system: "+in_nml["input"]["koord_system"]])
                  f.writelines(["*u and v in m/s"+"\n", "*ex in kJ/m²"+"\n", "*hx in m\n"])
                  test1.loc[:,["*lintyp","icolor","x","y","s","Nr","Zeit","u","v","ws","ex","hx", "Zeitschritt", "LU","Überwärmungsgebiet","Quellgebiet"]].to_csv(f,header=True, index=False, sep=" ")
                  f.close()
                elif (i == len(traj_list)-1) & (n == len(zeiten.zeit)-1):                 # if last part of last trajectory print also 999 to show end of file 
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
            else:                   # same for backward trajectory
              zeiten = zeiten[(zeiten.zeit <= max(test.Zeit)+1800) & (zeiten.zeit >= min(test.Zeit))]   # + 1800 to account for positions between ending time and next half hour
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
                test1["*lintyp"] = lintyp       # line style
                test1["s"] = strichdicke        # line width
                test1["x"] = test1.x+xlu        # convert to absolute coordinates
                test1["y"] = test1.y+ylu        # convert to absolute coordinates
                test1["Nr"] = i+1               # enumerate to know which parts belong together 
                test1["u"] = test1.u/100        # convert to m s⁻¹
                test1["v"] = test1.v/100        # convert to m s⁻¹
                test1["ex"] = test1.ex/10       # convert to kJ/m²
                test1["hx"] = test1.hx/10       # convert to m
                test1 = test1.round({"x":2,"y":2, "u":2, "v":2,"ws":2, "Zeitschritt":2, "Zeit":2, "hx":2, "ex":2})
                test1["LU"] = test1.LU.astype(int)
                test1.loc[:,["Quellgebiet","Überwärmungsgebiet"]] = test1.loc[:,["Quellgebiet","Überwärmungsgebiet"]].fillna("ob")
                test1.fillna(-99.99,inplace=True)                     # fill na values
                if Startzeit[i] < Endzeit[i]:
                  test1["icolor"] = zeiten.vw_farbe[n]
                else:
                  test1["icolor"] = zeiten.rw_farbe[n]
                if (i == 0) & (n == len(zeiten.zeit)-1):
                  f = open(out_file_name+out_ext, 'w')
                  writer = csv.writer(f)
                  writer.writerow(["*"+nml["output"]["commres"]])
                  writer.writerow(["*coordinate system: "+in_nml["input"]["koord_system"]])
                  f.writelines(["*u and v in m/s"+"\n", "*ex in kJ/m²"+"\n", "*hx in m\n"])
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
          
          else:                   # if time should not be displayed by colour in KLAM_21 GUI
            if test.Zeit.iloc[0] > test.Zeit.iloc[-1]:
              test["icolor"] = farbe[0]                 # forward colour is normally blue
            else: 
              test["icolor"] = farbe[1]                 # backward colour is normally red
            test["x"] = test.x + xlu       # convert to absolute coordinates
            test["y"] = test.y + ylu       # convert to absolute coordinates
            test["*lintyp"] = lintyp       # line style
            test["s"] = strichdicke        # line width
            test["Nr"] = i+1               # enumerate to know which parts belong together
            test["u"] = test.u/100         # convert to m s⁻¹
            test["v"] = test.v/100         # convert to m s⁻¹
            test["ex"] = test.ex/10        # convert to kJ/m
            test["hx"] = test.hx/10        # convert to m
            test = test.round({"x":2,"y":2, "u":2, "v":2, "ws":2, "Zeitschritt":2, "Zeit":2, "hx":2, "ex":2})
            test["LU"] = test.LU.astype(int)
            test.loc[:,["Quellgebiet","Überwärmungsgebiet"]] = test.loc[:,["Quellgebiet","Überwärmungsgebiet"]].fillna("ob")
            test.fillna(-99.99,inplace=True)
            if i == 0:
              f = open(out_file_name+out_ext, 'w')
              writer = csv.writer(f)
              writer.writerow(["*"+nml["output"]["commres"]])
              writer.writerow(["*coordinate system: "+in_nml["input"]["koord_system"]])
              f.writelines(["*u and v in m/s"+"\n", "*ex in kJ/m²"+"\n", "*hx in m\n"])
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
        print("Trajectories were saved as KLAM_21 compatible files.")
      
      if "geojson" in out_format:                                       # if geojson output wanted
        lines_df = pd.DataFrame({"geometry":[], "Nr":[], "x_start":[], "y_start":[], "t_start":[], "t_end":[], "Überwärmungsgebiet":[], "Quellgebiet":[]}) 
        points_df = pd.DataFrame()                                      # create empty data frames for output as lines and as points
        geom = []
        for i in range(0, len(traj_list)):   # loop over all calculated trajectories
          # get data frame from traj_list
          trajek = traj_list[i].copy()
          trajek["lahm"] = trajek.ws < ws_th      # find times where wind speed is smaller than the wind speed treshold
          trajek["summe"] = trajek.dt             
          # cumulative sum of time steps where wind speed is less than the treshold
          trajek["summe"] = trajek["summe"].sub(trajek["summe"].mask(trajek.lahm).ffill().fillna(0)).astype(int) 
          trajek["lahmer"] = trajek.ws.diff() < 0    # find decreasing wind speeds
          trajek["summe1"] = trajek.summe
          # cumulative sum of time steps where wind speed is less than treshold and decreasing
          trajek["summe1"] = trajek["summe1"].sub(trajek["summe1"].mask(trajek.lahmer).ffill().fillna(0)).astype(int)
          # cut trajectory if th_time is met or exceeded by cumulative sum of time steps where decreasing wind speed is less than treshold
          if max(trajek.summe1) >= th_time:
            ind_drop = trajek.loc[trajek.summe1>=th_time,"summe1"].index[0]
            trajekt = trajek.loc[:ind_drop,:"ws"]
            trajekt.loc[ind_drop,["u","v","ws","ex","hx","Zeitschritt"]] = [-1111,-1111,-11.11,-111.1,-111.1,-11.11]
          else: 
            trajekt = trajek.loc[:,:"ws"]
          # create data frame with trajectory starting point, values around every dt seconds and last calculated position (or last position before trajectory was cut 
          # because of wind speed)
          test = pd.DataFrame(trajekt.loc[trajekt.index==0])
          if Startzeit[i] < Endzeit[i]:
            for dti in range(dt+int(test.loc[0,"Zeit"]),int(test.loc[0,"Zeit"])+int(max(trajekt.dt)),dt):
              test = pd.concat([test, trajekt.loc[np.argsort(np.argsort(round(abs(trajekt['Zeit']-dti))))==0]])
          else:
            for dti in range(int(test.loc[0,"Zeit"])-dt,int(test.loc[0,"Zeit"])-int(max(trajekt.dt)),-dt):
              test = pd.concat([test, trajekt.loc[np.argsort(np.argsort(round(abs(trajekt['Zeit']-dti))))==0]])
          test = pd.concat([test,  trajekt.loc[trajekt.index==max(trajekt.index)]], ignore_index=True)
          test = test.drop_duplicates()                # if there are duplicate rows the duplicate is dropped.
          
          # spatial join of warm target areas with every output position, if wanted
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
          
          # spatial join of cold air production areas with temporally first position of trajectory (written for every output position)
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
        #   file.write(lines_gdf.to_json())                                     # alternative for saving
        lines_gdf.to_file(out_file_name+"_LS.geojson")                          # save trajectories as Linestrings
        points_df["x"] = points_df.x+xlu        # convert to absolute coordinates
        points_df["y"] = points_df.y+ylu        # convert to absolute coordinates
        points_df["u"] = points_df.u/100        # convert to m s⁻¹
        points_df["v"] = points_df.v/100        # convert to m s⁻¹
        points_df["ex"] = points_df.ex/10       # convert to kJ/m
        points_df["hx"] = points_df.hx/10       # convert to m
        points_df["geometry"] = points_df[["x", "y"]].apply(Point, axis=1)
        points_df.loc[:,["Quellgebiet","Überwärmungsgebiet"]] = points_df.loc[:,["Quellgebiet","Überwärmungsgebiet"]].fillna("ob")
        points_df.fillna(-99.99, inplace=True)
        points_df = points_df.round(2)          # round to 2 decimal points
        points_gdf = gpd.GeoDataFrame(points_df[["Nr", "x", "y", "Zeit","u", "v","ws", "ex", "hx", "Zeitschritt", "LU", "Überwärmungsgebiet", "Quellgebiet", "geometry"]], crs=in_nml["input"]["koord_system"], geometry="geometry")
        points_gdf.to_file(out_file_name+"_P.geojson")                                                                                                # save as Points
        # with open(directory+in_nml["output"]["out_dir"]+out_file+str(start_ein)+"_"+str(end_ein)+"_LU"+str(lu_start)+"_P.geojson", "w") as file:    # alternative way
        #   file.write(points_gdf.to_json())                                                                                                          # to save as geojson
        print("Trajectories saved as geoJSON LineStrings and Points.")
  
      print("Saving took so long: ", datetime.now()-now)

print("total duration: ",datetime.now()-now1)
