#!/usr/bin/env python
#
# This code has been written by Ali Asghar HAKAMI ZANJANI at SDU, Odense, Oct. 2020.
#
# This code is free, and you can redistribute it or modify it for your research purpose.
# If you find this code useful for your research please cite [Zanjani et al.; Biophysical Journal; 2023] if at all possible.
#
# In this code two-dimensional mean curvature profiles of the surfaces fitted on the upper and
# lower leaflets, and the surface passing through the center of a bilayer membrane,
# are calculated, and their time average is plotted as the output. Time serise of the dimensions of
# the simulation box, area of the fitted surface and projected surface on xy plane are written in log files.
#
# HOW TO USE THIS CODE:
# Using this code is very simple; You just need a structure (.gro), a trajectory (.xtc), and an index (.ndx) file.
#
# Make sure that the specified groups as the upper and lower leaflets, don't pass
# the z boundaries of the main periodic box during the trajectory.
#
# If there is a protein (or any other group) on the membrane and you are going to calculate
# the spatial average of the mean curvature of the membrane inside the area of a circle surrounding
# the projection of the protein on the membrane, during the time, first center the system about
# that group then use this code.

# Run MemCurv.py as:
# ./MemCurv.py structure.gro trajectory.xtc index.ndx
# (Please substitute "structure", "trajectory" and "index" with the names of your files.)
#
# WARNING: Running this code will delete and overwrite all previously generated .dat or .png files. in the
# current folder.
#
# Press any key to continue or ^C (Control+C) to terminate.
#
# Please select a group for upper monolayer, from the list (the list is created from the given index file)
#
# Please select a group for lowr monolayer, from the list (the list is created from the given index file)
#
# Please select a group for protein (or any other group) for the spatial average of the mean curvature
# beneath that group from the list or insert -1 if you don't want to specify any group:
#
# Please enter Nx: 2
# Please enter Ny: 2
# Nx and Ny are the upper limits of the partial sums for Fourier series in fitting step.
# Nx=Ny=2 gives a reasonable surface fitted on a flat bilayer POPC/POPS membrane.     
#
# Please enter the number of grids in x direction: 200
# Please enter the number of grids in y direction: 200 
# 200 grids in x and y directions, give suitable resolution.
#
# Index of the phosphorus atoms in the upper and lower leaflets can specify the upper and lower
# layers of the membrane.

# Index of different columns in output log.dat files:
#  0: frame number (frame)
#  1: x dimension of bilayer (Lx)
#  2: y dimension of bilayer (Ly)
#  3: z dimension of bilayer (Lz)
#  4: surface projection of bilayer onto xy-plane (Axy = Lx*Ly)
#  5: surface of bilayer (ASur)
#  6: x of protein (Prx)
#  7: y of protein (Pry)
#  8: z of protein (Prz)
#  9: radius of protein (PrR)
# 10: surface projection of protein onto xy-plane (APr = pi*PrR^2)
# 11: surface of bilayer beneath protein (ASurPr)
# 12: average of mean curvature over surface (Hm)
# 13: average of mean curvature over surface beneath protein (HPrm)

from scipy import stats
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import os
import sys
import numpy as np
import MDAnalysis as mda
import matplotlib.pyplot as plt
import scipy.optimize as optimize
################################################################################

def GetGroupList(IndexFile):
# Gets the name of the index file, reads the name of the different groups from the
# index file and returns it as a list of [group name, line in index]. 
   GroupList = []
   with open(IndexFile) as fp:
      line = fp.readline()
      cnt  = 1
      while line:
         if '[' in line:
            item = line.strip("[\n ]")
            GroupList.append((item, cnt))
         line = fp.readline()
         cnt += 1
   GroupList.append(('EndOfFile', cnt))
   fp.close()
   return GroupList
################################################################################

def GetGroupName(GroupList,GroupName):
# Gets the GroupList (output of  GetGroupList) and name of a group and asks user
# to select the id of the group from the list.
   print ("\nPlease select a group for {} from the list:".format(GroupName))
   cnt = 0
   for i in GroupList[: -1]:
      print("{}: {}".format(cnt, i[0]))
      cnt += 1
   return "Index of " + GroupName+': '
################################################################################

def GetGroupIndex(GrCode, GrList, IndexFile):
# Gets the code of a group, group list (output of GetGroupList) and the name of
# the index file and returns the index of all atoms of the group in MDA format. 
   GroupIndex = []
   BeginLine  = GrList[GrCode][1]
   EndLine    = GrList[GrCode+1][1]
   with open(IndexFile) as fp:
      line = fp.readline()
      cnt  = 1
      while cnt < EndLine:
         if cnt > BeginLine:
            GroupIndex += [int(i)-1 for i in line.split()]
         line = fp.readline()
         cnt += 1
   fp.close()
   return GroupIndex
################################################################################

def GetGroupIndexMda(GrCode, GrList, IndexFile): # Gets the code of a group and
# group list (output of GetGroupList) and the name of index file and returns the
# index of all atoms of the group in MDA format. 
   GroupIndex = []
   BeginLine  = GrList[GrCode][1]
   EndLine    = GrList[GrCode+1][1]
   with open(IndexFile) as fp:
      line = fp.readline()
      cnt  = 1
      while cnt < EndLine:
         if cnt > BeginLine:
            GroupIndex += [int(i)-1 for i in line.split()]
         line = fp.readline()
         cnt += 1
   fp.close()
   GroupIndexMda = "index"
   for i in range(len(GroupIndex)):
      GroupIndexMda += " "+str(GroupIndex[i])
   return GroupIndexMda
################################################################################

def Zfunc(data, *Amn):
   global Lx, Ly, m_x, n_y
   z = 0
   for m in range (-m_x,m_x+1):
      for n in range (-n_y,n_y+1):
         z += Amn[(m+m_x)*(2*n_y+1)+(n+n_y)] * (np.cos(2*np.pi*m*data[:,0]/Lx+2*np.pi*n*data[:,1]/Ly) + np.sin(2*np.pi*m*data[:,0]/Lx+2*np.pi*n*data[:,1]/Ly))
   return z
################################################################################
################################################################################
################################################################################

gro  = sys.argv[1]
xtc  = sys.argv[2]
ndx  = sys.argv[3]
GrLs = GetGroupList(ndx)

print("\nWARNING: Running this code will delete and overwrite all previously generated png and dat files.")
input("************************************************************************************************\nPress any key to continue or ^C (Control+C) to terminate.")

os.system('rm *.dat')
os.system('rm *.png')

print('\nPlease make sure that the specified groups as the upper and lower leaflets, do not pass the z boundaries of the main periodic box during the trajectory.')
input("\nPress any key to continue ...")
UpLayer  = int(input(GetGroupName(GrLs, 'upper monolayer')))
LowLayer = int(input(GetGroupName(GrLs, 'lower monolayer')))
print('\nIf there is a protein (or any other group) on the membrane and you are going to calculate the spatial average of the mean curvature of the membrane beneath that group (inside the area of a circle surrounding the projection of the protein on the membrane) during the time, you should center the system in xy plane about that group before using this code. Please insert -1 if you do not want to specify any group.')
input("\nPress any key to continue ...")

Protein       = int(input(GetGroupName(GrLs, 'protein')))
UpLayerIndex  = GetGroupIndexMda(UpLayer, GrLs, ndx)
LowLayerIndex = GetGroupIndexMda(LowLayer, GrLs, ndx)
if Protein != -1:
   ProteinIndex = GetGroupIndexMda(Protein, GrLs, ndx)

u = mda.Universe(gro, xtc)

print ("\n{} consists of {} frames.".format(xtc,len(u.trajectory)))

BeginFrame = int(input ("\nPlease enter the begin frame to analysis (0 <= Number <= {}): ".format(len(u.trajectory)-1)))
EndFrame   = int(input ("\nPlease enter the end frame to analysis ({} <= Number <= {}): ".format(BeginFrame,len(u.trajectory)-1)))
m_x        = int(input ("\nPlease enter Nx: "))
n_y        = int(input ("\nPlease enter Ny: "))
Resx       = int(input ("\nPlease enter the number of grids in x direction: "))
Resy       = int(input ("\nPlease enter the number of grids in y direction: "))

Uplay = u.select_atoms(UpLayerIndex)
Lolay = u.select_atoms(LowLayerIndex)
if Protein != -1:
   Prot     = u.select_atoms(ProteinIndex)

X01   = np.linspace(0, 100, Resx+1)
Y01   = np.linspace(0, 100, Resy+1)
X0,Y0 = np.meshgrid(X01, Y01)
ZmU   = 0 * X0
HmU   = 0 * X0
ZmL   = 0 * X0
HmL   = 0 * X0
ZmM   = 0 * X0
HmM   = 0 * X0
num   = 0

logdata = ['#0.frame','#1.Lx','#2.Ly','#3.Lz','#4.Axy','#5.ASur','#6.Prx','#7.Pry','#8.Prz','#9.PrR','#10.APr','#11.ASurPr','#12.Hm','#13.HPrm']

for layname in ['Upper','Lower','Middle']:
   np.savetxt(layname + 'log.dat', np.reshape(logdata,(1,len(logdata))),fmt='%4s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\t%17s\t%17s')

for ts in u.trajectory[BeginFrame:EndFrame+1]:
   Lx     = ts.dimensions[0]/10.0
   Ly     = ts.dimensions[1]/10.0
   Lz     = ts.dimensions[2]/10.0

   UpCoor = Uplay.positions/10.0
   LoCoor = Lolay.positions/10.0
   if Protein != -1:
      PrCoor = Prot.positions/10.0
      Prx    = np.mean(PrCoor[:,0])
      Pry    = np.mean(PrCoor[:,1])
      Prz    = np.mean(PrCoor[:,2])
      PrR2   = np.max((PrCoor[:,0]-Prx)**2+(PrCoor[:,1]-Pry)**2)
      PrR    = np.sqrt(PrR2)
   else:
      Prx    = Lx/2.0
      Pry    = Ly/2.0
      Prz    = 0.0
      PrR2   = 1.0
      PrR    = 1.0
   X1     = np.linspace(0, Lx, Resx+1)
   Y1     = np.linspace(0, Ly, Resy+1)
   X, Y   = np.meshgrid(X1, Y1)
   ZM     = 0 * X
   fxM    = 0 * X
   fxxM   = 0 * X
   fyM    = 0 * X
   fyyM   = 0 * X
   fxyM   = 0 * X
   dx     = Lx/Resx
   dy     = Ly/Resy 
   LgM    = open('Middlelog.dat','ab')

   for [data, layname,Zm,Hm] in [[UpCoor,'Upper',ZmU,HmU], [LoCoor,'Lower',ZmL,HmL]]:
      Par   = open(layname + 'Params.dat','ab')
      Lg    = open(layname + 'log.dat','ab')
      guess = np.zeros((2*m_x+1)*(2*n_y+1))
      guess[int(((2*m_x+1)*(2*n_y+1)-1)/2)] = np.mean(data[:,2])
      params, pcov = optimize.curve_fit(Zfunc, data[:,:2], data[:,2], guess)
      np.savetxt(Par, np.reshape(params,(1,(2*m_x+1)*(2*n_y+1))), fmt='%.8e', delimiter='\t')
      Z   = 0 * X
      fx  = 0 * X
      fxx = 0 * X
      fy  = 0 * X
      fyy = 0 * X
      fxy = 0 * X

      for n in range (-m_x,m_x+1):
         for m in range (-n_y,n_y+1):
            topimlx  = 2*np.pi*m/Lx
            topimlx2 = topimlx**2
            topinly  = 2*np.pi*n/Ly
            topinly2 = topinly**2
            A_ind    = (m+m_x)*(2*n_y+1)+(n+n_y)
            Arg      = topimlx*X+topinly*Y
            CosArg   = np.cos(Arg)
            SinArg   = np.sin(Arg)
            
            Z   += params[A_ind] * (CosArg + SinArg) # Z=f(x,y)       
            fx  += params[A_ind] * topimlx * (CosArg - SinArg) # fx=dZ/dx
            fxx -= params[A_ind] * topimlx2 * (CosArg + SinArg) # fxx=d2Z/dx2
            fy  += params[A_ind] * topinly * (CosArg - SinArg) # fy=dZ/dy
            fyy -= params[A_ind] * topinly2 * (CosArg + SinArg) # fyy=d2Z/dy2
            fxy -= params[A_ind] * topimlx * topinly * (CosArg + SinArg) # fxy=d2Z/dxdy

      H   = -(fxx*(1+fy**2)-2*fxy*fx*fy+fyy*(1+fx**2))/(2*(1+fx**2+fy**2)**1.5) # mean curvature (We choose the unit vector normal to the surface downward whereas the unit vector of the z axis is upward so there is a negative sign in the formula.) 
      Zm += Z
      Hm += H
      HP  = []
      fxP = []
      fyP = []
      for x in range(int((Prx-PrR)/dx), int((Prx+PrR)/dx)+1):
         ylim = PrR**2-(dx*x-Prx)**2
         if ylim < 0:
            ylim = 0
         for y in range(int((Pry-np.sqrt(ylim))/dy),int((Pry+np.sqrt(ylim))/dy)+1):
            HP.append(H[x,y])
            fxP.append(fx[x,y])
            fyP.append(fy[x,y])
            
      fA      = np.sqrt(1+fx**2+fy**2)
      fAP     = np.sqrt(1+np.array(fxP)**2+np.array(fyP)**2)
      ASur    = dx*dy*np.sum(fA)
      ASurPr  = dx*dy*np.sum(fAP)
      
      HsT     = np.sum(H*fA*dx*dy)
      HmT     = 1.0/ASur * HsT
      
      HsPr    = np.sum(np.array(HP)*fAP*dx*dy)
      HmPr    = 1.0/ASurPr * HsPr
      
      logdata = [ts.frame,Lx,Ly,Lz,Lx*Ly,ASur,Prx,Pry,Prz,PrR,np.pi*PrR**2,ASurPr,HmT,HmPr]
      np.savetxt(Lg, np.reshape(logdata,(1,len(logdata))),fmt='%4d\t%10.5f\t%10.5f\t%10.5f\t%10.5f\t%10.5f\t%10.5f\t%10.5f\t%10.5f\t%10.5f\t%10.5f\t%10.5f\t%.12e\t%.12e')
      Par.close()
      Lg.close()

      ZM   += 0.5*Z
      fxM  += 0.5*fx
      fxxM += 0.5*fxx
      fyM  += 0.5*fy
      fyyM += 0.5*fyy
      fxyM += 0.5*fxy
 
   HM   = -(fxxM*(1+fyM**2)-2*fxyM*fxM*fyM+fyyM*(1+fxM**2))/(2*(1+fxM**2+fyM**2)**1.5) # mean curvature
   ZmM += ZM
   HmM += HM
   HPM  = []
   fxPM = []
   fyPM = []
   for x in range(int((Prx-PrR)/dx), int((Prx+PrR)/dx)+1):
      ylim = PrR**2-(dx*x-Prx)**2
      if ylim < 0:
         ylim = 0
      for y in range(int((Pry-np.sqrt(ylim))/dy),int((Pry+np.sqrt(ylim))/dy)+1):
         HPM.append(HM[x,y])
         fxPM.append(fxM[x,y])
         fyPM.append(fyM[x,y])
   fAM      = np.sqrt(1+fxM**2+fyM**2)
   fAPM     = np.sqrt(1+np.array(fxPM)**2+np.array(fyPM)**2)
   ASurM    = dx*dy*np.sum(fAM)
   ASurPrM  = dx*dy*np.sum(fAPM)
   HsTM     = np.sum(HM*fAM*dx*dy)
   HmTM     = 1.0/ASurM * HsTM
   HsPrM    = np.sum(np.array(HPM)*fAPM*dx*dy)
   HmPrM    = 1.0/ASurPrM * HsPrM
   logdataM = [ts.frame,Lx,Ly,Lz,Lx*Ly,ASur,Prx,Pry,Prz,PrR,np.pi*PrR**2,ASurPrM,HmTM,HmPrM]
   np.savetxt(LgM, np.reshape(logdata,(1,len(logdataM))),fmt='%4d\t%10.5f\t%10.5f\t%10.5f\t%10.5f\t%10.5f\t%10.5f\t%10.5f\t%10.5f\t%10.5f\t%10.5f\t%10.5f\t%.12e\t%.12e')
   LgM.close()
   num += 1
   print("frame ",ts.frame)
   
ZmU = ZmU * 1.0/num
HmU = HmU * 1.0/num
ZmL = ZmL * 1.0/num
HmL = HmL * 1.0/num
ZmM = ZmM * 1.0/num
HmM = HmM * 1.0/num

np.savetxt("ZUp.dat",  ZmU, fmt='%.8e')
np.savetxt("HUp.dat",  HmU, fmt='%.8e')
np.savetxt("ZLow.dat", ZmL, fmt='%.8e')
np.savetxt("HLow.dat", HmL, fmt='%.8e')
np.savetxt("ZMid.dat", ZmM, fmt='%.8e')
np.savetxt("HMid.dat", HmM, fmt='%.8e')

#######################################
#################PLOTS#################
#######################################
Thickness = ZmU-ZmL
fsize     = 60
cm        = 'bwr_r'
for [data, name] in [[HmU, 'HmU'], [HmL, 'HmL'],[HmM, 'HmM'],[Thickness, 'Thickness']]:
   fig, ax      = plt.subplots(figsize=(19, 15))
   z_min, z_max = np.min(data), np.max(data)
   c            = ax.pcolormesh(data, cmap=cm, vmin=z_min, vmax=z_max)
   ax.tick_params(labelsize=fsize, size=0.05, pad=0.05)
   cb           = fig.colorbar(c, ax=ax, pad=0.04)
   cb.ax.tick_params(labelsize=fsize, pad=10)
   if name in ['HmU','HmL','HmM']:
      cb.ax.set_ylabel("Mean Curvature (1/nm)", rotation=90, size=fsize, labelpad=20, fontname='FreeSerif')
   else:
      cb.ax.set_ylabel("Membrane Thickness (nm)", rotation=90, size=fsize, labelpad=20,fontname='FreeSerif')
   plt.xticks([0, Resx/4.0, Resx/2.0, 3/4.0*Resx, Resx], (0, round(Lx/4.0,2), round(Lx/2.0,2), round(3*Lx/4.0,2), round(Lx,2))) 
   plt.yticks([0, Resy/4.0, Resy/2.0, 3/4.0*Resy, Resy], (0, round(Ly/4.0,2), round(Ly/2.0,2), round(3*Ly/4.0,2), round(Ly,2)))
   
   plt.xlabel("x (nm)", fontsize=fsize, labelpad=0, fontname='FreeSerif')
   plt.ylabel("y (nm)", fontsize=fsize, labelpad=-15, fontname='FreeSerif')
   
   ax.tick_params(axis='both', which='major', pad=10)

   f = plt.gcf()
   f.set_size_inches(19, 15)
   f.savefig('%s.png'%name,dpi=300,bbox_inches='tight')
