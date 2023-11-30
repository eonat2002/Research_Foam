#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 13:23:29 2023

@author: elifonat
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 12:55:55 2023

@author: elifonat
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 12:09:38 2023

@author: elifonat
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 13:49:04 2023

@author: r92830873
"""


import cv2

import numpy as np

import random

import matplotlib.pyplot as pyplot

from matplotlib import pyplot as plt

from scipy import ndimage

from skimage import io, color, measure, feature

import scipy.stats as st


#Here's a basic walkthrough for image processing and data analysis of a smaple
#foam.  It's full of suboptimal pieces, but that's what you're here for!


#Phase the first

#Load the figure and truncating it to a reasonable region
statsVector = []
perimeterVector = []
nSidesMatrix = np.zeros(shape = (400,400))

for k in range(18,19):
    image = "sample"+ str(k)+".JPG"
    print(image)
    img4 = cv2.imread(image,0)
    
    #img4 = img4[500:3400,1500:4500]
    img4 = img4[1500:2000,4000:4500]
    
    # img4 = img4[1700:2000,4000:4200]
    
    

    
    #Note you need to get your pixel scale correct
    
    #Phase the second, denoising and thresholding
    
    # plt.hist(img2.flat, bins = 100, range = (0,255))
    
    #This is the adaptive thresholding step.  Take a look at the documentation 
    #and get an idea what each piece means.
    
    thresh = cv2.adaptiveThreshold(img4, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 151, 3.5)
    
    
    
    
    #Phase the third, binarizing
    
    # mask = thresh == 255
    
    # maskcut = 1-mask
    # maskcut =  maskcut.astype('uint8')
    
    #This is the kernel applied to each of the IP operations.  
    kernel = np.ones((3,3),np.uint8)
    
    #Here we're performing on "opening" operation
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 1)
    
    
    #And now a dilation: this has to be applied to the inverse of the image, meaning a 0 becomes a 1, 
    #and vise versa
    
    
    
    sure_bg = cv2.dilate(255-opening, kernel, iterations = 2)
    sure_bg = 255-sure_bg
    plt.imshow(255-sure_bg)
    
    
    #Now a distance transform
    dist_transform = cv2.distanceTransform(sure_bg, cv2.DIST_L2, 3)
    
    ret2, sure_fg = cv2.threshold(dist_transform, .001*dist_transform.max(), 255, 0)
    
    
    #What does it look like so far?  This gives a bigger picture outside of spyder.  Just click on the picture
    #and hit any key to close
    
    sure_fg = np.uint8(sure_fg)
    # cv2.imshow("Sure rg", 255-sure_fg)
    # cv2.waitKey(0)
    
    
    
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    sure_fg = np.uint8(sure_bg)
    # cv2.imshow("SUnknown", unknown)
    # cv2.waitKey(0)
    
    
    #This is using connected components to isolate individual cells.
    
    ret, markers = cv2.connectedComponents(sure_bg)
    
    
    
    markers[unknown == 255] = -1
    
    
    #What does this do?
    # img4[markers == -1] = [0,255,255]
    
    
    
    
    
    # # For big visualization
    # cv2.imshow("colored grains", img6)
    # cv2.waitKey(0)
    
    # strats = cv2.connectedComponentsWithStats(sure_fg)
    
    
    
    
    #markers = cv2.watershed(img4, markers)
    
    
    
    
    
    
    
    # cv2.destroyAllWindows()
    # cv2.waitKey(1)
    # cv2.waitKey(1)
    # cv2.waitKey(1)
    # cv2.waitKey(1)
    
    
    
    
    
    #Now it's off to the analysis!  We need to build an adjacency matrix and 
    #also retrieve statistics for perimeter and area which are provided by python packages.
    
    
    
    # #Nearest neighbors looking at border cases
    #this connect components marker seems to be awful, practice caution here...
    #ret, markers = cv2.connectedComponents(opening)
    
    
        
        
    def get_adjacent_indices(i, j, markers, rad):
        m = np.shape(markers)[0]
        n = np.shape(markers)[1]
        adjacent_indices = []
        if i > rad:
            adjacent_indices.append(markers[i-rad,j])
            if j> rad:
                adjacent_indices.append(markers[i-rad,j-rad])
        if i < m-rad:
            adjacent_indices.append(markers[i+rad,j])
            if j < n-rad:
                adjacent_indices.append(markers[i+rad,j+1])
        if j > rad:
            adjacent_indices.append(markers[i,j-rad])
            if i < m-rad:
                adjacent_indices.append(markers[i+rad,j-rad])
        if j < n-rad:
            adjacent_indices.append(markers[i,j+rad])
            if i > rad:
                adjacent_indices.append(markers[i-rad,j+rad])
        return set(adjacent_indices)
    
    
    
    
    
    
    def nbors(markers, rad):
    
        l = np.max(markers)
        li = np.shape(markers)[0]
        lj = np.shape(markers)[1]
        nbortrix = np.zeros(shape = (l, l))
        
    
            #horizontal scan
        for ss in range(li):
            QQ = markers[ss,:]
            zz = [ i for i in range(1,lj-1) if (QQ[i] != QQ[i+1] or QQ[i] != QQ[i-1]) and QQ[i] != 0]
        
            for rr in zz:
                v = markers[ss,rr]
                l = get_adjacent_indices(ss, rr, markers, rad)
                ll = [a for a in l if a!= 0 and a!= v]
                for ww  in ll:
                    nbortrix[ww-1,v-1] = 1
                    nbortrix[v-1,ww-1] = 1
        for ss in range(lj):
            QQ = markers[:,ss]
            zz = [ i for i in range(1,li-1) if (QQ[i] != QQ[i+1] or QQ[i] != QQ[i-1]) and QQ[i] != 0]
        
            for rr in zz:
                v = markers[rr,ss]
                l = get_adjacent_indices(rr, ss, markers, rad)
                ll = [a for a in l if a!= 0 and a!= v]
                for ww  in ll:
                    nbortrix[ww-1,v-1] = 1
                    nbortrix[v-1,ww-1] = 1
        return nbortrix
    
    
    
    admat = nbors(markers, 15)
    degrees = [sum(admat[i,:]) for i in range(np.max(markers))]

    
    #maskcut = cv2.dilate(maskcut, kernel, iterations = 1)
    #mask = cv2.erode(maskcut, kernel, iterations = 1)
    # # plt.imshow(maskcut)
    
    
    # mask = 1-maskcut
    
    
    
    # s=  [ [1,1,1], [1,1,1], [1,1,1]]
    
    # label_mask, num_labels = ndimage.label(opening, structure = s)
    
    # img3 = color.label2rgb(label_mask, bg_label = 0)
    
    # plt.imshow(img3)
    
    # # cv2.imshow("colored labels", img3)
    # # cv2.waitKey(0)
    
    
    
    
    # #Phase the fifth...grain stats
    
    # clusters = measure.regionprops(label_mask,maskcut)
    
    
    # for prop in clusters:
    #     print('Label: {} Area: {}'.format(prop.label, prop.area))
        
        
    # areas = [prop.area for prop in clusters]
    # areas = np.sort(areas)
    # areas = areas[0:-5]
    # plt.hist(areas, bins= 30 )   
    # # plt.xscale('log')
    
    
    # [p for p in areas if p>10]
    
    
    regions = measure.regionprops(markers, intensity_image = img4)
    
    propList = ['Area', 
                'Perimeter',
                'Centroid'
                'Label']
    
    
    
    statz = np.zeros(shape = (len(regions), 6))
    
    
    #The numbering here is not off by 1, it correponds directly to the grain marker number
    for i in range(len(regions)):
        statz[i,0] = regions[i]['Area']
        statz[i,1] = regions[i]['Perimeter']
        statz[i,2] = regions[i]['Centroid'][0]
        statz[i,3] = regions[i]['Centroid'][1]
        #Now use degrees to pluck out neighbors.  Careful! Need to use proper labeling...
        statz[i,4] = degrees[regions[i]['Label']-1]
        statz[i,5] = regions[i]['Label']
    statsVector = statsVector + list(statz[:,1])
    perimeterVector = perimeterVector + [list(statz[:,1])]

          
    # list of lists
    # perimeters of all the snapshots
    # not the entire foam
    # scale to capture 3000 cells
    #for i in range(1,31):
     #image = "sample"+ str(i)+".jpg"
      #  print(image)
       # cropRegion = image.crop(300,300,300,300) # scale image
        #statz[i,1] = regions[i]['Perimeter']
    
    
    
    
    
    #area histogram
    plt.hist(statz[:,0], bins= 30 )   
    
    
    #perimeter histogram
    plt.hist(statz[:,1], bins= 30 )   
    
    #side histogram
    plt.hist(statz[:,4])   
    
    
    
    markers3 = np.matrix.copy(markers)
    for ii in range(np.shape(markers3)[0]):
         for jj in range(np.shape(markers3)[1]):
             if markers[ii,jj] == 0:
                     markers3[ii,jj] = 20
             else:
                     markers3[ii,jj] = degrees[markers[ii,jj]-1]
             
    degrees = [int(s) for s in degrees]
    for s in degrees:
       nSidesMatrix[k,s] = nSidesMatrix[k,s] + 1
        
    totalCells = 0
    for cell in perimeterVector:
        totalCells = totalCells + len(cell)
    
    
    #go through matrix, make a list of all the elements with sides
 #   elementsList = []
  #  for row in nSidesMatrix:
   #     for column in nSidesMatrix:
    #        if row == k and column == s:
     #           elementsList = elementsList.append(nSidesMatrix[k,s])
    
    nSidesMatrix[k,:] = nSidesMatrix[k,:] / sum(nSidesMatrix[k,:])
    
    # #At this point, we'd like to put a labeling on an image.
    colorVector = []
    p = np.array([0,0,200])
    for c in range(1,20):
         colorVector = colorVector + [c*10 + p[0],p[1],p[2] - c*10]
    
    num_shades = 30
    red_value = 255
    shades_of_red = np.zeros((num_shades, 3), dtype=np.uint8)
    shades_of_red[:, 0] = red_value
    shades_of_red[:, 1:] = np.linspace(255, 0, num_shades)[:, np.newaxis]
    plt.imshow([shades_of_red])
    plt.axis('off')
    plt.show()
    shades_of_red = (1/255) * shades_of_red
    shades_of_red = [list(q) for q in shades_of_red]
    
    
 
   
    img6 = color.label2rgb(markers3, bg_label = 0, bg_color = 'blue',colors = shades_of_red, kind = 'overlay')
    fig, ax = plt.subplots()
    ax.imshow(img6)
    fig, ax = plt.subplots()
    ax.imshow(img6)
  
   
    
  #   # shift the label
  #   # play around with font
  #   scale = 20.3/4000
  #   image =  plt.imshow(markers3,cmap = 'Reds')
  #   image.set_extent(np.array(image.get_extent())*scale)
  #   ax.autoscale(False)
   
  #   for qq in range(len(statz)):
  #        ax.text(statz[qq,3]*scale, statz[qq,2]*scale, int(statz[qq,4]), horizontalalignment = 'center',family = 'serif',weight = 'bold',color = 'black')
    
  #   fig.canvas.draw()
  #  #fig.set_facecolor('black')
  #    # Create a colorbar with shades of red using the 'Reds' colormap
  #   colorbar = plt.colorbar(image, location = 'right')
    
  # #   # Customize the colorbar
  #   colorbar.set_label('number of sides')
  #   plt.clim(min(degrees),max(degrees))
  # # #  plt.legend()
  #    # Display the plot
  #   plt.xlabel('cm')
  #   plt.ylabel('cm')
  #   plt.title("Foam of Cells and their Number of Sides",loc = 'center',fontweight ='bold',fontsize = 10,fontstyle='italic')
    

   # x= 100,200,300,400,500
    #y = 100,200,300,400,500
    #scale = 20.3/4000
    
   #  ax.autoscale_view()
   
   #  plt.show()
   #  pyplot.ion()

   #  x = [scale * value for value in x]
   #  y = [scale * value for value in y]
    
   # # line = pyplot.plot(x,y)       
   # # ax= pyplot.gca()
   # # ax.relim()
   # # pyplot.draw()
            
            
    #Run this code snippet to kill images   
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1) 
    markers1 = np.ones_like(markers)
    shiftedMarkers = np.add(markers,markers1)
    
    #edge1 = np.unique(shiftedMarkers[0,:])
    #print(edge1)

sidesMatrix = np.load('sidesMatrix.npy',allow_pickle=True)
for m in range(3,31):
     plt.title('Statistical Topologies')
     plt.xlabel('Number of n-gons')
     plt.ylabel('Ratio of n-gons to total cells')
     plt.xlim(0,20)
     plt.plot(sidesMatrix[m,:],alpha = 0.4)

# #np.save('sidesMatrix',nSidesMatrix)
column = []
for col in range(np.shape(sidesMatrix)[1]):
     colSum = np.sum(sidesMatrix[:,col])
     column.append(colSum)
     rows= []
     for row in range(np.shape(sidesMatrix)[0]):
         rowSum = np.sum(sidesMatrix[row,:])
         rows.append(rowSum)
        
averageSides = column / sum(rows)
plt.plot(averageSides,label = 'average sides for a cell',linewidth = 2,color = 'k')
plt.legend()
plt.show()


