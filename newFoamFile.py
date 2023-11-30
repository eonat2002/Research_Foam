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

from matplotlib import pyplot as plt

from scipy import ndimage

from skimage import io, color, measure, feature

import scipy.stats as st



#Here's a basic walkthrough for image processing and data analysis of a smaple
#foam.  It's full of suboptimal pieces, but that's what you're here for!


#Phase the first

#Load the figure and truncating it to a reasonable region
perimeterVector = []
areaVector = []
sidesVector = []
for k in range(15,16):
    image = "sample"+ str(k)+".JPG"
    print(image)
    img4 = cv2.imread(image,0)
    
    img4 = img4[500:2500,3000:5000]
    
    #img4 = img4[1500:3000,3500:4500]
    
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
    
    
    # maskcut = cv2.dilate(maskcut, kernel, iterations = 1)
    # # mask = cv2.erode(maskcut, kernel, iterations = 1)
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
    perimeterVector = perimeterVector + [list(statz[:,1])]
    areaVector = areaVector + [list(statz[:,0])] 
    sidesVector = sidesVector + [list(statz[:,4])]
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
    plt.hist(areaVector[0], bins= 30 )   
    
    
    #perimeter histogram
    plt.hist(perimeterVector[0], bins= 30 )   
    
    #side histogram
    plt.hist(sidesVector[0])   
    
    totalPerimeter = 0
    for element in perimeterVector:
       totalPerimeter =  totalPerimeter + sum(element)
    #Plugging in numbers
    
    totalCells = 0
    for cell in perimeterVector:
        totalCells = totalCells + len(cell)
    
    sampleMean = totalPerimeter / totalCells
    
    perimeterList = []
    for item in perimeterVector:
        for subitem in item:
            perimeterList.append(subitem)
    
    sidesList = []
    for side in sidesVector:
        for subSide in side:
            sidesList.append(subSide)
            
    areaList = []
    for area in areaVector:
        for subArea in area:
            areaList.append(subArea)
    #create 95% confidence interval
    st.t.interval(alpha = 0.95, df = len(perimeterList)-1,loc = np.mean(perimeterList), 
                  scale = st.sem(perimeterList))
    
    # find all cells with 3 sides 

    # areaPerimRatio = []
    # areaThree = [areaList[f] for f in range(len(sidesList)) if sidesList[f] == 3.0]
    # print(areaThree)
    # perimThree = [perimeterList[g] for g in range(len(sidesList)) if sidesList[g] == 3.0]
    # for i in range(len(areaThree)):
    #     if areaThree[i] != 0.0 and perimThree[i] != 0.0:
    #         areaPerimRatio.append(areaThree[i] / perimThree[i])
    
    # averageRatio = sum(areaPerimRatio) / len(areaPerimRatio)
    # print(averageRatio)
     
    # # 4 sided cells
    # areaPerimRatio1 = []
    # areaFour = [areaList[f] for f in range(len(sidesList)) if sidesList[f] == 4.0]
    # perimFour = [perimeterList[g] for g in range(len(sidesList)) if sidesList[g] == 4.0]
    # for i in range(len(areaFour)):
    #     if areaFour[i] != 0.0 and perimFour[i] != 0.0:
    #         areaPerimRatio1.append(areaFour[i] / perimFour[i])
    
    # averageRatio1 = sum(areaPerimRatio1) / len(areaPerimRatio1)
    # print(averageRatio1)
    
    # # 5 sided cells 
    # areaPerimRatio2 = []
    # areaFive = [areaList[f] for f in range(len(sidesList)) if sidesList[f] == 5.0]
    # perimFive = [perimeterList[g] for g in range(len(sidesList)) if sidesList[g] == 5.0]
    # for i in range(len(areaFive)):
    #     if areaFive[i] != 0.0 and perimFive[i] != 0.0:
    #         areaPerimRatio2.append(areaFive[i] / perimFive[i])
    
    # averageRatio2 = sum(areaPerimRatio2) / len(areaPerimRatio2)
    # print(averageRatio2)
    
    # # 6 sided cells 
    # areaPerimRatio3 = []
    # areaSix = [areaList[f] for f in range(len(sidesList)) if sidesList[f] == 6.0]
    # perimSix = [perimeterList[g] for g in range(len(sidesList)) if sidesList[g] == 6.0]
    # for i in range(len(areaSix)):
    #     if areaSix[i] != 0.0 and perimSix[i] != 0.0:
    #         areaPerimRatio3.append(areaSix[i] / perimSix[i])
    
    # averageRatio3 = sum(areaPerimRatio3) / len(areaPerimRatio3)
    # print(averageRatio3)
    
    # # 7 sided cells 
    # areaPerimRatio4 = []
    # areaSeven = [areaList[f] for f in range(len(sidesList)) if sidesList[f] == 7.0]
    # perimSeven = [perimeterList[g] for g in range(len(sidesList)) if sidesList[g] == 7.0]
    # for i in range(len(areaSeven)):
    #     if areaSeven[i] != 0.0 and perimSeven[i] != 0.0:
    #         areaPerimRatio4.append(areaSeven[i] / perimSeven[i])
    
    # averageRatio4 = sum(areaPerimRatio4) / len(areaPerimRatio4)
    # print(averageRatio4)
    
    # # 8 sided cells 
    # areaPerimRatio5 = []
    # areaEight = [areaList[f] for f in range(len(sidesList)) if sidesList[f] == 8.0]
    # perimEight = [perimeterList[g] for g in range(len(sidesList)) if sidesList[g] == 8.0]
    # for i in range(len(areaEight)):
    #     if areaEight[i] != 0.0 and perimEight[i] != 0.0:
    #         areaPerimRatio5.append(areaEight[i] / perimEight[i])
    
    # averageRatio5 = sum(areaPerimRatio5) / len(areaPerimRatio5)
    # print(averageRatio5)
    
    # # 9 sided cells
    # areaPerimRatio6 = []
    # areaNine = [areaList[f] for f in range(len(sidesList)) if sidesList[f] == 9.0]
    # perimNine = [perimeterList[g] for g in range(len(sidesList)) if sidesList[g] == 9.0]
    # for i in range(len(areaNine)):
    #     if areaNine[i] != 0.0 and perimNine[i] != 0.0:
    #         areaPerimRatio6.append(areaNine[i] / perimNine[i])
    
            
    
    
    degrees = [int(s) for s in degrees]
    areaList = list(areaVector[0])
    perimeterList = list(perimeterVector[0])
    perimArea = []
    overallRatio = []
    for m in range(3,20):
        area = [areaList[f] for f in range(len(degrees)) if degrees[f] == m]
        perim = [perimeterList[g] for g in range(len(degrees)) if degrees[g] == m]
        perimArea = []
        for i in range(len(area)):
            if area[i] != 0 and perim[i] != 0:
                perimArea.append(area[i] / (perim[i])**2)
       
        ratio = sum(perimArea) / len(perimArea)
        overallRatio.append(ratio)
  
  
    x_axis = range(3,20)
    #y_axis = overallRatio   
    #n = [3,4,5,6,7,8,9,10]
    RATIO = []
    for n in range(3,20):
        rat = (1/(np.tan(np.pi/n)) * (1/(4*n)))
        RATIO.append(rat)
        
    plt.plot(x_axis,RATIO)
    plt.axhline(y = 1/(4*np.pi), color = 'r')
    plt.plot(x_axis,overallRatio)
    
    #At this point, we'd like to put a labeling on an image.    
    img6 = color.label2rgb(markers, bg_label = 0)
    fig, ax = plt.subplots()
    ax.imshow(img6)
    fig, ax = plt.subplots()
    ax.imshow(img6)
    
    for qq in range(len(statz)):
        ax.text(statz[qq,3], statz[qq,2], str(qq))
            
    
    
            
            
    #Run this code snippet to kill images   
      
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)
    
    markers1 = np.ones_like(markers)
    shiftedMarkers = np.add(markers,markers1)
    
    edge1 = np.unique(shiftedMarkers[0,:])
    print(edge1)
# for l in range(1,3):
#     plt.plot(x_axis, y_axis)
#     plt.xlabel('N-gons')
#     plt.ylabel('average ratio of area and perimeter')
#     plt.title('number of sides VS ratio of A/P')
#     plt.xticks(np.arange(3,10,1))
#     plt.yticks(np.arange(min(y_axis),max(y_axis), 1))
#     plt.show()