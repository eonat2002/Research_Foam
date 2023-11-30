#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 11:04:36 2023

@author: elifonat
"""

import cv2

import numpy as np

from matplotlib import pyplot as plt

from scipy import ndimage

from skimage import io, color, measure, feature




#Here's a basic walkthrough for image processing and data analysis of a smaple
#foam.  It's full of suboptimal pieces, but that's what you're here for!


#Phase the first

#Load the figure and truncating it to a reasonable region
statsVector = []
for k in range(1,2):
    image = "sample"+ str(k)+".JPG"
    print(image)
    img4 = cv2.imread(image,0)
    
    #img4 = img4[500:2500,3000:5000]
    
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
    
    
    
    def nbors(markers, rad):
        l = np.max(markers)
        li = np.shape(markers)[0]
        lj = np.shape(markers)[1]
        nbortrix = np.zeros(shape = (l, l))
        for i in range(0,li):
            for j in range(0,lj):
                # print([i,j])
                #Go through up and down cases 
                if markers[i,j] == 0:
                    a = markers[ min(i+rad,li-1),j   ]
                    b = markers[   i,min(j+rad, lj-1)   ]
                    c = markers[   max(0,i-rad),j   ]
                    d = markers[   i,max(0,j-rad)  ]
                    if ((a != c) and (a!= 0) and (c != 0))  :                    
                        nbortrix[a-1,c-1] = 1
                        nbortrix[c-1,a-1] = 1
                    if ((b != d) and (b!= 0) and (d != 0))  :
                        nbortrix[b-1,d-1] = 1
                        nbortrix[d-1,b-1] = 1
    
        return(nbortrix)
                    
    
    admat = nbors(markers,5)
    print(admat)
    
    
    degrees = [sum(admat[i,:]) for i in range(np.max(markers))]
    print(degrees)
    
    
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
    statsVector = statsVector + list(statz[:,1])
          
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
    
    
    #Plugging in numbers
    
    
    
    # #At this point, we'd like to put a labeling on an image.
    
    
    img6 = color.label2rgb(degrees, bg_label = 0)
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