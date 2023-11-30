#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 10:52:06 2023

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

from scipy.spatial.distance import cdist

import itertools as it




#Here's a basic walkthrough for image processing and data analysis of a smaple
#foam.  It's full of suboptimal pieces, but that's what you're here for!


#Phase the first

#Load the figure and truncating it to a reasonable region
img4 = cv2.imread("dsc00295.jpg",0)

img4 = img4[500:2500,3000:5000]

#img4 = img4[1300:1800,4100:4500]

#img4 = img4[1700:2000,4000:4200]




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



#Here we try including the diagonal entries as well


# def nbors(markers, rad):
#     l = np.max(markers)
#     li = np.shape(markers)[0]
#     lj = np.shape(markers)[1]
#     nbortrix = np.zeros(shape = (l, l))
#     for i in range(0,li):
#         for j in range(0,lj):
#             # print([i,j])
#             #Go through up and down cases 
#             if markers[i,j] == 0:
#                 pp = set([  markers[ min(i+rad,li-1),j   ],  markers[   i,min(j+rad, lj-1)   ] , markers[   max(0,i-rad),j   ] , markers[   i,max(0,j-rad)  ] ])
#                 pp = [q for q in pp if q != 0]
#                 if pp != []:
#                     for qq in it.combinations(pp,2):
#                         nbortrix[qq[0]-1,qq[1]-1] = 1
#                         nbortrix[qq[1]-1,qq[0]-1] = 1


#     return(nbortrix)



#Now a method that starts on border pixels


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
            

# #area histogram
# plt.hist(statz[:,0], bins= 30 )   

# #perimeter histogram
# plt.hist(statz[:,1], bins= 30 )   

# #side histogram
# plt.hist(statz[:,4])   


#Getting Aboav quantities

aboqty = [ [] for i in  range(20)]



# for i in range(len(regions)):
#     #nside no's o of nbors in cell i
#     zz = [statz[j,4] for j, x in enumerate(admat[i,:]) if x== 1]
#     aboqty[int(statz[i,4])] += zz
    
    
#What are the boundary cells?
bcells = np.unique(     list(markers[0,:])+ list(markers[:,0]) +list(markers[-1,:])+list(markers[-1,:])     )
bcells = np.delete(bcells, 0)


#average cell sides...ought to be close to 6 for interior cells

np.mean( [degrees[i] for i in range(len(degrees)) if i not in bcells])


# print(aboqty)
mn = [np.mean(i) for i in aboqty]

# print(mn)

#Time to plot Aboavivity...or whatever it's called

# p = range(2,10)
# q = [ pp*mn[pp] for pp in p]
# plt.plot( p,q)


#Plugging in numbers

degrees = [int(s) for s in degrees]
for s in degrees:
    nSidesMatrix[k,s] = nSidesMatrix[k,s] + 1


# #At this point, we'd like to put a labeling on an image.


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





    

    # print(np.mean(degrees))
    
    
bcells = np.unique(     list(markers[0,:])+ list(markers[:,0]) +list(markers[-1,:])+list(markers[-1,:])     )
bcells = np.delete(bcells, 0)


#average cell sides...ought to be close to 6 for interior cells

print("bordercell avg is")
print(np.mean( [degrees[i] for i in range(len(degrees)) if i not in bcells]))




