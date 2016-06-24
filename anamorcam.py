# -*- coding: utf-8 -*-

import time
import PIL.Image as Image
import numpy as np

import cv2

#takes an RGB image  file xxx.jpg
#and writes anaglyph.jpg

#2 choices for the transform 
#- conical_anamorphosis
#- cylindrical_anamorphosis



def cylindrical_anamorphosis(size_src=None,
                            dpi=72,
                            r_in=3,
                           dump2fn=None):
   
   sx,sy = size_src

   # calculates the radius in pixels
   r_cylinder = int(r_in*dpi)
   # sets theta to be between 0 and pi (maybe user should be able to set this?)
   pi = np.pi
   # sets the maximum radius (in pixels) as the radius of the cylinder
   #plus the height of the original image.  Maybe the user should be able to set this,
   #but then we'd have to interpolate to resize the image.
   max_r = r_cylinder + sy
   
   # the final image has dimensions 2*max_r x max_r,
   # because this is the widest that a circle with radius max_r will be
   XS,YS = np.mgrid[0:2*max_r,0:max_r]
   
   #print max_r, 2*max_r
   
   #flatten the component arrays for easier manipulation
   #and normalize so the the center is at the middle of an edge
   XS = max_r - XS.ravel()  
   YS = max_r - YS.ravel()
   
   #calulate the inverse image map
   RS = np.sqrt(XS**2 + YS**2)
   TS = np.arctan2(YS,XS)/pi
   #need floor here otherwise go out of bounds
   X0 = np.floor(sx*TS)
   Y0 = np.floor(RS - r_cylinder)
   
   #make a boolean mask to hide stuff
   MSK = (r_cylinder < RS) & (RS < max_r) & (TS < 1) 
   MSK = np.invert( MSK)
   
   #set anything in the that shouldn't be seen to the value at 0,0
   X0[MSK] = 0
   Y0[MSK] = 0
   
   #should I cast to integers before doing the expression ???
   mapping = (sx*Y0 + X0).astype(int)
   
   if dump2fn:
      fp = file(dump2fn,'wb')
      np.save(fp, mapping.reshape(2*max_r,max_r))
   return mapping,2*max_r,max_r
   

def conical_anamorphosis(size_src=None,
                         base=6.76, height=7.89 ):
   '''Calcule une anamorphose conique.


   src -- size of image
   base -- le diamètre de la base du cone (cm)
   height -- la hauteur du cone (cm)

   Retourne :
   flattened numpy array of integers
   '''
   # angle au sommet du cône
   angle = 2*np.tan(base/(2*height))
   # rayon disque sous le cône
   r1 = base/2
   # rayon disque réfléchi
   r2 = height*np.tan(angle)
   print("Image finale imprimée de taille %.2f cm"%(2*r2))

   sx,sy = size_src 
   XS,YS = np.mgrid[0:sx,0:sy]
    
   #flatten the component arrays for easier manipulation
   #and normalize so the the center is at the middle of an edge
   X1 = XS.ravel() - sx//2
   Y1 = YS.ravel() - sy//2
    
    #calulate the inverse image map
   RS = np.sqrt(X1**2 + Y1**2)*(r2/(sx//2))
   TS = np.arctan2(Y1,X1)

   #need floor here otherwise go out of bounds
   RR = sx//2 + ( sx/2/(r2 - r1) )*(r1 - RS)
   X0 = np.floor(RR*np.cos(TS) + sx//2)
   Y0 = np.floor(RR*np.sin(TS) + sy//2)
   
   #make a boolean mask to hide stuff outside the annulus
   MSK = ( RS < r1) | (RS > r2) 

   #set anything in the that shouldn't be seen to the value at (0,0)
   X0[MSK] = 0
   Y0[MSK] = 0
   
   return (sx*Y0 + X0).astype(int),sx,sy


def transform_img(img=None,
                  transform=None,
                  default_color=(255,255,255)):


   img.putpixel((0,0), default_color)
   SRC = np.asarray(img)
   mapping,ww,hh = transform(pic.size)
   
   TGT = np.zeros( (ww,hh, 3),
                    dtype=np.uint8)
   
   ##loop over the color channels
   for rgb_channel in range(3):
       ##used to do this with a take()
       ##but fancy indexing just as quick
       TT = SRC[:,:,rgb_channel].ravel()[mapping]
       TGT[:,:,rgb_channel] = TT.reshape((ww,hh))
       
   return Image.fromarray(TGT)

class Cache_Transform(object):
   '''class that caches a mapping
   to be applied in a video loop
   the object implements a __call__ method'''
   
   def __init__(self,transform=None,
                     size_src=None):
      
      self.mapping,self.ww,self.hh = transform(size_src)
      
   def __call__(self,size_src=None):
      return self.mapping,self.ww,self.hh
      
               
if __name__ == '__main__' :

   webcam = cv2.VideoCapture(0)
   cv2.namedWindow("anamorphose")

   rval = True
   while rval:
      rval, frame = webcam.read()
      cv2.imshow("anamorphose", frame)
      pic=Image.fromarray(cv2.cvtColor(frame,
                             cv2.COLOR_BGR2RGB))
      sx,sy = pic.size
      sz = max(pic.size)
      key = cv2.waitKey(20)
      if key in [27, ord('Q'), ord('q')]: # exit on ESC or q
          break
         

   anamorphosis = Cache_Transform(cylindrical_anamorphosis,(sz/2,sz/2))
   
   while rval:
      rval, frame = webcam.read()
      pic=Image.fromarray(cv2.cvtColor(frame,
                                       cv2.COLOR_BGR2RGB))
      
      pic = pic.resize((sz/2,sz/2)).transpose(Image.ROTATE_180)
     
      yy = transform_img(img=pic,
                        transform=anamorphosis)

      yy = yy.transpose(Image.ROTATE_90)
      cv2.imshow("anamorphose",
                 cv2.cvtColor(np.array(yy),
                 cv2.COLOR_RGB2BGR))
      
      key = cv2.waitKey(20)
      if key in [27, ord('Q'), ord('q')]: # exit on ESC or q
          break

     

