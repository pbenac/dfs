"""
Excerpted and translated from zernike.c

/* zernike.c is a depository for zernike polynomials and their derivatives.
 * There are two types of zernike terms: 1) those for an unobstructed
 * circular pupil which we'll call uZ for "simple" zernike, and 2) those for
 * a centrally obstructed circular pupil cZ.  Their derivatives are duZ/dx,
 * duZ/dy, dcZ/dx, and dcZ/dy. scw: 4-15-99; update 8-17-00 */

 /* NOTE: this convention has theta=0 along the y-axis!  It also appears
  * that theta increases towards the x-axis (clockwise). Adopted from Optical
  * Shop Testing II (why can't they leave well enough alone?). So now,
  * x=rsinth and y=rcosth. */
"""

from numpy import power as pow
import numpy

#monomial zernike expansion

def uZ(n, x, y):

    if n==1:
        
        return (x)		# tilt about y-axis 

    elif n==2:
        return (y)		# tilt about x-axis 

    elif n==3:
        return (2*y*y + 2*x*x -1)	# defocus 

    elif n==4: 
        return (2*x*y)		# astig at +/-45-deg 

    elif n==5: 
        return (y*y - x*x)	# astig at 0/90 deg 

    elif n==6: 
        return (3*x*x*x + 3*x*y*y - 2*x)	# coma along x 

    elif n==7: 
        return (3*y*y*y + 3*y*x*x - 2*y)	# coma along y 

    # 3rd order spherical 
    elif n==8: 
        return (1 - 6*y*y - 6*x*x + 6*y*y*y*y + 12*x*x*y*y + 6*x*x*x*x)
    
    elif n==9: 
        return (3*x*y*y - x*x*x)	# trefoil base on x-axis 
    
    elif n==10: 
        return (y*y*y - 3*x*x*y)	# trefoil base on y-axis 
    
    elif n==11: 
        return (8*x*x*x*y + 8*y*y*y*x - 6*x*y) # 5thast45 	
    
    elif n==12: 
        return (4*y*y*y*y - 4*x*x*x*x + 3*x*x - 3*y*y) # 5thast0 
    
    elif n==13: 
        return (4*y*y*y*x - 4*x*x*x*y) # 4th1 
    
    elif n==14: 
        return (y*y*y*y - 6*x*x*y*y + x*x*x*x) # 4th2 
    
    elif n==15: 
        return (4*x*x*x - 12*x*y*y + 15*x*y*y*y*y + 10*x*x*x*y*y
                - 5*x*x*x*x*x) # hitrefX 
    
    elif n==16: 
        return (12*x*x*y - 4*y*y*y + 5*y*y*y*y*y - 10*x*x*y*y*y
                - 15*x*x*x*x*y) # hitrefY 
    
    elif n==17: 
        return (3*x - 12*x*y*y - 12*pow(x,3) + 10*x*pow(y,4)
                + 20*pow(x,3)*y*y + 10*pow(x,5))	# 5thCX 
    
    elif n==18: 
        return (3*y - 12*y*y*y - 12*x*x*y + 10*pow(y,5)
                + 20*x*x*pow(y,3) + 10*pow(x,4)*y) # 5thCY 
    
    elif n==19: 
        return (20*pow(x,6) + 20*pow(y,6) + 60*x*x*pow(y,4) 
                + 60*pow(x,4)*y*y - 30*pow(x,4) - 30*pow(y,4) - 60*x*x*y*y
                + 12*x*x + 12*y*y - 1) # 6th order spherical 
    
    elif n==20: 
        return (5*x*y*y*y*y - 10*x*x*x*y*y + x*x*x*x*x) # 5th1 
    
    elif n==21: 
        return (y*y*y*y*y - 10*x*x*y*y*y +5*x*x*x*x*y) # 5th2 


"""
/* duZdx() and duZdy() return uZ terms differentiated by x and y for an
 * unobstructed circular pupil.  Both of these routines have been checked
 * with MathCad's phasediff stuff.  NOTE:  phasediff uses 0 origin while all
 * this NR stuff uses 1 origin. These coefficients have UNITY amplitude, and
 * the calling routine must scale them appropriately. */

/* monomial representation of the gradients for an unobstructed circular
 * pupil. */
"""
def duZdx(n, x, y):

    if n==1: 
        return (numpy.ones(x.shape))			# dTy/dx 

    elif n==2: 
        return (numpy.zeros(x.shape))			# dTx/dx 
    
    elif n==3: 
        return (4*x)		# dD/dx 
    
    elif n==4: 
        return (2*y)		# dAst45/dx 
    
    elif n==5: 
        return (-2*x)		# dAst0/dx 
    
    elif n==6: 
        return (9*pow(x,2) + 3*pow(y,2) - 2)	# dCx/dx 
    
    elif n==7: 
        return (6*x*y)		# dCy/dx 
    
    elif n==8: 
        return (24*pow(x,3) + 24*x*pow(y,2) -12*x)  # dS3/dx 
    
    elif n==9: 
        return ((3*pow(y,2) - 3*pow(x,2)))		# dTRFx/dx -- changed y**3 to y**2 BAM 2020-02-27
    elif n==10: 
        return (-6*x*y)	# dTRFy/dx 
    
    elif n==11: 
        return (8*pow(y,3) +24*pow(x,2)*y - 6*y)	# d5ast45/dx 
    
    elif n==12: 
        return (6*x - 16*pow(x,3))	# d5ast0/dx 
    
    elif n==13: 
        return (4*y*y*y - 12*x*x*y) # d4th1/dx 
    
    elif n==14: 
        return (4*x*x*x - 12*x*y*y) # d4th2/dx 
    
    elif n==15: 
        return (12*x*x - 12*y*y + 15*y*y*y*y + 30*x*x*y*y
                - 25*x*x*x*x) # dhiTrX/dx 
    
    elif n==16: 
        return (24*x*y - 20*x*y*y*y - 60*x*x*x*y) # dhiTrY/dx 
    
    elif n==17: 
        return (3 - 12*y*y -36*x*x + 10*pow(y,4) + 60*x*x*y*y
                + 50*pow(x,4)) 	# d5thCX/dx 
    
    elif n==18: 
        return (40*x*y*y*y - 24*x*y + 40*x*x*x*y) # d5thCY/dx 
    
    elif n==19: 
        return (120*pow(x,5) + 120*x*pow(y,4) + 240*x*x*x*y*y
                - 120*x*x*x - 120*x*y*y + 24*x) # dsph6/dx 
    
    elif n==20: 
        return (5*y*y*y*y - 30*x*x*y*y + 5*x*x*x*x) # d5th1/dx 
    
    elif n==21: 
        return (20*x*x*x*y - 20*x*y*y*y)	# d5th2/dx 



def duZdy (n, x, y):

    if n==1: 
        return (numpy.zeros(x.shape))			# dTy/dy 
    
    elif n==2: 
        return (numpy.ones(x.shape))			# dTx/dy 
    
    elif n==3: 
        return (4*y)		# dD/dy 
    
    elif n==4: 
        return (2*x)		# dAst45/dy 
    
    elif n==5: 
        return (2*y)		# dAst0/dy 
    
    elif n==6: 
        return (6*x*y)		# dCx/dy 
    
    elif n==7: 
        return (9*pow(y,2) + 3*pow(x,2) - 2)		# dCy/dy 
    
    elif n==8: 
        return (24*pow(y,3) + 24*pow(x,2)*y - 12*y) # dS3/dy 
    
    elif n==9: 
        return (6*x*y)		#dTRFx/dy 
    
    elif n==10: 
        return (3*pow(y,2) - 3*pow(x,2))		# dTRFy/dy 
    
    elif n==11: 
        return (8*pow(x,3) + 24*pow(y,2)*x - 6*x)	# d5ast45/dy 
    
    elif n==12: 
        return (16*pow(y,3) - 6*y) # d5ast0/dy 
    
    elif n==13: 
        return (12*y*y*x - 4*x*x*x) # d4th1/dy  
    
    elif n==14: 
        return (4*y*y*y - 12*x*x*y) # d4th2/dy 
    
    elif n==15: 
        return (60*x*y*y*y - 24*x*y + 20*x*x*x*y) # dhiTrX/dy 
    
    elif n==16: 
        return (12*x*x - 12*y*y + 25*y*y*y*y - 30*x*x*y*y
                - 15*x*x*x*x) # dhiTrY/dy 
    
    elif n==17: 
        return (40*x*y*y*y - 24*x*y + 40*x*x*x*y) # d5thCX/dy 
    
    elif n==18: 
        return (3 - 36*y*y - 12*x*x + 50*pow(y,4) + 60*x*x*y*y
                + 10*pow(x,4))	# d5thCY/dy 
    
    elif n==19: 
        return (120*pow(y,5) + 240*x*x*y*y*y + 120*pow(x,4)*y
                - 120*y*y*y - 120*x*x*y + 24*y)	# dsph6/dy 
    
    elif n==20: 
        return (20*x*y*y*y - 20*x*x*x*y)	# d5th1/dy 
    
    elif n==21: 
        return (5*y*y*y*y - 30*x*x*y*y + 5*x*x*x*x) # d5th2/dy 
