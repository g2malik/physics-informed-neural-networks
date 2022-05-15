import numpy as np
import matplotlib.pyplot as plt

#-----------------------------------------#

length = 0.2
width = 0.01


k = 385 #Copper (W/m K)

h = 20 # (W/m^2 K)
T_inf = 25
T_base = 100

#-----------------------------------------#
x_range = np.linspace (0, length, 50)

A_c = width*width
Perimeter = width*4

theta_b = T_base - T_inf

m = (h*Perimeter/(k * A_c))**0.5

#-----------------------------------------#

theta_ratio = (np.cosh(m*(length - x_range)) + (h/(m*k)*np.sinh(m*(length - x_range))))/(np.cosh(m*length) + (h/(m*k)*np.sinh(m*(length))))
theta = theta_ratio * theta_b
T_profile = theta + T_inf

print(x_range)
print(T_profile)

plt.plot([(-width/2), (-width/2), (width/2), (width/2)], [0, length, length, 0])
plt.xlim([(-width*2), (width*2)])
plt.ylim([0, length*1.5])

T_contour = np.reshape(T_profile, (len(T_profile), 1)) 
plt.imshow(T_contour, extent = [-width/2, width/2, 0, length], origin='lower', aspect='auto', interpolation='bilinear', cmap='YlOrRd')
plt.colorbar()

plt.show()