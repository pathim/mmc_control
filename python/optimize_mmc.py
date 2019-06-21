#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 15:56:36 2019

@author: Patrick Himmelmann
"""

from casadi import Opti,MX,Function,vertcat,jacobian,sumsqr,sum1,sum2,horzcat,simpleRK,cumsum,evalf,substitute
from numpy import array,ones,eye,vstack,zeros,sqrt,linspace,pi,hstack,mean,diag,atleast_2d,savetxt

# Define normalized parameters
amp_vac = 1 # AC-voltage (amplitude)
amp_iac = 1 # AC-current (amplitude)
vdc = 1.6   # DC-voltage
L = 1.5e-3  # arm inductance
La = 1e-3   # external inductance on AC-side
Ldc = 1e-3  # external inductance on DC-side
R = 1e-3    # arm resitance
Ra = 1e-3   # external resistance on AC-side
Rdc = 1e-3  # external resistance on AC-side
f = 50      # AC frequency
phi = 0     # Phase shift of AC current
N = 102     # Number of time steps (divisible by six so all six arms can be identical)

# Define state-space matrices
A =array([
[ -(5*L**2*R + 4*L**2*Ra + 3*L**2*Rdc + 6*L*La*R + 12*L*Ldc*R + 6*L*La*Rdc + 12*L*Ldc*Ra + 12*La*Ldc*R)/(6*L*(L + 2*La)*(L + 3*Ldc)),                    (L**2*R + 2*L**2*Ra - 3*L**2*Rdc + 6*L*Ldc*R - 6*L*La*Rdc + 6*L*Ldc*Ra + 6*La*Ldc*R)/(6*L*(L + 2*La)*(L + 3*Ldc)),                    (L**2*R + 2*L**2*Ra - 3*L**2*Rdc + 6*L*Ldc*R - 6*L*La*Rdc + 6*L*Ldc*Ra + 6*La*Ldc*R)/(6*L*(L + 2*La)*(L + 3*Ldc)),                   (L**2*R - 4*L**2*Ra + 3*L**2*Rdc + 6*L*La*R + 6*L*La*Rdc - 12*L*Ldc*Ra + 12*La*Ldc*R)/(6*L*(L + 2*La)*(L + 3*Ldc)),                                (L**2*R + 2*L**2*Ra + 3*L**2*Rdc + 6*L*La*Rdc + 6*L*Ldc*Ra - 6*La*Ldc*R)/(6*L*(L + 2*La)*(L + 3*Ldc)),                                               (L**2*R + 2*L**2*Ra + 3*L**2*Rdc + 6*L*La*Rdc + 6*L*Ldc*Ra - 6*La*Ldc*R)/(6*L*(L + 2*La)*(L + 3*Ldc))],
[                (L**2*R + 2*L**2*Ra - 3*L**2*Rdc + 6*L*Ldc*R - 6*L*La*Rdc + 6*L*Ldc*Ra + 6*La*Ldc*R)/(6*(L**2 + 2*La*L)*(L + 3*Ldc)), -(5*L**2*R + 4*L**2*Ra + 3*L**2*Rdc + 6*L*La*R + 12*L*Ldc*R + 6*L*La*Rdc + 12*L*Ldc*Ra + 12*La*Ldc*R)/(6*(L**2 + 2*La*L)*(L + 3*Ldc)),                  (L**2*R + 2*L**2*Ra - 3*L**2*Rdc + 6*L*Ldc*R - 6*L*La*Rdc + 6*L*Ldc*Ra + 6*La*Ldc*R)/(6*(L**2 + 2*La*L)*(L + 3*Ldc)),                              (L**2*R + 2*L**2*Ra + 3*L**2*Rdc + 6*L*La*Rdc + 6*L*Ldc*Ra - 6*La*Ldc*R)/(6*(L**2 + 2*La*L)*(L + 3*Ldc)),                 (L**2*R - 4*L**2*Ra + 3*L**2*Rdc + 6*L*La*R + 6*L*La*Rdc - 12*L*Ldc*Ra + 12*La*Ldc*R)/(6*(L**2 + 2*La*L)*(L + 3*Ldc)),                                             (L**2*R + 2*L**2*Ra + 3*L**2*Rdc + 6*L*La*Rdc + 6*L*Ldc*Ra - 6*La*Ldc*R)/(6*(L**2 + 2*La*L)*(L + 3*Ldc))],
[                (L**2*R + 2*L**2*Ra - 3*L**2*Rdc + 6*L*Ldc*R - 6*L*La*Rdc + 6*L*Ldc*Ra + 6*La*Ldc*R)/(6*(L**2 + 2*La*L)*(L + 3*Ldc)),                  (L**2*R + 2*L**2*Ra - 3*L**2*Rdc + 6*L*Ldc*R - 6*L*La*Rdc + 6*L*Ldc*Ra + 6*La*Ldc*R)/(6*(L**2 + 2*La*L)*(L + 3*Ldc)), -(5*L**2*R + 4*L**2*Ra + 3*L**2*Rdc + 6*L*La*R + 12*L*Ldc*R + 6*L*La*Rdc + 12*L*Ldc*Ra + 12*La*Ldc*R)/(6*(L**2 + 2*La*L)*(L + 3*Ldc)),                              (L**2*R + 2*L**2*Ra + 3*L**2*Rdc + 6*L*La*Rdc + 6*L*Ldc*Ra - 6*La*Ldc*R)/(6*(L**2 + 2*La*L)*(L + 3*Ldc)),                              (L**2*R + 2*L**2*Ra + 3*L**2*Rdc + 6*L*La*Rdc + 6*L*Ldc*Ra - 6*La*Ldc*R)/(6*(L**2 + 2*La*L)*(L + 3*Ldc)),                                (L**2*R - 4*L**2*Ra + 3*L**2*Rdc + 6*L*La*R + 6*L*La*Rdc - 12*L*Ldc*Ra + 12*La*Ldc*R)/(6*(L**2 + 2*La*L)*(L + 3*Ldc))],
[               (L**2*R - 4*L**2*Ra + 3*L**2*Rdc + 6*L*La*R + 6*L*La*Rdc - 12*L*Ldc*Ra + 12*La*Ldc*R)/(6*(L**2 + 2*La*L)*(L + 3*Ldc)),                              (L**2*R + 2*L**2*Ra + 3*L**2*Rdc + 6*L*La*Rdc + 6*L*Ldc*Ra - 6*La*Ldc*R)/(6*(L**2 + 2*La*L)*(L + 3*Ldc)),                              (L**2*R + 2*L**2*Ra + 3*L**2*Rdc + 6*L*La*Rdc + 6*L*Ldc*Ra - 6*La*Ldc*R)/(6*(L**2 + 2*La*L)*(L + 3*Ldc)), -(5*L**2*R + 4*L**2*Ra + 3*L**2*Rdc + 6*L*La*R + 12*L*Ldc*R + 6*L*La*Rdc + 12*L*Ldc*Ra + 12*La*Ldc*R)/(6*(L**2 + 2*La*L)*(L + 3*Ldc)),                  (L**2*R + 2*L**2*Ra - 3*L**2*Rdc + 6*L*Ldc*R - 6*L*La*Rdc + 6*L*Ldc*Ra + 6*La*Ldc*R)/(6*(L**2 + 2*La*L)*(L + 3*Ldc)),                                 (L**2*R + 2*L**2*Ra - 3*L**2*Rdc + 6*L*Ldc*R - 6*L*La*Rdc + 6*L*Ldc*Ra + 6*La*Ldc*R)/(6*(L**2 + 2*La*L)*(L + 3*Ldc))],
[                            (L**2*R + 2*L**2*Ra + 3*L**2*Rdc + 6*L*La*Rdc + 6*L*Ldc*Ra - 6*La*Ldc*R)/(6*(L**2 + 2*La*L)*(L + 3*Ldc)),                 (L**2*R - 4*L**2*Ra + 3*L**2*Rdc + 6*L*La*R + 6*L*La*Rdc - 12*L*Ldc*Ra + 12*La*Ldc*R)/(6*(L**2 + 2*La*L)*(L + 3*Ldc)),                              (L**2*R + 2*L**2*Ra + 3*L**2*Rdc + 6*L*La*Rdc + 6*L*Ldc*Ra - 6*La*Ldc*R)/(6*(L**2 + 2*La*L)*(L + 3*Ldc)),                  (L**2*R + 2*L**2*Ra - 3*L**2*Rdc + 6*L*Ldc*R - 6*L*La*Rdc + 6*L*Ldc*Ra + 6*La*Ldc*R)/(6*(L**2 + 2*La*L)*(L + 3*Ldc)), -(5*L**2*R + 4*L**2*Ra + 3*L**2*Rdc + 6*L*La*R + 12*L*Ldc*R + 6*L*La*Rdc + 12*L*Ldc*Ra + 12*La*Ldc*R)/(6*(L**2 + 2*La*L)*(L + 3*Ldc)),                                 (L**2*R + 2*L**2*Ra - 3*L**2*Rdc + 6*L*Ldc*R - 6*L*La*Rdc + 6*L*Ldc*Ra + 6*La*Ldc*R)/(6*(L**2 + 2*La*L)*(L + 3*Ldc))],
[             (L**2*R + 2*L**2*Ra + 3*L**2*Rdc + 6*L*La*Rdc + 6*L*Ldc*Ra - 6*La*Ldc*R)/(6*(2*L**2*La + 3*L**2*Ldc + L**3 + 6*L*La*Ldc)),               (L**2*R + 2*L**2*Ra + 3*L**2*Rdc + 6*L*La*Rdc + 6*L*Ldc*Ra - 6*La*Ldc*R)/(6*(2*L**2*La + 3*L**2*Ldc + L**3 + 6*L*La*Ldc)),  (L**2*R - 4*L**2*Ra + 3*L**2*Rdc + 6*L*La*R + 6*L*La*Rdc - 12*L*Ldc*Ra + 12*La*Ldc*R)/(6*(2*L**2*La + 3*L**2*Ldc + L**3 + 6*L*La*Ldc)),   (L**2*R + 2*L**2*Ra - 3*L**2*Rdc + 6*L*Ldc*R - 6*L*La*Rdc + 6*L*Ldc*Ra + 6*La*Ldc*R)/(6*(2*L**2*La + 3*L**2*Ldc + L**3 + 6*L*La*Ldc)),   (L**2*R + 2*L**2*Ra - 3*L**2*Rdc + 6*L*Ldc*R - 6*L*La*Rdc + 6*L*Ldc*Ra + 6*La*Ldc*R)/(6*(2*L**2*La + 3*L**2*Ldc + L**3 + 6*L*La*Ldc)), -(5*L**2*R + 4*L**2*Ra + 3*L**2*Rdc + 6*L*La*R + 12*L*Ldc*R + 6*L*La*Rdc + 12*L*Ldc*Ra + 12*La*Ldc*R)/(6*(2*L**2*La + 3*L**2*Ldc + L**3 + 6*L*La*Ldc))],
])

B = array([
[ -(5*L**2 + 6*L*La + 12*L*Ldc + 12*La*Ldc)/(6*L*(L + 2*La)*(L + 3*Ldc)),                 (L**2 + 6*Ldc*L + 6*La*Ldc)/(6*L*(L + 2*La)*(L + 3*Ldc)),                  (L**2 + 6*Ldc*L + 6*La*Ldc)/(6*L*(L + 2*La)*(L + 3*Ldc)),                  (L**2 + 6*La*L + 12*La*Ldc)/(6*L*(L + 2*La)*(L + 3*Ldc)),                            (L**2 - 6*La*Ldc)/(6*L*(L + 2*La)*(L + 3*Ldc)),                                          (L**2 - 6*La*Ldc)/(6*L*(L + 2*La)*(L + 3*Ldc))],
[             (L**2 + 6*Ldc*L + 6*La*Ldc)/(6*(L**2 + 2*La*L)*(L + 3*Ldc)), -(5*L**2 + 6*L*La + 12*L*Ldc + 12*La*Ldc)/(6*(L**2 + 2*La*L)*(L + 3*Ldc)),                (L**2 + 6*Ldc*L + 6*La*Ldc)/(6*(L**2 + 2*La*L)*(L + 3*Ldc)),                          (L**2 - 6*La*Ldc)/(6*(L**2 + 2*La*L)*(L + 3*Ldc)),                (L**2 + 6*La*L + 12*La*Ldc)/(6*(L**2 + 2*La*L)*(L + 3*Ldc)),                                        (L**2 - 6*La*Ldc)/(6*(L**2 + 2*La*L)*(L + 3*Ldc))],
[             (L**2 + 6*Ldc*L + 6*La*Ldc)/(6*(L**2 + 2*La*L)*(L + 3*Ldc)),               (L**2 + 6*Ldc*L + 6*La*Ldc)/(6*(L**2 + 2*La*L)*(L + 3*Ldc)),  -(5*L**2 + 6*L*La + 12*L*Ldc + 12*La*Ldc)/(6*(L**2 + 2*La*L)*(L + 3*Ldc)),                          (L**2 - 6*La*Ldc)/(6*(L**2 + 2*La*L)*(L + 3*Ldc)),                          (L**2 - 6*La*Ldc)/(6*(L**2 + 2*La*L)*(L + 3*Ldc)),                              (L**2 + 6*La*L + 12*La*Ldc)/(6*(L**2 + 2*La*L)*(L + 3*Ldc))],
[             (L**2 + 6*La*L + 12*La*Ldc)/(6*(L**2 + 2*La*L)*(L + 3*Ldc)),                         (L**2 - 6*La*Ldc)/(6*(L**2 + 2*La*L)*(L + 3*Ldc)),                          (L**2 - 6*La*Ldc)/(6*(L**2 + 2*La*L)*(L + 3*Ldc)),  -(5*L**2 + 6*L*La + 12*L*Ldc + 12*La*Ldc)/(6*(L**2 + 2*La*L)*(L + 3*Ldc)),                (L**2 + 6*Ldc*L + 6*La*Ldc)/(6*(L**2 + 2*La*L)*(L + 3*Ldc)),                              (L**2 + 6*Ldc*L + 6*La*Ldc)/(6*(L**2 + 2*La*L)*(L + 3*Ldc))],
[                       (L**2 - 6*La*Ldc)/(6*(L**2 + 2*La*L)*(L + 3*Ldc)),               (L**2 + 6*La*L + 12*La*Ldc)/(6*(L**2 + 2*La*L)*(L + 3*Ldc)),                          (L**2 - 6*La*Ldc)/(6*(L**2 + 2*La*L)*(L + 3*Ldc)),                (L**2 + 6*Ldc*L + 6*La*Ldc)/(6*(L**2 + 2*La*L)*(L + 3*Ldc)),  -(5*L**2 + 6*L*La + 12*L*Ldc + 12*La*Ldc)/(6*(L**2 + 2*La*L)*(L + 3*Ldc)),                              (L**2 + 6*Ldc*L + 6*La*Ldc)/(6*(L**2 + 2*La*L)*(L + 3*Ldc))],
[        (L**2 - 6*La*Ldc)/(6*(2*L**2*La + 3*L**2*Ldc + L**3 + 6*L*La*Ldc)),          (L**2 - 6*La*Ldc)/(6*(2*L**2*La + 3*L**2*Ldc + L**3 + 6*L*La*Ldc)), (L**2 + 6*La*L + 12*La*Ldc)/(6*(2*L**2*La + 3*L**2*Ldc + L**3 + 6*L*La*Ldc)), (L**2 + 6*Ldc*L + 6*La*Ldc)/(6*(2*L**2*La + 3*L**2*Ldc + L**3 + 6*L*La*Ldc)), (L**2 + 6*Ldc*L + 6*La*Ldc)/(6*(2*L**2*La + 3*L**2*Ldc + L**3 + 6*L*La*Ldc)), -(5*L**2 + 6*L*La + 12*L*Ldc + 12*La*Ldc)/(6*(2*L**2*La + 3*L**2*Ldc + L**3 + 6*L*La*Ldc))]
])

C = hstack((eye(3),eye(3)))

F =array([
[                -(4*L**2 + 12*Ldc*L)/(6*L*(L + 2*La)*(L + 3*Ldc)),                  (2*L**2 + 6*Ldc*L)/(6*L*(L + 2*La)*(L + 3*Ldc)),                    (2*L**2 + 6*Ldc*L)/(6*L*(L + 2*La)*(L + 3*Ldc)),                   (3*L**2 + 6*La*L)/(6*L*(L + 2*La)*(L + 3*Ldc))],
[                (2*L**2 + 6*Ldc*L)/(6*(L**2 + 2*La*L)*(L + 3*Ldc)),              -(4*L**2 + 12*Ldc*L)/(6*(L**2 + 2*La*L)*(L + 3*Ldc)),                  (2*L**2 + 6*Ldc*L)/(6*(L**2 + 2*La*L)*(L + 3*Ldc)),                 (3*L**2 + 6*La*L)/(6*(L**2 + 2*La*L)*(L + 3*Ldc))],
[                (2*L**2 + 6*Ldc*L)/(6*(L**2 + 2*La*L)*(L + 3*Ldc)),                (2*L**2 + 6*Ldc*L)/(6*(L**2 + 2*La*L)*(L + 3*Ldc)),                -(4*L**2 + 12*Ldc*L)/(6*(L**2 + 2*La*L)*(L + 3*Ldc)),                 (3*L**2 + 6*La*L)/(6*(L**2 + 2*La*L)*(L + 3*Ldc))],
[              -(4*L**2 + 12*Ldc*L)/(6*(L**2 + 2*La*L)*(L + 3*Ldc)),                (2*L**2 + 6*Ldc*L)/(6*(L**2 + 2*La*L)*(L + 3*Ldc)),                  (2*L**2 + 6*Ldc*L)/(6*(L**2 + 2*La*L)*(L + 3*Ldc)),                -(3*L**2 + 6*La*L)/(6*(L**2 + 2*La*L)*(L + 3*Ldc))],
[                (2*L**2 + 6*Ldc*L)/(6*(L**2 + 2*La*L)*(L + 3*Ldc)),              -(4*L**2 + 12*Ldc*L)/(6*(L**2 + 2*La*L)*(L + 3*Ldc)),                  (2*L**2 + 6*Ldc*L)/(6*(L**2 + 2*La*L)*(L + 3*Ldc)),                -(3*L**2 + 6*La*L)/(6*(L**2 + 2*La*L)*(L + 3*Ldc))],
[ (2*L**2 + 6*Ldc*L)/(6*(2*L**2*La + 3*L**2*Ldc + L**3 + 6*L*La*Ldc)), (2*L**2 + 6*Ldc*L)/(6*(2*L**2*La + 3*L**2*Ldc + L**3 + 6*L*La*Ldc)), -(4*L**2 + 12*Ldc*L)/(6*(2*L**2*La + 3*L**2*Ldc + L**3 + 6*L*La*Ldc)), -(3*L**2 + 6*La*L)/(6*(2*L**2*La + 3*L**2*Ldc + L**3 + 6*L*La*Ldc))],
])

# Transformation matrix. Needed for some constraints
T =array([
[  6**(1/2)/6,  6**(1/2)/6,  6**(1/2)/6,  6**(1/2)/6,  6**(1/2)/6, 6**(1/2)/6],
[        1/2,       -1/2,          0,       -1/2,        1/2,         0],
[  3**(1/2)/6,  3**(1/2)/6, -3**(1/2)/3, -3**(1/2)/6, -3**(1/2)/6, 3**(1/2)/3],
[       -1/2,        1/2,          0,       -1/2,        1/2,         0],
[ -3**(1/2)/6, -3**(1/2)/6,  3**(1/2)/3, -3**(1/2)/6, -3**(1/2)/6, 3**(1/2)/3],
[ -6**(1/2)/6, -6**(1/2)/6, -6**(1/2)/6,  6**(1/2)/6,  6**(1/2)/6, 6**(1/2)/6],
])

# Define given signals
t=MX.sym('t') # time variable
yd=Function('yd',[t],[amp_iac*(2*pi*f*t-MX([0,1,2])*2*pi/3-phi).cos()]) # Reference signal for AC-current
dyd=Function('dyd',[t],[jacobian(yd(t),t)]) # time derivative of AC-current
z=Function('z',[t],[vertcat(amp_vac*(2*pi*f*t-MX([0,1,2])*2*pi/3).cos(),vdc)]) # external voltages

# Define state-space model
x = MX.sym('x',13,1) # states: time,arm currents, arm energy
v = MX.sym('p',6,1)  # Arm voltages
xdot = Function('xdot',[x,v],[vertcat( # state derivatives
	1,                      # time
	A@x[1:7]+B@v+F@z(x[0]), # currents
	v*x[1:7]                # energies
)])

def integrate(x,u,dt,derivative):
	integrator=simpleRK(derivative,2)
	startval=vertcat(0,x[:,0])
	for i in range(1,u.shape[1]):
		startval=integrator(startval,u[:,i-1],dt)
		x[:,i]=startval[1:]

#Define timegrid over which to minimize
t_grid=linspace(0,0.02,N+1) # N+1 steps because there is one step after the end
dt=t_grid[1] # timestep

def calc_energy_tenth_power(use_V0=False,Idc_non_const=False):
	opti = Opti()
	opti.solver('ipopt')
	X0 = opti.variable(12,1) # Start values for currents and energies
	V = opti.variable(6,N+1) # arm voltages
	
	# calculate states
	states = MX(12,N+1)
	states[:,0] = X0
	integrate(states,V,dt,xdot)

	I = states[:6,:]      # view for the current states
	energy = states[6:,:] # view for the energy states

	opti.minimize(sumsqr((1000*energy)**5)) # optimization objective
	opti.subject_to((C@I)[:2,:]==yd(atleast_2d(t_grid))[:2,:]) # constrain output current to reference singnals
	opti.subject_to(states[:,0]==states[:,-1]) # constrain states to periodicity
	opti.subject_to(V[:,0]==V[:,-1]) # constrain arm voltages to periodicity
	if not use_V0: opti.subject_to(sum1(V)==0) # constrain star point voltage to zero
	opti.subject_to(opti.bounded(-1.94,V,1.94)) # define maximum values for arm voltages
	if not Idc_non_const: opti.subject_to((T@I)[5,1:]==(T@I)[5,0]) # contrain DC-current to be constant
	
	sol=opti.solve()
	r_V,r_states=substitute([V,states],[X0,V],[sol.value(X0),sol.value(V)])
	
	return evalf(r_V),evalf(r_states)


def prepare_data(x):
	W=x[6:,:].toarray()
	W=W.T-(W.max(1)+W.min(1))/2
	data=hstack((atleast_2d(t_grid).T,W))
	return data
	
if __name__=='__main__':
	v,states=calc_energy_tenth_power(use_V0=False,Idc_non_const=True)
	data=prepare_data(states)
	savetxt('data/pow10.csv',data,delimiter=',')
	v,states=calc_energy_tenth_power(use_V0=True,Idc_non_const=True)
	data=prepare_data(states)
	savetxt('data/pow10_V0.csv',data,delimiter=',')