%% script to create data for plots (minimize enery pulsation)
% define variables
N = 100; % #of steps
frequency = 50; % frequency
t = linspace(0,1/frequency,N); % timebase
M2C=prepare_M2C(parameters);

%% energy pulsation without compensation
data_energy(M2C,[zeros(N,2)';-sqrt(6)/(2*1.6)*ones(N,1)'],linspace(0,0.02,N),'noCompensation');

%% energy pulsation with analytical compensation of 2nd harmonic
a=sqrt(3)/(2*1.6);
data_energy(M2C,[a*sin(2*(2*pi*frequency)*t+pi/3);a*cos(2*(2*pi*frequency)*t+pi/3);-sqrt(6)/(2*1.6)*ones(N,1)'],linspace(0,0.02,N),'analytical2ndHarmonic');

%% energy pulsation with pso optimized harmonics
rng default  % For reproducibility
Nvar = 18;
ub = 2*ones(Nvar,1);
lb = -2*ones(Nvar,1);

options = optimoptions('particleswarm','SwarmSize',200,'HybridFcn',@fmincon,'FunctionTolerance',1e-7,'MaxStallIterations',2*Nvar);
[c,fval,exitflag,output] = particleswarm(@(x)objfun(M2C,linspace(0,1/frequency,1e3),x),Nvar,lb,ub,options);

%create xf trajectory
    xft = [...
    c( 1)*cos(2*t*(2*pi*frequency))+c( 2)*sin(2*t*(2*pi*frequency)) + c( 3)*cos(3*t*(2*pi*frequency))+c( 4)*sin(3*t*(2*pi*frequency)) + c( 5)*cos(4*t*(2*pi*frequency))+c( 6)*sin(4*t*(2*pi*frequency)) ;...
    c( 7)*cos(2*t*(2*pi*frequency))+c( 8)*sin(2*t*(2*pi*frequency)) + c( 9)*cos(3*t*(2*pi*frequency))+c(10)*sin(3*t*(2*pi*frequency)) + c(11)*cos(4*t*(2*pi*frequency))+c(12)*sin(4*t*(2*pi*frequency)) ;...
    -sqrt(6)/(2*1.6)*ones(length(t),1)' + ...
    c(13)*cos(2*t*(2*pi*frequency))+c(14)*sin(2*t*(2*pi*frequency)) + c(15)*cos(3*t*(2*pi*frequency))+c(16)*sin(3*t*(2*pi*frequency)) + c(17)*cos(4*t*(2*pi*frequency))+c(18)*sin(4*t*(2*pi*frequency)) ;
    ];

data_energy(M2C,xft,t,'pso2ndTo4th');