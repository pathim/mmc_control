function [M2C] = prepare_M2C(params)
%PREPARE_M2C Do symbolic calculations to get needed Matrices
%   Derive the matrices needed for the calculations

L=sym('L','positive');
R=sym('R','positive');
La=sym('La','positive');
Ra=sym('Ra','positive');
Ldc=sym('Ldc','positive');
Rdc=sym('Rdc','positive');
U0=sym('U0','real');

i=sym('i',[6,1],'real');
di=sym('di',[6,1],'real');
u=sym('u',[6,1],'real');
z=[sym('Ur','real');sym('Us','real');sym('Ut','real');sym('Udc','real')];

LM=L*eye(6);                         % Uncoupled inductors
%LM=[L+Ls,   0,   0,  -L,   0,   0
%       0,L+Ls,   0,   0,  -L,   0
%       0,   0,L+Ls,   0,   0,  -L
%      -L,   0,   0,L+Ls,   0,   0
%       0,  -L,   0,   0,L+Ls,   0
%       0,   0,  -L,   0,   0,L+Ls]; % Inductors coupled per phase
RM=R*eye(6);
uM=  sym([1 0 0 -1/2; 0 1 0 -1/2;0 0 1 -1/2;1 0 0 1/2; 0 1 0 1/2;0 0 1 1/2]);
RdcM=Rdc*[1 1 1 0 0 0; 1 1 1 0 0 0; 1 1 1 0 0 0; 0 0 0 1 1 1; 0 0 0 1 1 1; 0 0 0 1 1 1];
LdcM=Ldc*[1 1 1 0 0 0; 1 1 1 0 0 0; 1 1 1 0 0 0; 0 0 0 1 1 1; 0 0 0 1 1 1; 0 0 0 1 1 1];
RaM=  Ra*[1 0 0 1 0 0; 0 1 0 0 1 0; 0 0 1 0 0 1; 1 0 0 1 0 0; 0 1 0 0 1 0; 0 0 1 0 0 1];
LaM=  La*[1 0 0 1 0 0; 0 1 0 0 1 0; 0 0 1 0 0 1; 1 0 0 1 0 0; 0 1 0 0 1 0; 0 0 1 0 0 1];

em=LM*di==-RM*i-u-uM*z-RdcM*i-LdcM*di+U0-RaM*i-LaM*di;
nb=sum(di)==0;

es=solve([em;nb],[di;U0]);
gls=[es.di1;es.di2;es.di3;es.di4;es.di5;es.di6];
A=jacobian(gls,i);
B=jacobian(gls,u);
F=jacobian(gls,z);
C=[ 1 0 0 1 0 0; 0 1 0 0 1 0; 0 0 1 0 0 1];
[Ti,~]=eig(A);

T=orth(Ti)';

At=simplify(T*A*T');
Bt=simplify(T*B*T');
Ft=simplify(T*F);
Ct=simplify(T*(C'*C)*T');
Cnt = null(Ct);

Bp = T'*pinv(Bt)*T; % directly calculating pinv(B) does not work in a reasonable time
Cp = pinv(C);

Bn = null(B);
Cn = null(C);

M2C.A=eval(subs(A,params));
M2C.Bp=eval(subs(Bp,params));
M2C.Bn=eval(subs(Bn,params));
M2C.Cp=eval(subs(Cp,params));
M2C.Cn=eval(subs(Cn,params));
M2C.T=eval(subs(T,params));
M2C.Cnt=eval(subs(Cnt,params));
M2C.F=eval(subs(F,params));
end