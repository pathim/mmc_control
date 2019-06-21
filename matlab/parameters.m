function [params] = parameters()
%PARAMETERS Define circuit parameters
%   This functions gives a struct defining the numerical values of the
%   circuit elements
	params.amp_iac = 1;
	params.amp_uac = 1;
	params.udc = 1.6*params.amp_uac;

	params.L   = 1.5e-3;
	params.La  = 1e-3;
	params.Ldc = 1e-3;
	params.R   = 0; % 1e-3
	params.Rdc = 0;
	params.Ra  = 0;

	params.phi=0;
	params.w=2*pi*50;    
end

