function [x,out] = l1_projgrad(x0, A, b, mu, opts);
	%--------------------------------------------%
	%This program implements the projection gradient 
	%method with continuation method.
	%
	%Author: Yifei Wang, 2018
	%--------------------------------------------%
	if ~isfield(opts,'s');    		opts.s           = 4e-4;   end
	if ~isfield(opts,'maxiter');    opts.maxiter     = 180;    end
	if ~isfield(opts,'cont_num');   opts.cont_num    = 6;      end
	if ~isfield(opts,'cont_alpha'); opts.cont_alpha  = 10;     end
	if ~isfield(opts,'tol');        opts.tol         = 1e-6;   end

	s = opts.s;
	maxiter = opts.maxiter;
	cont_num = opts.cont_num;
	cont_alpha = opts.cont_alpha;
	tol = opts.tol;


	[~, n] = size(A);
	ATA = A'*A;
	bTb = b'*b;
	ATb = A'*b;
	mui = mu*cont_alpha^(cont_num-1);
	x_p = x0.*(x0>0);
	x_n = x_p - x0;

	%main loop
	for i=1:cont_num
		for j=1:maxiter
			new_x_p = x_p - s*(ATA*(x_p-x_n)+mui*ones(n,1)-ATb);
			new_x_n = x_n - s*(ATA*(x_n-x_p)+mui*ones(n,1)+ATb);
			x_p = new_x_p.*(new_x_p>0);
			x_n = new_x_n.*(new_x_n>0);
		end
		if i<cont_num
			mui = mui/cont_alpha;
		end
	end

	x = x_p-x_n;
	out = [];
	out.val = 0.5*norm(A*x-b)^2+mu*norm(x,1);
	out.mu = mui;


end