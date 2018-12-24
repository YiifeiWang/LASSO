function [x,out] = l1_subgrad(x0, A, b, mu, opts);
	%--------------------------------------------%
	%This program implements the subgradient 
	%method with continuation method.
	%
	%Author: Yifei Wang, 2018
	%--------------------------------------------%
	if ~isfield(opts,'s');          opts.s          = 2.8e-4;   end
	if ~isfield(opts,'maxiter');    opts.maxiter    = 300;    end
	if ~isfield(opts,'cont_num');   opts.cont_num   = 6;      end
	if ~isfield(opts,'cont_alpha'); opts.cont_alpha = 10;     end
	if ~isfield(opts,'display');    opts.display    = 0;      end
	if ~isfield(opts,'tol');        opts.tol        = 1e-6;   end

	s = opts.s;		
	maxiter = opts.maxiter;
	cont_num = opts.cont_num;
	cont_alpha = opts.cont_alpha;
	tol = opts.tol;


	ATA=A'*A;
	ATb=A'*b;

	[m, n] = size(A);
	mui = mu*cont_alpha^(cont_num-1);
	x = x0;
	
	%main loop
	for i=1:cont_num
		for j=1:maxiter
			x=x-s*g(x);
			if opts.display&&mod(j,50)==0; disp([i j F(x)]); end
		end
		if i<cont_num
			mui = mui/cont_alpha;
		end
	end

	out.val = F(x);
	out.s = s;
	out.mu = mui;

	function [Fxx]=F(xx)
		Fxx = 0.5*norm(A*xx-b)^2+mui*norm(xx,1);
	end

	function [gxx]=g(xx) 
		gxx = ATA*xx-ATb+mui*sign(xx);
	end


end