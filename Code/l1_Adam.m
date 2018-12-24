function [x,out] = l1_Adam(x0, A, b, mu, opts);
	%--------------------------------------------%
	% This program implements Adam with continuation method.
	%
	% Author: Yifei Wang, 2018
	%--------------------------------------------%
	if ~isfield(opts,'s');          opts.s          = 1e-1;      end
	if ~isfield(opts,'rho1'); 		opts.rho1       = 0.9;   end
	if ~isfield(opts,'rho2'); 		opts.rho2	    = 0.999;   end
	if ~isfield(opts,'cont_alpha'); opts.cont_alpha = 0.5;    end
	if ~isfield(opts,'epsilon'); 	opts.epsilon    = 1e-8;   end	
	if ~isfield(opts,'subiter');	opts.subiter    = 50;	  end
	if ~isfield(opts,'finiter');	opts.finiter    = 300;	  end
	if ~isfield(opts,'itPrint');    opts.itPrint    = 0;      end

	% copy paramter
	s = opts.s;	
	rho1 = opts.rho1;
	rho2 = opts.rho2;
	cont_alpha = opts.cont_alpha;
	epsilon = opts.epsilon;
	subiter = opts.subiter;
	finiter = opts.finiter;
	itPrint = opts.itPrint;

	[m, n]=size(A);

	x = x0;
	% set up print format
	if itPrint > 0
	    if ispc; str1 = '  %10s'; str2 = '  %7s';
	    else     str1 = '  %10s'; str2 = '  %7s'; end
	    stra = ['%5s', str2, str2, str2 '\n'];
	    str_head = sprintf(stra, 'iter', 'obj', 'subg', 'mu');
	    str_num = ['%4d %2.1e %+2.1e %+2.1e \n'];    
	end

	ATb = A'*b;
	ATA = A'*A;

	mui = max(mu, cont_alpha* max(abs(ATb(:))));
	r = zeros(n,1);
	u = zeros(n,1);

	%main loop
	if itPrint>0
		fprintf('%s\n', str_head);
	end
	k = 0;
	g = subgrad(x);
	while mui>mu
		iter = 1;
		while iter<=subiter;
			r = rho1*r+(1-rho1)*g;
			u = rho2*u+(1-rho2)*g.*g;
			s_adapt = s*(1-rho2^(k+iter))^0.5/(1-rho1^(k+iter));
			x = x-s_adapt*r./(u.^0.5+epsilon);
			iter = iter+1;
			g = subgrad(x);
			if itPrint>0
				if mod(iter,itPrint)==0
					f = func(x);
					nrm_g = norm(g);
					fprintf(str_num, k+iter, f, nrm_g,mui);
				end
			end
		end
		k = k+subiter;
		% change mui and lambdaj
		mui = max(mu, cont_alpha*mui);
		g = subgrad(x);
	end

	iter = 1;
	while iter<=finiter
		r = rho1*r+(1-rho1)*g;
		u = rho2*u+(1-rho2)*g.*g;
		s_adapt = s*(1-rho2^(k+iter))^0.5/(1-rho1^(k+iter));
		x = x-s_adapt*r./(u.^0.5+epsilon);
		iter = iter+1;
		g = subgrad(x);
		if itPrint>0
			if mod(iter,itPrint)==0
				f = func(x);
				nrm_g = norm(g);
				fprintf(str_num, k+iter, f, nrm_g,mui);
			end
		end
	end

	out.val = func(x);

	function [fxx] = func(xx)
		fxx = 0.5*norm(A*xx-b)^2+mui*norm(xx,1);
	end

	function [gxx] = subgrad(xx)
		gxx = ATA*xx-ATb+mui*sign(xx);
	end
	
	

end