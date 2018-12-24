function [x,out] = l1_RMSProp(x0, A, b, mu, opts);
	%--------------------------------------------%
	% This program implements RMSProp with continuation method.
	%
	% Author: Yifei Wang, 2018
	%--------------------------------------------%
	if ~isfield(opts,'s');          opts.s          = 4e-2;   end
	if ~isfield(opts,'s_decay');    opts.s_decay    = 0.14;   end
	if ~isfield(opts,'rho'); 		opts.rho        = 0.9;    end
	if ~isfield(opts,'cont_alpha'); opts.cont_alpha = 0.1;    end
	if ~isfield(opts,'epsilon'); 	opts.epsilon    = 1e-8;   end	
	if ~isfield(opts,'subiter');	opts.subiter    = 280;	  end
	if ~isfield(opts,'finiter');	opts.finiter    = 280;	  end
	if ~isfield(opts,'itPrint');    opts.itPrint    = 0;      end

	% copy paramter
	s = opts.s;	
	s_decay = opts.s_decay;
	rho = opts.rho;
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
	    str_head = sprintf(stra, 'iter', 'obj', 'subg', 'mu', 's');
	    str_num = ['%4d %2.1e %+2.1e  %+2.1e  %2.1e  \n'];    
	end

	ATb = A'*b;
	ATA = A'*A;

	mui = max(mu, cont_alpha* max(abs(ATb(:))));
	r = zeros(n,1);

	%main loop
	if itPrint>0
		fprintf('%s\n', str_head);
	end
	k = 0;
	g = subgrad(x);
	while mui>mu
		iter = 1;
		while iter<=subiter;
			r = rho*r+(1-rho)*g.*g;
			x = x-s*g./(r.^0.5+epsilon);
			iter = iter+1;
			g = subgrad(x);
			if itPrint>0
				if mod(iter+k,itPrint)==0
					f = func(x);
					nrm_g = norm(g);
					fprintf(str_num, k+iter, f, nrm_g,mui,s);
				end
			end
		end
		k = k+subiter;
		% change mui and lambdaj
		mui = max(mu, cont_alpha*mui);
		s = s*s_decay;
		g = subgrad(x);
		%r = zeros(n,1);
	end

	iter = 1;
	while iter<=finiter
		r = rho*r+(1-rho)*g.*g;
		x = x-s*g./(r.^0.5+epsilon);
		iter = iter+1;
		g = subgrad(x);
		if itPrint>0
			if mod(iter+k,itPrint)==0
				f = func(x);
				nrm_g = norm(g);
				fprintf(str_num, k+iter, f, nrm_g,mui,s);
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