function [x,out] = l1_dual_ADMM(x0, A, b, mu, opts);
	%--------------------------------------------%
	% This program implements ADMM for the dual problem
	% with continuation method.
	%
	% Author: Yifei Wang, 2018
	%--------------------------------------------%
	if ~isfield(opts,'t');          opts.t          = 1e-2;   end
	if ~isfield(opts,'cont_alpha'); opts.cont_alpha = 0.1;    end
	if ~isfield(opts,'subiter');	opts.subiter    = 10;	  end
	if ~isfield(opts,'finiter');	opts.finiter    = 10;	  end
	if ~isfield(opts,'itPrint');    opts.itPrint    = 0;      end

	% copy paramter
	t = opts.t;				
	cont_alpha = opts.cont_alpha;
	subiter = opts.subiter;
	finiter = opts.finiter;
	itPrint = opts.itPrint;

	[m, n]=size(A);

	% set up print format
	if itPrint > 0
	    if ispc; str1 = '  %10s'; str2 = '  %7s';
	    else     str1 = '  %10s'; str2 = '  %7s'; end
	    stra = ['%5s', str2, str2, str2 '\n'];
	    str_head = sprintf(stra, 'iter', 'obj', 'cst', 'mu');
	    str_num = ['%4d %2.1e %+2.1e %+2.1e \n'];    
	end

	ATb = A'*b;
	ItAA = eye(m)+t*A*A';

	mui = max(mu, cont_alpha* max(abs(ATb(:))));
	z = zeros(m,1);
	lambda = zeros(n,1);
	%main loop
	if itPrint>0
		fprintf('%s\n', str_head);
	end
	k = 0;
	while mui>mu
		iter = 0;
		while iter<subiter;
			v = lambda/t+A'*z;
			w = v-shringkage(v,mui);
			z = ItAA\(-b-A*(lambda-t*w));
			lambda = lambda+t*(A'*z-w);
			iter = iter+1;
			if itPrint>0
				if mod(iter,itPrint)==0
					f = 0.5*norm(z)^2+b'*z;
					cst = mui-max(abs(A'*z));
					fprintf(str_num, k+iter, f, cst,mui);
				end
			end
		end
		k = k+subiter;
		% change mui and lambdaj
		mui = max(mu, cont_alpha*mui);
		
		
	end

	iter = 0;
	while iter<finiter
		v = lambda/t+A'*z;
		w = v-shringkage(v,mui);
		z = ItAA\(-b-A*(lambda-t*w));
		lambda = lambda+t*(A'*z-w);
		iter = iter+1;
		if itPrint>0
			if mod(iter,itPrint)==0
				f = 0.5*norm(z)^2+b'*z;
				cst = mui-max(abs(A'*z));
				fprintf(str_num, k+iter, f, cst,mui);
			end
		end
	end

	x = -lambda;
	out.val = 0.5*norm(A*x-b)^2+mui*norm(x,1);

	function ss = shringkage(xx,mumu)
	    ss = sign(xx).*max(abs(xx)-mumu,0);
	end	

end