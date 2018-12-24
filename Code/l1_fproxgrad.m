function [x,out] = l1_fproxgrad(x0, A, b, mu, opts);
	%--------------------------------------------%
	%This program implements the fast proximal gradient 
	%method with continuation method.
	%
	%Author: Yifei Wang, 2018
	%--------------------------------------------%
	if ~isfield(opts,'s');          opts.s          = 5e-4;   end
	if ~isfield(opts,'cont_alpha'); opts.cont_alpha = 0.5;    end
	if ~isfield(opts,'maxiter');	opts.maxiter    = 1e4;	  end
	if ~isfield(opts,'itPrint');    opts.itPrint    = 0;      end
	if ~isfield(opts,'tol1');       opts.tol1       = 1e-6;   end
	if ~isfield(opts,'tol2');       opts.tol2       = 1e-8;   end

	%copy paramter
	s = opts.s;				
	maxiter = opts.maxiter;
	cont_alpha = opts.cont_alpha;
	itPrint = opts.itPrint;
	tol1 = opts.tol1;
	tol2 = opts.tol2;

	% set up print format
	if itPrint > 0
	    if ispc; str1 = '  %10s'; str2 = '  %7s';
	    else     str1 = '  %10s'; str2 = '  %7s'; end
	    stra = ['%5s', str2, str2, str1, str1, str2, '\n'];
	    str_head = sprintf(stra, 'iter', 'obj', 'mu', 'prox_nrm', 'ratio');
	    str_num = ['%4d  %+2.2e  %+2.1e %+2.1e %+2.1e \n'];    
	end

	ATA = A'*A;
	ATb = A'*b;

	x = x0;
	mui = max(mu, cont_alpha* max(abs(ATb(:))));
	f = F(x);
	ratio = 1;
	%main loop
	k = 1;
	iter = 1;
	if itPrint>0
		fprintf('%s\n', str_head);
	end
	while mui>mu
		while ratio>=tol1
			fp = f; xp = x;
			y = x + (k-1)/(k+2)*(x-xp);
			[grad, Gsy, res] = compres(y);
			x = y - Gsy;
			f = F(x);
			ratio = abs((f-fp)/fp);
			k = k+1;
			iter = iter+1;
			if itPrint>0
				if mod(iter,itPrint)==0
					 fprintf(str_num, iter, f, mui, res, ratio);
				end
			end
			if (iter>maxiter);break;end
		end
		mui = max(mu, cont_alpha*(min(mui, max(abs(ATA*x - ATb)))));
		fp = f; xp = x;
		k = 1;
		y = x + (k-1)/(k+2)*(x-xp);
		[grad, Gsy, res] = compres(y);
		iter = iter+1;
		x = y - Gsy;
		f = F(x);
		ratio = abs((f-fp)/fp);
		k = k+1;
		if (iter>maxiter);break;end
	end

	while ratio>=tol2
		fp = f; xp = x;
		y = x + (k-1)/(k+2)*(x-xp);
		[grad, Gsy, res] = compres(y);
		x = y - Gsy;
		f = F(x);
		ratio = abs((f-fp)/fp);
		iter = iter+1;
		k = k+1;
		if itPrint>0
			if mod(iter,itPrint)==0
				 fprintf(str_num, iter, f, mui, res, ratio);
			end
		end
		if (iter>maxiter);break;end
	end

	out.val = F(x);

	function [Fxx]=F(xx)
		Fxx = 0.5*norm(A*xx-b)^2+mui*norm(xx,1);
	end

	function ss = shringkage(xx,mumu)
	    ss = sign(xx).*max(abs(xx)-mumu,0);
	end

	function [grad, Gsx, res] = compres(xx)
		grad = ATA*xx - ATb;
		yx = xx - s*grad;
		Gsx = xx - shringkage(yx, s*mui);
		res = norm(Gsx);
	end

end