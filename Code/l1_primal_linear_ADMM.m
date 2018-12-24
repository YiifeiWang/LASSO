function [x,out] = l1_primal_linear_ADMM(x0, A, b, mu, opts);
	%--------------------------------------------%
	% This program implements ADMM with linearization for the primal problem
	% with continuation method.
	%
	% Author: Yifei Wang, 2018
	%--------------------------------------------%
	if ~isfield(opts,'t');          opts.t          = 1e2;   end
	if ~isfield(opts,'s');          opts.s          = 5e-4;   end
	if ~isfield(opts,'cont_alpha'); opts.cont_alpha = 0.1;    end
	if ~isfield(opts,'subiter');	opts.subiter    = 200;	  end
	if ~isfield(opts,'finiter');	opts.finiter    = 200;	  end
	if ~isfield(opts,'itPrint');    opts.itPrint    = 0;      end

	% copy paramter
	t = opts.t;
	s = opts.s;				
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
	    str_head = sprintf(stra, 'iter', 'obj', 'obj2', 'mu');
	    str_num = ['%4d %2.1e %+2.1e %+2.1e \n'];    
	end

	ATb = A'*b;
	ATA = A'*A;

	ATAtI = ATA+t*eye(n);

	mui = max(mu, cont_alpha* max(abs(ATb(:))));
	x = x0;
	y = x;
	z = zeros(n,1);
	%main loop
	if itPrint>0
		fprintf('%s\n', str_head);
	end
	k = 0;
	while mui>mu
		iter = 0;
		while iter<subiter;
			x = x-s*(ATA*x-ATb+t*(x-y+z/t));
			y = shringkage(x+z/t,mui/t);
			z = z+t*(x-y);
			
			iter = iter+1;
			if itPrint>0
				if mod(iter,itPrint)==0
					f1 = F(x);
					f2 = F(y);
					fprintf(str_num, k+iter, f1, f2,mui);
				end
			end
		end
		k = k+subiter;
		% change mui and lambdaj
		mui = max(mu, cont_alpha*mui);
		
		
	end

	iter = 0;
	while iter<finiter
		x = x-s*(ATA*x-ATb+t*(x-y+z/t));
		y = shringkage(x+z/t,mui/t);
		z = z+t*(x-y);
		
		iter = iter+1;
		if itPrint>0
			if mod(iter,itPrint)==0
				f1 = F(x);
				f2 = F(y);
				fprintf(str_num, k+iter, f1, f2,mui);
			end
		end
	end

	out.val = F(x);

	function Fxx = F(xx)
		Fxx = 0.5*norm(A*xx-b)^2+mui*norm(xx,1);
	end

	function ss = shringkage(xx,mumu)
	    ss = sign(xx).*max(abs(xx)-mumu,0);
	end	

end