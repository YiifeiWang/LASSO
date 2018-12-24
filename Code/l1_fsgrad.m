function [x,out] = l1_fsgrad(x0, A, b, mu, opts);
	%--------------------------------------------%
	% This program implements the fast gradient 
	% method to the smoothed primal problem
	% with continuation method.
	%
	% Author: Yifei Wang, 2018
	%--------------------------------------------%
	if ~isfield(opts,'s');          opts.s          = 4e-4;   end
	if ~isfield(opts,'cont_alpha'); opts.cont_alpha = 0.2;    end
	if ~isfield(opts,'lambda');    	opts.lambda    	= 1e-6;   end
	if ~isfield(opts,'lambdaj');    opts.lambdaj    = 1e-3;   end
	if ~isfield(opts,'lambdaPow');  opts.lambdaPow  = 0.5;    end
	if ~isfield(opts,'subiter');	opts.subiter    = 200;	  end
	if ~isfield(opts,'finiter');	opts.finiter    = 700;	  end
	if ~isfield(opts,'itPrint');    opts.itPrint    = 0;      end
	if ~isfield(opts,'BB');    	    opts.BB       	= 1;   	  end

	% copy paramter
	s = opts.s;				
	cont_alpha = opts.cont_alpha;
	lambda = opts.lambda;
	lambdaj = opts.lambdaj;
	lambdaPow = opts.lambdaPow;
	subiter = opts.subiter;
	finiter = opts.finiter;
	itPrint = opts.itPrint;

	[~, n]=size(A);

	% set up print format
	if itPrint > 0
	    if ispc; str1 = '  %10s'; str2 = '  %7s';
	    else     str1 = '  %10s'; str2 = '  %7s'; end
	    stra = ['%5s', str2, str2, str2, str2, str1, str2, '\n'];
	    str_head = sprintf(stra, 'iter', 'obj', 'mu', 'lambda');
	    str_num = ['%4d %+2.1e %+2.1e %+2.1e \n'];    
	end

	ATA = A'*A;
	ATb = A'*b;
	bTb = b'*b;

	x = x0;
	mui = max(mu, cont_alpha* max(abs(ATb(:))));
	k = 0;
	xp = x;
	y = x + (k-1)/(k+2)*(x-xp);
	[f, grad] = Fml(y);
	x = y - s*grad;
	k = k+1;
	iter = 0;

	%main loop
	if itPrint>0
		fprintf('%s\n', str_head);
	end
	while (lambdaj > lambda) || (mui>mu)
		while k<subiter;
			y = x + (k-1)/(k+2)*(x-xp);
			[f, grad] = Fml(y);
			xp = x;
			x = y - s*grad;
			k = k+1;
			iter = iter+1;
			if itPrint>0
				if mod(iter,itPrint)==0
					 fprintf(str_num, iter, f, mui, lambdaj);
				end
			end
		end
		% change mui and lambdaj
		mui = max(mu, cont_alpha*(min(mui, max(ATA*x-ATb))));
		lambdaj = max(lambdaj*lambdaPow, lambda);
		s = min(opts.s,lambdaj);
		k=0;
		xp = x;
		y = x + (k-1)/(k+2)*(x-xp);
		[f, grad] = Fml(y);
		x = y - s*grad;
		k = k+1;
		iter = iter+1;
		if itPrint>0
			if mod(iter,itPrint)==0
				 fprintf(str_num, iter, f, mui, lambdaj);
			end
		end
	end

	while k<finiter
		y = x + (k-1)/(k+2)*(x-xp);
		[f, grad] = Fml(y);
		xp = x;
		x = y - s*grad;
		k = k+1;
		iter = iter+1;
		if itPrint>0
			if mod(iter,itPrint)==0
				 fprintf(str_num, iter, f, mui, lambdaj);
			end
		end
	end

	out.val = F(x);

	function [Fxx]=F(xx)
		Fxx = 0.5*norm(A*xx-b)^2+mui*norm(xx,1);
	end

	function [Fxx, gFxx]=Fml(xx)
		grad = ATA*x-ATb;
		Fxx = 0.5*((grad-ATb)'*x + bTb) + mui*Huber(xx, lambdaj);
		gFxx = grad + mui*gHuber(xx, lambdaj);
	end

	function [Hxx]=Huber(xx, ll)
		Hxx=sum((xx.^2/(2*ll)-abs(xx)+ll/2).*(abs(xx)<ll))+norm(xx,1)-n*ll/2;
	end

	function [gHxx]=gHuber(xx, ll)
		gHxx=xx/ll.*(abs(xx)<ll)+sign(xx).*(abs(xx)>=ll);
	end

end