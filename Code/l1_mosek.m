function [x,out]=l1_mosek(x0, A, b, mu, opts3);
	[m, n] = size(A);
	ATA = A'*A;
	mu_m_ATb = -A'*b+mu*ones(n,1);
	mu_p_ATb = A'*b+mu*ones(n,1);
	bTb = b'*b;
	A_tmp = cat(2,ATA,-ATA);
	q = cat(1,A_tmp,-A_tmp);
	c = cat(1,mu_m_ATb,mu_p_ATb);
	a = ones(1,2*n);
	blc = [];
	buc = [];
	blx = sparse(2*n,1);
	bux = [];
	[res] = mskqpopt(q,c,a,blc,buc,blx,bux);
	xx = res.sol.itr.xx;
	x = xx(1:n)-xx(n+1:2*n);
	out = [];
	out.res = res;
	out.val = 0.5*norm(A*x-b)^2+mu*norm(x,1);


end