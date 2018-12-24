function [x,out]=l1_gurobi(x0, A, b, mu, opts3);
	[m, n] = size(A);
	ATA = A'*A;
	mu_m_ATb = -A'*b+mu*ones(n,1);
	mu_p_ATb = A'*b+mu*ones(n,1);
	A_tmp = cat(2,ATA,-ATA);
	model = [];
	model.Q = 0.5*sparse(cat(1,A_tmp,-A_tmp));
	model.obj = cat(1,mu_m_ATb,mu_p_ATb);
	model.A = sparse(eye(2*n));
	model.sense = '>';
	model.rhs = zeros(2*n,1);
	params.BarConvTol = 1e-12;
	gurobi_write(model, 'qp.lp');
	results = gurobi(model, params);
	xx = results.x;
	x = xx(1:n)-xx(n+1:2*n);
	out = [];
	out.results = results;
	out.val = 0.5*norm(A*x-b)^2+mu*norm(x,1);

end