function Test_hw05_03

	% min 0.5 ||Ax-b||_2^2 + mu*||x||_1

	% generate data
	n = 1024;
	m = 512;

	% set the random seed
	rng(9);

	A = randn(m,n);
	u = sprandn(n,1,0.1);
	b = A*u;

	mu = 1e-3;

	x0 = rand(n,1);

	errfun = @(x1, x2) norm(x1-x2)/(1+norm(x1));

	% cvx calling mosek
	opts1 = []; %modify options
	tic; 
	[x1, out1] = l1_cvx_mosek(x0, A, b, mu, opts1);
	t1 = toc;

	% augmented lagrangian method for the dual problem
	opts2 = []; %modify options
	tic; 
	[x2, out2] = l1_dual_ALM(x0, A, b, mu, opts2);
	t2 = toc;

	% ADMM for the dual problem
	opts3 = []; %modify options
	tic; 
	[x3, out3] = l1_dual_ADMM(x0, A, b, mu, opts3);
	t3 = toc;

	% ADMM with linearization for the primal problem
	opts4 = []; %modify options
	tic; 
	[x4, out4] = l1_primal_linear_ADMM(x0, A, b, mu, opts4);
	t4 = toc;

	% ADMM for the primal problem
	opts5 = []; %modify options
	tic; 
	[x5, out5] = l1_primal_ADMM(x0, A, b, mu, opts5);
	t5 = toc;

	% print comparison results with cvx-call-mosek
	fprintf('     cvx-call-mosek: cpu: %5.2f\n', t1);
	fprintf('                ALM: cpu: %5.2f, err-to-cvx-mosek: x %3.2e\n optval %3.2e\n', t2, errfun(x1, x2), (out2.val-out1.val)/out1.val);
	fprintf('             ADMM-d: cpu: %5.2f, err-to-cvx-mosek: x %3.2e\n optval %3.2e\n', t3, errfun(x1, x3), (out3.val-out1.val)/out1.val);
	fprintf('           ADMM-l-p: cpu: %5.2f, err-to-cvx-mosek: x %3.2e\n optval %3.2e\n', t4, errfun(x1, x4), (out4.val-out1.val)/out1.val);
	fprintf('             ADMM-p: cpu: %5.2f, err-to-cvx-mosek: x %3.2e\n optval %3.2e\n', t5, errfun(x1, x5), (out5.val-out1.val)/out1.val);

	
end