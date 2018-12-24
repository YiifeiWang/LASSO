function [x,out]=l1_cvx_gurobi(x0, A, b, mu, opts1)
	tic; 
	size_A = size(A);
	n = size_A(2);
	cvx_solver gurobi
	cvx_begin quiet
	    variable x(n)
	    minimize(0.5* sum_square( A * x - b ) + mu * norm( x, 1 ) )
	cvx_end

	t1 = toc;
	out=[];
	out.t = t1;
	out.val = 0.5* sum_square( A * x - b ) + mu * norm( x, 1 );
end 