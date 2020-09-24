include("UQ.jl")

"""
    SSampling(m::Integer, scale::Real=10,  N::Integer=100, N_samples::Integer=20, N_matrices::Integer=500; randomization="A")

    This function gather samples of (x_exact - x_approximate)^{T} \Sigma^{-1} (x_exact - x_approximate).
    The exact solution is sampled either from x ~ N(0, I) (`randomization`= anything but "A") or x ~ N(0, A^{-1}) (`randomization`="A").
    The matrix A = U D U^{T} drawn as follows. U ~ uniform over O(N), D_{ii} ~ exp distribution sith scale `s`
    `N` fixes dimension, `N_samples` -- number of x generated for each matrix, `N_matrices` -- number of matrices generated.

    (see the article, for more details)
    The output `Statistics` is an array of the size (5, `N_samples`*`N_matrices`).
    `Statistics[1, :]` -- samples, correspond to the mode of inverse gamma distribution (with additional observation) [compare with chi^2]
    `Statistics[2, :]` -- samples, correspond to the mode of inverse gamma distribution (without additional observation) [compare with chi^2]
    `Statistics[3, :]` -- samples, correspond to the hierarchical modelling (with additional observation) [compare with F]
    `Statistics[4, :]` -- samples, correspond to the hierarchical modelling (without additional observation) [compare with F]
    `Statistics[5, :]` -- \Sigma = A^{-1} - V (V^T A V)^-1 V^T, i.e. Bayesian CG [compare with chi^2]
"""
function SSampling(m::Integer, scale::Real=10,  N::Integer=100, N_samples::Integer=20, N_matrices::Integer=500; randomization="A")
    Statistics = zeros(5, N_samples*N_matrices)
    for j=1:N_matrices
        A, s, U = randS(N, scale)
        inv_A = U*Diagonal(s.^(-1))*transpose(U)
        L = U*Diagonal(s.^(-1/2))
        if randomization == "A"
            draw = (n=N, l=L) -> randx(zeros(n), l)
        else
            draw = (n=N) -> randx(zeros(n))
        end

        for i=1:N_samples
            x = draw()
            b = A*x
            V = Krylov(A, b, m)
            y = projection(zeros(N), A, b, V, V)
            sigma = inv_A - V*inv(transpose(V)*A*V)*transpose(V)
            eps = 0
            k = 0
            while k == 0
                try
                    k += 1
                    Statistics[5, i + (j-1)*N_samples] = transpose(x - y)*pinv(sigma + I*eps)*(x - y)
                catch LAPACKException
                    k -= 1
                    eps = 1e-6
                end
            end

            delta = inv(transpose(V)*A*V)*transpose(V)*b
            alpha_2 = m
            beta_2 = norm(delta)^2
            s_2 = beta_2/(alpha_2 + 2)

            z = draw()
            b = A*z
            V = Krylov(A, b, m)
            P = I - V*inv(transpose(V)*A*V)*transpose(V)*A
            sample = P*z
            alpha_1 = (N-m)
            beta_1 = norm(sample)^2
            s_1 = beta_1/(alpha_1 + 2)

            squared_error = norm(x - y)^2
            Statistics[1, i + (j-1)*N_samples] = squared_error/s_1
            Statistics[2, i + (j-1)*N_samples] = squared_error/s_2
            Statistics[3, i + (j-1)*N_samples] = (alpha_1/beta_1)*squared_error/(N-m)
            Statistics[4, i + (j-1)*N_samples] = (alpha_2/beta_2)*squared_error/(N-m)
        end
    end
    return Statistics
end

"""
    ASampling(m::Integer, scale::Real=10,  N::Integer=100, N_samples::Integer=20, N_matrices::Integer=500; randomization="A")

    This function gather samples of (x_exact - x_approximate)^{T} \Sigma^{-1} (x_exact - x_approximate).
    The exact solution is sampled either from x ~ N(0, I) (`randomization`= anything but "A") or b ~ N(0, I) (`randomization`="A").
    The matrix A = U D V^{T} drawn as follows. U ~ uniform over O(N), V ~ uniform over O(N), D_{ii} ~ exp distribution sith scale `s`
    `N` fixes dimension, `N_samples` -- number of x generated for each matrix, `N_matrices` -- number of matrices generated.

    (see the article, for more details)
    The output `Statistics` is an array of the size (5, `N_samples`*`N_matrices`).
    `Statistics[1, :]` -- samples, correspond to the mode of inverse gamma distribution (with additional observation) [compare with chi^2]
    `Statistics[2, :]` -- samples, correspond to the mode of inverse gamma distribution (without additional observation) [compare with chi^2]
    `Statistics[3, :]` -- samples, correspond to the hierarchical modelling (with additional observation) [compare with F]
    `Statistics[4, :]` -- samples, correspond to the hierarchical modelling (without additional observation) [compare with F]
    `Statistics[5, :]` -- \Sigma = A^{-1} - V (V^T A V)^-1 V^T, i.e. Bayesian CG [compare with chi^2]
"""
function ASampling(m::Integer, scale::Real=10,  N::Integer=100, N_samples::Integer=20, N_matrices::Integer=500; randomization="A", symmetric=true)
    Statistics = zeros(5, N_samples*N_matrices)
    for j=1:N_matrices
        if symmetric
            A, s, U = randS(N, scale)
            L = U*Diagonal(s.^(-1))*transpose(U)
            inv_A = L*transpose(L)
        else
            A, s, U, V = randA(N, scale)
            L = V*Diagonal(s.^(-1))*transpose(U)
            inv_A = L*transpose(L)
        end

        if randomization == "A"
            draw = (n=N, l=L) -> randx(zeros(n), l)
        else
            draw = (n=N) -> randx(zeros(n))
        end

        for i=1:N_samples
            x = draw()
            b = A*x
            V = Krylov(A, b, m)
            y = projection(zeros(N), A, b, V, A*V)
            sigma = inv_A - V*inv(transpose(A*V)*A*V)*transpose(V)
            eps = 0
            k = 0
            while k == 0
                try
                    k += 1
                    Statistics[5, i + (j-1)*N_samples] = transpose(x - y)*pinv(sigma + I*eps)*(x - y)
                catch LAPACKException
                    k -= 1
                    eps = 1e-6
                end
            end

            delta = inv(transpose(A*V)*A*V)*transpose(A*V)*b
            alpha_2 = m
            beta_2 = norm(delta)^2
            s_2 = beta_2/(alpha_2 + 2)

            z = draw()
            b = A*z
            V = Krylov(A, b, m)
            P = I - V*inv(transpose(A*V)*A*V)*transpose(A*V)*A
            sample = P*z
            beta_1 = norm(sample)^2
            alpha_1 = (N-m)
            s_1 = beta_1/(alpha_1 + 2)

            squared_error = norm(x - y)^2
            Statistics[1, i + (j-1)*N_samples] = squared_error/s_1
            Statistics[2, i + (j-1)*N_samples] = squared_error/s_2
            Statistics[3, i + (j-1)*N_samples] = (alpha_1/beta_1)*squared_error/(N-m)
            Statistics[4, i + (j-1)*N_samples] = (alpha_2/beta_2)*squared_error/(N-m)
        end
    end
    return Statistics
end
