using Random, LinearAlgebra, SpecialFunctions, PyPlot, KernelDensity, PyCall
stats = pyimport("scipy.stats")

"Generate (m, n) matrix uniformly distributed over Stiefel manifold."
function rando(m::Integer, n::Integer)
    M = randn((m, n))
    Q = Matrix(qr(M).Q)
end

"Generate matrix with exponentially-distributed singular values and singular vectors ~ O(n)."
function randA(n::Integer, scale::Real)
    s = randexp(n)*scale
    V = rando(n, n)
    U = rando(n, n)
    A = U*Diagonal(s)*transpose(V)
    return A, s, U, V
end

"Generate random symmetric matrix with exponentially-distributed eigenvalues and eigenvectors ~ O(n)."
function randS(n::Integer, scale::Real)
    s = randexp(n)*scale
    U = rando(n, n)
    A = U*Diagonal(s)*transpose(U)
    return A, s, U
end

"Generate solution vector from x0 + V N(0, I) + Y G N(0, I)."
function randx(x0::Array{<:Real, 1}, V::Array{<:Real, 2}, Y::Array{<:Real, 2}, G::Array{<:Real, 2})
    v = randn(size(V)[2])
    y = randn(size(Y)[2])
    x = x0 + V*v + Y*G*y
end

"Generate solution vector from x0 + N(0, I)."
function randx(x0::Array{<:Real, 1})
    x = x0 + randn(size(x0))
end

"Generate solution vector from x0 + L N(0, I)."
function randx(x0::Array{<:Real, 1}, L::Array{<:Real, 2})
    x = x0 + L*randn(size(L)[2])
end

"Naive implementation of projection method."
function projection(x0::Array{<:Real, 1}, A::Array{<:Real, 2}, b::Array{<:Real, 1}, V::Array{<:Real, 2}, W::Array{<:Real, 2})
    x = x0 + V*inv(transpose(W)*A*V)*transpose(W)*(b - A*x0)
end

"Krylov subspace K_{n}(A, v)."
function Krylov(A::Array{<:Real, 2}, v::Array{<:Real, 1}, n::Integer)
    V = Array{Float64, 2}(undef, size(v)[1], n)
    V[:, 1] = v/norm(v)
    for i=2:n
        V[:, i] = A*V[:, i-1]
        for j=1:i-1
            V[:, i] = V[:, i] - dot(V[:, i], V[:, j])*V[:, j]
        end
        V[:, i] = V[:, i]/norm(V[:, i])
    end
    return V
end

"PDF of chi-squared distribution."
function chi2PDF(x::Real, n::Integer)
    return x^(n/2-1)*exp(-x/2)/(2^(n/2)*gamma(n/2))
end

"PDF of F-distribution."
function FPDF(x::Real, d1::Integer, d2::Integer)
    return (d1/d2)^(d1/2)*x^(d1/2-1)*(1+d1*x/d2)^(-(d1+d2)/2)/beta(d1/2, d2/2)
end

"Compute orthonormal basis for Null(A), where A is full-rank MxN matrix with N>M."
function NullSpace(A::Array{<:Real, 2})
    svdA = svd(A, full=true)
    return svdA.V[:, (size(A)[1]+1):size(A)[2]]
end

"Approximation of int dx|F1-F2| with central Riemann sum."
function Integrate(F1::Array{<:Real, 1}, F2::Array{<:Real, 1}, h::Real)
    F = abs.(F1 - F2)
    s = h*(sum(F[2:(size(F)[1]-1)]) + (F[1] + F[size(F)[1]])/2)
end

"Symmetric matices, separate estimation of the scale for each run, mean of inverse gamma."
function SymSamplingIn(N_rounds::Integer, scale::Real, m::Integer,  N::Integer=100, N_samples::Integer=5000; randomization="A")
    A, s, U = randS(N, scale)
    L = U*Diagonal(s.^(-1/2))
    inv_A = U*Diagonal(s.^(-1))*transpose(U)
    Statistics_inv = zeros(N_samples)
    Statistics_proj = zeros(N_samples)
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
        statistics = transpose(x - y)*pinv(sigma)*(x - y)
        Statistics_inv[i] = statistics

        alpha = 0
        for i=1:N_rounds
            z = draw()
            b = A*z
            V = Krylov(A, b, m)
            P = I - V*inv(transpose(V)*A*V)*transpose(V)*A
            sample = P*z
            alpha = alpha + norm(sample)^2
        end
        alpha = alpha/(N_rounds*(N-m)+2)

        statistics = (norm(x - y)^2)/alpha
        Statistics_proj[i] = statistics
    end
    x = LinRange(0, 500, 5000)
    kernel_inv = stats.gaussian_kde(Statistics_inv.*(Statistics_inv.<500).*(Statistics_inv.>0))
    kernel_proj = stats.gaussian_kde(Statistics_proj.*(Statistics_proj.<500))
    stat_inv = kernel_inv(x)
    stat_proj = kernel_proj(x)
    return stat_inv, stat_proj
end

"Symmetric matices, the single estimation of the scale for all run, mean of inverse gamma."
function SymSamplingOut(N_rounds::Integer, scale::Real, m::Integer,  N::Integer=100, N_samples::Integer=5000; randomization="A")
    A, s, U = randS(N, scale)
    L = U*Diagonal(s.^(-1/2))
    inv_A = U*Diagonal(s.^(-1))*transpose(U)
    Statistics_inv = zeros(N_samples)
    Statistics_proj = zeros(N_samples)
    if randomization == "A"
        draw = (n=N, l=L) -> randx(zeros(n), l)
    else
        draw = (n=N) -> randx(zeros(n))
    end

    alpha = 0
    for i=1:N_rounds
        z = draw()
        b = A*z
        V = Krylov(A, b, m)
        P = I - V*inv(transpose(V)*A*V)*transpose(V)*A
        sample = P*z
        alpha = alpha + norm(sample)^2
    end
    alpha = alpha/(N_rounds*(N-m)+2)

    for i=1:N_samples
        x = draw()
        b = A*x
        V = Krylov(A, b, m)
        y = projection(zeros(N), A, b, V, V)
        sigma = inv_A - V*inv(transpose(V)*A*V)*transpose(V)
        statistics = transpose(x - y)*pinv(sigma)*(x - y)
        Statistics_inv[i] = statistics

        statistics = (norm(x - y)^2)/alpha
        Statistics_proj[i] = statistics
    end
    x = LinRange(0, 500, 5000)
    kernel_inv = stats.gaussian_kde(Statistics_inv.*(Statistics_inv.<500).*(Statistics_inv.>0))
    kernel_proj = stats.gaussian_kde(Statistics_proj.*(Statistics_proj.<500))
    stat_inv = kernel_inv(x)
    stat_proj = kernel_proj(x)
    return stat_inv, stat_proj
end


"Symmetric matices, separate estimation of the scale for each run, multivariate Student."
function HSymSamplingIn(N_rounds::Integer, scale::Real, m::Integer,  N::Integer=100, N_samples::Integer=5000; randomization="A")
    A, s, U = randS(N, scale)
    L = U*Diagonal(s.^(-1/2))
    inv_A = U*Diagonal(s.^(-1))*transpose(U)
    Statistics_proj = zeros(N_samples)
    if randomization == "A"
        draw = (n=N, l=L) -> randx(zeros(n), l)
    else
        draw = (n=N) -> randx(zeros(n))
    end

    for i=1:N_samples
        alpha = 0
        for i=1:N_rounds
            z = draw()
            b = A*z
            V = Krylov(A, b, m)
            P = I - V*inv(transpose(V)*A*V)*transpose(V)*A
            sample = P*z
            alpha = alpha + norm(sample)^2
        end
        alpha = alpha/N_rounds

        x = draw()
        b = A*x
        V = Krylov(A, b, m)
        y = projection(zeros(N), A, b, V, V)
        statistics = (norm(x - y)^2)/alpha
        Statistics_proj[i] = statistics
    end

    x = LinRange(0, 10, 5000)
    kernel_proj = stats.gaussian_kde(Statistics_proj.*(Statistics_proj.<10))
    stat_proj = kernel_proj(x)
    return stat_proj
end

"Symmetric matices, the single estimation of the scale for all run, multivariate Student."
function HSymSamplingOut(N_rounds::Integer, scale::Real, m::Integer,  N::Integer=100, N_samples::Integer=5000; randomization="A")
    A, s, U = randS(N, scale)
    L = U*Diagonal(s.^(-1/2))
    inv_A = U*Diagonal(s.^(-1))*transpose(U)
    Statistics_proj = zeros(N_samples)
    if randomization == "A"
        draw = (n=N, l=L) -> randx(zeros(n), l)
    else
        draw = (n=N) -> randx(zeros(n))
    end

    alpha = 0
    for i=1:N_rounds
        z = draw()
        b = A*z
        V = Krylov(A, b, m)
        P = I - V*inv(transpose(V)*A*V)*transpose(V)*A
        sample = P*z
        alpha = alpha + norm(sample)^2
    end
    alpha = alpha/N_rounds

    for i=1:N_samples
        x = draw()
        b = A*x
        V = Krylov(A, b, m)
        y = projection(zeros(N), A, b, V, V)
        statistics = (norm(x - y)^2)/alpha
        Statistics_proj[i] = statistics
    end

    x = LinRange(0, 10, 5000)
    kernel_proj = stats.gaussian_kde(Statistics_proj.*(Statistics_proj.<10))
    stat_proj = kernel_proj(x)
    return stat_proj
end
