function f(a::Int64)
    
    ϕ1(α) = let a = a
        α -> ϕ(α, a)
    end

    a += 1

    return a
end

function ϕ(α, a::Int64)
    α * a
end

@code_warntype f(10)