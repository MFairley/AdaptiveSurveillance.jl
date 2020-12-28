function f(a::Int64)
    
    ϕ1 = let a = a
        α -> ϕ(α, a)
    end

    i += 1
    while i < 10
        a += 1
    end

    return a
end

function ϕ(α::Int64, a::Int64)
    α * a
end

@code_warntype f(10)