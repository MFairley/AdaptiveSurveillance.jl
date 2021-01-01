# Test Data
# Logistic
const β_true = 0.015008
const p0_true = 0.01
const Γ_true = 100
const n = 200
const W = Int.([1.0, 1.0, 1.0, 2.0, 4.0, 4.0, 3.0, 2.0, 2.0, 2.0, 0.0, 4.0, 3.0, 2.0, 2.0, 
2.0, 4.0, 2.0, 1.0, 3.0, 3.0, 1.0, 3.0, 1.0, 3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 0.0, 2.0, 8.0, 2.0, 5.0, 
3.0, 0.0, 2.0, 2.0, 4.0, 1.0, 3.0, 2.0, 5.0, 2.0, 5.0, 3.0, 3.0, 3.0, 1.0, 1.0, 3.0, 1.0, 1.0, 3.0, 
2.0, 1.0, 2.0, 3.0, 1.0, 4.0, 6.0, 2.0, 2.0, 1.0, 2.0, 1.0, 4.0, 2.0, 0.0, 1.0, 2.0, 1.0, 3.0, 3.0, 
2.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 5.0, 3.0, 0.0, 2.0, 1.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0, 1.0, 3.0, 
3.0, 2.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 3.0, 1.0, 4.0, 1.0, 1.0, 3.0, 1.0, 2.0, 4.0, 2.0, 1.0, 1.0, 
4.0, 1.0, 0.0, 2.0, 1.0, 5.0, 2.0, 3.0, 3.0, 4.0, 0.0, 6.0, 4.0, 3.0, 4.0, 3.0, 1.0, 4.0, 5.0, 5.0, 
1.0, 2.0, 5.0, 4.0, 4.0, 4.0, 7.0, 3.0, 2.0, 2.0, 7.0, 5.0, 3.0, 2.0, 3.0, 4.0, 4.0, 3.0, 6.0, 1.0, 
6.0, 4.0, 1.0, 4.0, 8.0, 2.0, 2.0, 3.0, 3.0, 2.0, 4.0, 3.0, 8.0, 2.0, 7.0, 7.0, 5.0, 8.0, 9.0, 6.0, 
8.0, 8.0, 4.0, 5.0, 8.0, 2.0, 8.0, 4.0, 6.0, 6.0, 13.0, 9.0, 12.0, 8.0, 6.0, 8.0, 4.0, 5.0, 9.0, 4.0, 
9.0, 8.0, 4.0, 7.0, 10.0, 11.0, 12.0, 8.0, 11.0, 9.0, 12.0, 12.0, 13.0, 13.0, 10.0, 10.0, 11.0, 15.0, 
12.0, 11.0, 4.0, 7.0, 7.0, 9.0, 13.0, 11.0, 13.0, 16.0, 11.0, 9.0, 16.0, 12.0, 11.0, 15.0, 11.0, 9.0, 
16.0, 11.0, 13.0, 16.0, 12.0, 17.0, 11.0, 20.0, 16.0, 16.0, 19.0, 15.0, 12.0, 13.0, 12.0, 14.0, 15.0, 
17.0, 22.0, 20.0, 16.0, 20.0, 17.0, 18.0, 17.0, 19.0, 16.0, 19.0, 17.0, 23.0, 24.0, 22.0, 19.0, 19.0, 
20.0, 19.0, 23.0, 28.0, 20.0, 20.0, 24.0, 26.0, 25.0, 26.0, 21.0, 34.0, 29.0, 28.0, 23.0, 29.0, 28.0, 
27.0, 36.0, 34.0, 29.0, 22.0, 17.0, 29.0, 28.0, 23.0, 39.0, 20.0, 28.0, 31.0, 23.0, 37.0, 31.0, 39.0, 
49.0])
const t = collect(1:length(W))

# Sampling
const L = 2
const β_true_L = ones(Float64, L) * β_true
const p0_true_L = ones(Float64, L) * p0_true
const Γ_true_L = ones(Int64, L) * typemax(Int64)
Γ_true_L[1] = 1
const maxiters = 150
const α = 1000 # the higher, the less false positives