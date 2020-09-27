using Random
using StatsFuns
using Plots
using Turing

# True parameters
const β_true = 0.015008
const p0_true = 0.01
const Γ_true = 99.0
const p_true = vcat(ones(100) * p0_true, [0.01, 0.010149677104094834, 0.01030157121960764, 0.01045571448460294, 0.010612139527349752, 0.010770879345959944, 0.010931967381624325, 0.011095437562033569, 0.011261324301378207, 0.011429662500348627, 0.011600487546135083, 0.011773835312427683, 0.011949742159416387, 0.012128244933791046, 0.012309380968741323, 0.012493188083956797, 0.012679704588429742, 0.012868969573904944, 0.01306102245604069, 0.01325590299048054, 0.013453651525559869, 0.01365430900230586, 0.013857916954437517, 0.014064517508365664, 0.01427415338319293, 0.014486867890713772, 0.014702704935414451, 0.014921709014473043, 0.015143925217759454, 0.015369399227835397, 0.015598177319954387, 0.01583030636206179, 0.016065833814794726, 0.01630480773148223, 0.01654727675814502, 0.016793290133495783, 0.01704289768893887, 0.017296149848570547, 0.017553097629332104, 0.017813793323521863, 0.01807828984383767, 0.0183466402058569, 0.018618898177224206, 0.01889511827765156, 0.019175355778918202, 0.019459666704870683, 0.01974810783142283, 0.020040736686555777, 0.020337611550317946, 0.020638791454825042, 0.020944336184260073, 0.021254306274873336, 0.021568763014982423, 0.021887768444972224, 0.02221138535729492, 0.02253967729646995, 0.022872708559084094, 0.023210544193791427, 0.023553250001313236, 0.023900892534438237, 0.024253539098022286, 0.024611257748988674, 0.02497411729632784, 0.025342187301097635, 0.02571553807642323, 0.026094240846014645, 0.026478368420834467, 0.026867994013378065, 0.02726319164508419, 0.027664036247390097, 0.028070603661731572, 0.028482970639542922, 0.028901214842256977, 0.029325414841305058, 0.029755650118117066, 0.030192001064121367, 0.030634548980744872, 0.031083376079413, 0.03153856548154972, 0.03200020121857749, 0.03246836823191731, 0.032943152372988664, 0.033424640403209624, 0.03391291999399669, 0.03440807972676498, 0.03491020909292809, 0.035419398493898115, 0.03593573924108562, 0.03645932355589991, 0.03699024456974855, 0.03752859632403779, 0.03807447377017233, 0.0386279727695554, 0.039189190093588795, 0.03975822342367261, 0.04033517135120606, 0.04092013337758594, 0.041513209914208475, 0.04211450152401414, 0.042724107234636134, 0.043342131827775694, 0.04396868162840564, 0.044603863789412446, 0.04524778629159621, 0.045900557943670706, 0.046562288382263325, 0.04723308807191511, 0.047913068305080776, 0.04860234120212864, 0.04930101971134076, 0.05000921760891267, 0.05072704949895371, 0.0514546308134868, 0.052192077812448494, 0.052939507583689066, 0.053697038042972295, 0.05446478793397573, 0.05524287682829057, 0.05603142512542158, 0.05683055405278719, 0.05764038566571952, 0.05846104284746437, 0.059292649309181, 0.06013532958994251, 0.06098920905673564, 0.06185441390446071, 0.06273107115593152, 0.06361930866187583, 0.06451925510093486, 0.06543103997966371, 0.0663547936325307, 0.0672906472219182, 0.06823873273812185, 0.0691991829993512, 0.07017213165172943, 0.07115771316929323, 0.07215606285399334, 0.07316729612588209, 0.0741915341819413, 0.07522892181179185, 0.07627960347449453, 0.07734372329829686, 0.07842142508063314, 0.07951285228812432, 0.08061814805657816, 0.08173745519098904, 0.08287091616553824, 0.08401867312359354, 0.0851808678777097, 0.08635764190962801, 0.08754913637027652, 0.08875549207977015, 0.08997684952741038, 0.09121334887168549, 0.09246512994027047, 0.09373233223002714, 0.09501509490700391, 0.09631355680643577, 0.09762785643274499, 0.09895813195954, 0.10030452122961625, 0.10166716175495574, 0.10304619071672741, 0.10444174496528673, 0.10585396102017623, 0.10728297507012446, 0.1087289229730475, 0.1101919402560481, 0.1116721621154147, 0.1131697234166237, 0.11468475869433782, 0.11621740215240646, 0.11776778766386538, 0.11933604877093773, 0.1209223186850333, 0.12252673028674767, 0.1241494161258645, 0.1257905084213534, 0.12745013906137084, 0.12912843960325965, 0.13082554127355028, 0.13254156889587373, 0.13427656383132638, 0.1360306350934822, 0.13780392052275764, 0.13959655413694563, 0.14140866613121555, 0.1432403828781132, 0.14509182692756079, 0.14696311700685694, 0.14885436802067678, 0.1507656910510718, 0.15269719335746992, 0.15464897837667557, 0.15662114572286956, 0.1586137911876091, 0.16062700673982777, 0.16266088052583577, 0.16471549686931966, 0.16679093627134228])

# Data
const n = 200

const W = [1.0, 1.0, 1.0, 2.0, 4.0, 4.0, 3.0, 2.0, 2.0, 2.0, 0.0, 4.0, 3.0, 2.0, 2.0, 
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
27.0, 36.0, 34.0, 29.0, 22.0, 17.0, 29.0, 28.0, 23.0, 39.0, 20.0, 28.0, 31.0, 23.0, 37.0, 31.0, 39.0, 49.0]

const t = collect(0.0:1.0:299.0)

const n_samples = length(W)

function epidemic_prediction(β, p0, Γ, t)
    max.(p0, logistic.(β .* t .- β .* Γ .+ logit.(p0)))
end

function epidemic_loglikeli(β, p0, Γ, n, W, t) 
    pe = max.(p0, logistic.(β .* t .- β * Γ .+ logit(p0)))
    return sum(W .* log.(pe) .+ (n .- W) .* log.(1 .- pe))
end

# Model
@model logistic_epidemic(n, W, t) = begin
    β ~ Beta(1, 1) # transmission rate
    p0 ~ Beta(1, 1) # initial prevalence
    Γ ~ Uniform(0, t[end]) # Geometric(0.01) # epidemic start time
    z = logit(p0)
    for (i, ti) in enumerate(t)
        # W[i] ~ Binomial(n, max(p0, logistic.(β * ti - β * Γ + z)))
        W[i] ~ BinomialLogit(n, max(0, β * (ti - Γ)) + z)
    end
end

# Inference
chain = sample(logistic_epidemic(n, W, t), NUTS(1000, 0.95), 2000)

p_samples = epidemic_prediction(chain[:β], chain[:p0], chain[:Γ], 350.0)

# advi = ADVI(10, 1000)
# q = vi(logistic_epidemic(n, W, t), advi)