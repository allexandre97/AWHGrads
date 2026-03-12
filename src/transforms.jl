"""
    map_phi_to_theta(phi_vec, t_min, t_max, p_0, k_sig)

Map unconstrained optimization variables `ϕ` into bounded physical parameters
`θ` using a shifted sigmoid.
"""
function map_phi_to_theta(phi_vec, t_min, t_max, p_0, k_sig)
    FT = eltype(phi_vec)
    return t_min .+ (t_max .- t_min) ./ (FT(1.0) .+ exp.(-k_sig .* (phi_vec .- p_0)))
end

"""
    get_chain_rule_multiplier(theta_vec, t_min, t_max, k_sig)

Return `∂θ/∂ϕ` for the sigmoid transform used by `map_phi_to_theta`.
"""
function get_chain_rule_multiplier(theta_vec, t_min, t_max, k_sig)
    FT = eltype(theta_vec)
    return k_sig .* (theta_vec .- t_min) .* (FT(1.0) .- (theta_vec .- t_min) ./ (t_max .- t_min))
end
