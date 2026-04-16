"""
    map_phi_to_theta(phi_vec, t_min, t_max, p_0, k_sig)

Backward-compatible bounded-sigmoid transform used by the LJ-only path.
"""
function map_phi_to_theta(phi_vec, t_min, t_max, p_0, k_sig)
    families = fill(:sigma, length(phi_vec))
    return map_phi_to_theta(phi_vec, t_min, t_max, p_0, k_sig, families)
end

"""
    map_phi_to_theta(phi_vec, t_min, t_max, p_0, k_sig, families)

Map unconstrained optimization variables `ϕ` into physical parameters `θ` using
family-specific transforms.
"""
function map_phi_to_theta(phi_vec, t_min, t_max, p_0, k_sig, families)
    FT = eltype(phi_vec)
    theta = similar(phi_vec)

    for i in eachindex(phi_vec)
        family = families[i]
        if family == :sigma || family == :epsilon
            theta[i] = t_min[i] + (t_max[i] - t_min[i]) / (FT(1.0) + exp(-k_sig * (phi_vec[i] - p_0[i])))
        elseif family == :charge_chi
            theta[i] = phi_vec[i] + p_0[i]
        elseif family == :charge_eta
            theta[i] = t_min[i] + log1p(exp(phi_vec[i] - p_0[i]))
        else
            throw(ArgumentError("Unsupported parameter family `$family` in map_phi_to_theta."))
        end
    end

    return theta
end

"""
    get_chain_rule_multiplier(theta_vec, t_min, t_max, k_sig)

Backward-compatible derivative for the bounded LJ transform.
"""
function get_chain_rule_multiplier(theta_vec, t_min, t_max, k_sig)
    families = fill(:sigma, length(theta_vec))
    phi_vec = copy(theta_vec)
    p_0 = zeros(eltype(theta_vec), length(theta_vec))
    return get_chain_rule_multiplier(phi_vec, theta_vec, t_min, t_max, p_0, k_sig, families)
end

"""
    get_chain_rule_multiplier(phi_vec, theta_vec, t_min, t_max, p_0, k_sig, families)

Return `∂θ/∂ϕ` for the family-specific transforms used by `map_phi_to_theta`.
"""
function get_chain_rule_multiplier(phi_vec, theta_vec, t_min, t_max, p_0, k_sig, families)
    FT = eltype(theta_vec)
    multiplier = similar(theta_vec)

    for i in eachindex(theta_vec)
        family = families[i]
        if family == :sigma || family == :epsilon
            multiplier[i] = k_sig * (theta_vec[i] - t_min[i]) * (FT(1.0) - (theta_vec[i] - t_min[i]) / (t_max[i] - t_min[i]))
        elseif family == :charge_chi
            multiplier[i] = one(FT)
        elseif family == :charge_eta
            multiplier[i] = inv(FT(1.0) + exp(-(phi_vec[i] - p_0[i])))
        else
            throw(ArgumentError("Unsupported parameter family `$family` in get_chain_rule_multiplier."))
        end
    end

    return multiplier
end
