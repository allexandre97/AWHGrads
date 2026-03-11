function default_simulation_config(; FT::DataType=Float32, AT=CuArray, device_id::Int=1)
    return SimulationConfig(
        device_id = device_id,
        FT = FT,
        AT = AT,
        Δt = FT(1)u"fs",
        T0 = FT(310)u"K",
        P0 = FT(1)u"bar",
        lambda_schedule = FT.(range(1.0, stop=0.0, length=21)),
        awh_budget_time = FT(20)u"ns",
        awh_block_time = FT(1.0)u"ns",
        awh_probe_time_solv = FT(0.75)u"ns",
        awh_probe_time_vac = FT(0.25)u"ns",
        md_time_production = FT(0.1)u"ns",
    )
end

function default_optimization_config(; FT::DataType=Float32)
    return OptimizationConfig(
        awh_convergence_tol = FT(1e-3),
        rewarm_fraction = FT(0.05),
        huber_delta = FT(2.0),
        kl_target = FT(0.1),
        eigenvalue_tol_scale = FT(1e-2),
        min_phi_step = FT(5e-4),
        max_phi_step_solute = FT(0.35),
        max_phi_step_solvent = FT(0.035),
        tiny_alpha_cutoff = FT(0.015625),
        restart_rmsd_tol_nm = FT(1e-5),
        ess_threshold_scale = FT(0.22),
        awh_split_tol_kT = FT(0.5),
        awh_parity_tol_kT = FT(0.1),
        awh_endpoint_min_fraction = FT(0.03),
        k_sigmoid = FT(1.0),
    )
end

function apply_simulation_config!(cfg::SimulationConfig)
    global FT = cfg.FT
    global AT = cfg.AT
    global Δt = cfg.Δt
    global T0 = cfg.T0
    global P0 = cfg.P0
    global lambda_schedule = cfg.lambda_schedule
    global num_lambda_states = length(cfg.lambda_schedule)
    global target_rho = FT(1.0 / num_lambda_states)

    data_dir = joinpath(dirname(pathof(Molly)), "..", "data")
    ff_dir = joinpath(data_dir, "force_fields")
    global ff = MolecularForceField(FT, joinpath.(ff_dir, ["tip3p_standard.xml", "gaff.xml", "ethanol.xml"])...; units=true)

    device!(cfg.device_id)
    return nothing
end
