# Index-map builders extracted from main_alch.jl

function build_index_maps(sys, processed_atom_types::Dict{String, Int})  
    n_atoms = length(sys.atoms)  
    idx_mass   = zeros(Int, n_atoms)  
    idx_σ      = zeros(Int, n_atoms)  
    idx_ϵ      = zeros(Int, n_atoms)
    
    for i in 1:n_atoms  
        atype = String(sys.atoms_data[i].atom_type)  
        if haskey(processed_atom_types, atype)  
            base_idx      = processed_atom_types[atype]  
            idx_σ[i]      = base_idx  
            idx_ϵ[i]      = base_idx + 1
        end  
    end  
    return ( 
        (idx_mass, idx_σ, idx_ϵ),  
        Tuple(() for _ in 1:length(sys.pairwise_inters)),  
        Tuple(() for _ in 1:length(sys.specific_inter_lists)),  
        Tuple(() for _ in 1:length(sys.general_inters))  
    )  
end  
