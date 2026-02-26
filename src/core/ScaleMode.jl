
# ============================================================
# Value Scaling Modes
# ============================================================

abstract type ValueScaling end
struct Levels <: ValueScaling end
struct Detrended <: ValueScaling end


# ============================================================
# Flow Scaling
# ============================================================

scale_flow(::Levels, flow, A_i) = flow
scale_flow(::Detrended, flow, A_i) = flow / A_i


# ============================================================
# Kernel Scaling
# ============================================================

scale_kernel(::Levels, sdf, A_old, A_new) = sdf
scale_kernel(::Detrended, sdf, A_old, A_new) = sdf * (A_new / A_old)


# String representation for scaling modes
convert_to_string(::Levels) = "Levels()"
convert_to_string(::Detrended) = "Detrended()"