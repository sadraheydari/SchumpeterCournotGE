"""
    ProgressBars

A log-distance based progress tracker designed for solvers with
geometric convergence.

This tracker:

- Assumes `log(diff)` decreases approximately linearly
- Computes smooth progress toward a threshold `tol`
- Estimates remaining iterations using local slope
- Displays ETA based on average iteration time
- Renders a colored progress bar in terminal

# Fields
- `tol::Float64`            : convergence tolerance
- `log_tol::Float64`        : log(tol)
- `log_diff0::Float64`      : initial log difference
- `bar_length::Int`         : width of progress bar
- `start_time::Float64`     : solver start timestamp
- `prev_log_diff::Float64`  : previous log difference
- `prev_iter::Int`          : previous iteration index
- `slope_avg::Float64`      : smoothed convergence slope

This tracker assumes geometric convergence.
If convergence is irregular, ETA may fluctuate.
"""
module ProgressBars

using Printf

export ProgressBar, update!, finish!,
         format_bar, render!, MultiTracker,
         YELLOW, GREEN, RED, CYAN, RESET

mutable struct ProgressBar
    max_iter::Int
    tol::Float64
    log_tol::Float64
    log_diff0::Float64
    bar_length::Int
    start_time::Float64
    prev_log_diff::Float64
    prev_iter::Int
    slope_avg::Float64
    color:: String
    initiated:: Bool
end

# ----------------------------------------------------
# ANSI Colors
# ----------------------------------------------------

const RESET  = "\e[0m"
const GREEN  = "\e[32m"
const YELLOW = "\e[33m"
const RED    = "\e[31m"
const CYAN   = "\e[36m"

# ----------------------------------------------------
# Initialization
# ----------------------------------------------------

"""
    init!(tol, diff0; bar_length=40)

Create and initialize a `ProgressBar`.

Must be called after first `diff` is computed.
"""
function ProgressBar(max_iter::Int, tol::Float64; diff0::Float64=Inf, bar_length::Int=40, color::String=GREEN)

    logd = log(diff0)

    _, cols = displaysize(stdout)
    otput_len = 50 # estimated length of non-bar output (iter, percent, diff, ETA)
    bar_length = max(10, min(bar_length, cols - otput_len))

    return ProgressBar(
        max_iter,
        tol,
        log(tol),
        logd,
        bar_length,
        time(),
        logd,
        1,
        0.0,
        color,
        false
    )
end

# ----------------------------------------------------
# Update
# ----------------------------------------------------

"""
        format_eta(seconds)
Format ETA in seconds to "HH:MM:SS.s" format.
"""
function format_eta(seconds::Float64)

    if !isfinite(seconds) || seconds < 0
        return "--:--:--.-"
    end

    h = floor(Int, seconds ÷ 3600)
    m = floor(Int, (seconds % 3600) ÷ 60)
    s = seconds % 60

    return @sprintf("%02d:%02d:%04.1f", h, m, s)
end


"""
    update!(pt, diff, iter)

Update progress display.

Arguments:
- `diff` : current maximum difference
- `iter` : current iteration index

Displays:
- iteration number
- colored progress bar
- percentage
- current diff
- ETA (seconds)

Uses exponential smoothing for slope stability.
"""
function update!(pt::ProgressBar, diff::Float64, iter::Int)

    if !pt.initiated
        pt.initiated = true
        pt.log_diff0 = log(diff)
    end

    log_diff = log(diff)
    pt.log_diff0 = max(pt.log_diff0, log_diff) # update initial log diff if it increases

    # ----- progress in log space
    progress_tol = 1 - (log_diff - pt.log_tol) /
                   (pt.log_diff0 - pt.log_tol)

    progress_tol = clamp(progress_tol, 0.0, 1.0)

    #----- progress in iteration space
    progress_iter = iter / pt.max_iter
    progress_iter = clamp(progress_iter, 0.0, 1.0)

    complete_progress = min(progress_tol, progress_iter)
    max_progress = max(progress_tol, progress_iter)

    filled = round(Int, complete_progress * pt.bar_length)
    half_filled = round(Int, (max_progress - complete_progress) * pt.bar_length)
    empty  = pt.bar_length - filled - half_filled

    halph_bar = progress_iter < progress_tol ? "▀" : "▄"
    bar = " " *
          repeat("█", filled) *
          repeat(halph_bar, half_filled) *
          repeat(" ", empty) *
          " "

    # ----- remaining time estimation
    elapsed = time() - pt.start_time
    avg_time_iter = elapsed / progress_iter
    avg_time_tol = elapsed / progress_tol

    eta_iter = (1 - progress_iter) * avg_time_iter
    eta_tol = (1 - progress_tol) * avg_time_tol

    eta = min(eta_iter, max(eta_tol, 0.0))

    # ----- color selection
    color = pt.color

    percent = round(progress_tol * 100, digits=1)

    diff_str = @sprintf("%.3e", diff)
    eta_str = format_eta(eta)
    percent_str = @sprintf("%5.1f%%", percent)
    iter_str = @sprintf("%3d", iter)

    eta_color = if eta_tol < eta_iter
        pt.color
    else
        CYAN
    end
    print("\r\033[K",
          CYAN, "Iter $iter_str ",
          color, bar, " ",
          "$percent_str ",
          "(diff=$diff_str)",
          eta_color,"    ETA=$eta_str")

    flush(stdout)

    pt.prev_log_diff = log_diff
    pt.prev_iter = iter
end

# ----------------------------------------------------
# Finish
# ----------------------------------------------------

"""
    finish!(pt, iter, diff)

Call once convergence is reached.

Prints final summary line and resets color.
"""
function finish!(pt::ProgressBar, iter::Int, diff::Float64)
    diff_str = @sprintf("%.3e", diff)
    eta_str = format_eta(time() - pt.start_time)
    println("\r\033[K")
    println(GREEN,
            "✓ Converged in ",
            YELLOW, "$iter",
            GREEN, " iterations after ",
            YELLOW, "$eta_str ",
            GREEN,
            "with diff=$diff_str",
            RESET)
end


"""
    format_bar(pt, diff, iter)

Returns the formatted progress bar string without printing it.
"""
function format_bar(pt::ProgressBar, diff::Float64, iter::Int)
    if !pt.initiated
        pt.initiated = true
        pt.log_diff0 = log(diff)
    end

    log_diff = log(diff)
    pt.log_diff0 = max(pt.log_diff0, log_diff)

    progress_tol = clamp(1 - (log_diff - pt.log_tol) / (pt.log_diff0 - pt.log_tol), 0.0, 1.0)
    progress_iter = clamp(iter / pt.max_iter, 0.0, 1.0)

    complete_progress = min(progress_tol, progress_iter)
    max_progress = max(progress_tol, progress_iter)

    filled = round(Int, complete_progress * pt.bar_length)
    half_filled = round(Int, (max_progress - complete_progress) * pt.bar_length)
    empty  = pt.bar_length - filled - half_filled

    halph_bar = progress_iter < progress_tol ? "▀" : "▄"
    bar = " " * repeat("█", filled) * repeat(halph_bar, half_filled) * repeat(" ", empty) * " "

    elapsed = time() - pt.start_time
    eta_iter = (1 - progress_iter) * (elapsed / progress_iter)
    eta_tol = (1 - progress_tol) * (elapsed / progress_tol)
    eta = min(eta_iter, max(eta_tol, 0.0))

    percent = round(progress_tol * 100, digits=1)
    diff_str = @sprintf("%.3e", diff)
    eta_str = format_eta(eta)
    percent_str = @sprintf("%5.1f%%", percent)
    iter_str = @sprintf("%3d", iter)
    
    eta_color = eta_tol < eta_iter ? pt.color : CYAN

    # Return the assembled string. Notice RESET is appended to prevent bleeding.
    return string(CYAN, "Iter $iter_str ", pt.color, bar, " ", 
                  "$percent_str (diff=$diff_str) ", 
                  eta_color, "ETA=$eta_str", RESET)
end


mutable struct MultiTracker
    drawn::Bool
    outer::String
    sym::String
    vfi::String
end

MultiTracker() = MultiTracker(false, "", "", "")

function render!(mt::MultiTracker)
    if mt.drawn
        # Move to start of line (\r), then up 2 lines (\033[2A) to reach the top of the 3-line block
        print("\r\033[2A")
    end
    
    # Print each line. \033[K ensures any leftover characters from previous longer strings are erased.
    print("\033[K", mt.outer, "\n")
    print("\033[K", mt.sym, "\n")
    print("\033[K", mt.vfi) # No newline on the last item to anchor the cursor
    
    mt.drawn = true
    flush(stdout)
end

end # module
