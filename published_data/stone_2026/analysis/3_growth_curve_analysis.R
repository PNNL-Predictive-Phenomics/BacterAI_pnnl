# growth curve analysis
# 1. plotting full curves, colored by pH, color intensity by depth, faceted by round
# 2. correlation of final OD / fitness with max growth rate
# 3. observation of diauxic growth curves

library(data.table)
library(ggplot2)

source('calc_depth_function.R')

read_biotek <- function(filename, signal = '600') {
    require(readxl)
    if(is.numeric(signal)) signal <- as.character(signal)
    y <- readxl::read_excel(filename, skip = 0)
    
    wp <- y[,1, drop = T] == signal
    wp[is.na(wp)] <- F
    wp <- cumsum(wp)
    z <- split(y[wp > 0,], wp[wp > 0])
    
    z <- lapply(z, function(xx) xx[-(1:2), -1])
    z <- lapply(z, function(xx) {
        cols <- xx[1,]
        dat <- as.matrix(xx[2:nrow(xx),])
        suppressWarnings(storage.mode(dat) <- 'numeric')
        colnames(dat) <- cols
        return(as.data.table(dat)[Time != 0])
    })
    
    bycols <- Reduce(intersect, lapply(z, names))
    z <- Reduce(function(x, y) merge.data.table(x, y, by = bycols, sort = F), z)
    melt(z, id.vars = bycols, value.name = 'value', variable.name = 'well')
}

# load data
ra <- rbind(fread('../data/Round1/results_all.csv'),
            fread('../data/Round2/results_all.csv'),
            fread('../data/Round3/results_all.csv'))

ingred <- readxl::read_excel('../data/ingredients.xlsx', sheet = 1) |> setDT()
ingred <- ingred[INGREDIENT %in% names(ra)]

# load growth curve data -- the above results data helps determine which experiments are bad
gc <- rbind(read_biotek('../data/experiment_request/2025-04-15/data/c2af7184_data.xlsx')[, round := 1],
            read_biotek('../data/experiment_request/2025-09-11/data/25e5dbfa.xlsx')[, round := 2],
            read_biotek('../data/experiment_request/2025-09-22/data/aab1ab3c_data.xlsx')[, round := 3])

# load plate maps
pm <- rbind(fread('../data/experiment_request/2025-04-15/plate_maps/PpAG5577$6b89594af61d036c3b9a9bf72cdc3e522d6dbea0/map.csv')[, round := 1],
            fread('../data/experiment_request/2025-09-11/plate_maps/PpAG5577$6e84dc13d2d930cba3b2b5d36efc0767f5b7919d/map.csv')[, round := 2],
            fread('../data/experiment_request/2025-09-22/plate_maps/PpAG5577$f6b26280bf6fe89f524b965dae10494d99fadb4a/map.csv')[, round := 3],
            fill = T)

pm[, environment := toupper(environment)]

# which plates were experiments?
gc <- merge(gc, 
            pm[, .(round, parent_well, solution_id, environment)], 
            all.x = T, 
            by.x = c('round', 'well'),
            by.y = c('round', 'parent_well'))

gc <- gc[!is.na(solution_id)
         ][, experiment_number := sub('expt([0-9]+)_.*', '\\1', solution_id)
           ][, experiment_number := fifelse(grepl('plate', experiment_number), 
                                            NA, 
                                            as.integer(experiment_number))]

# generate control experiment numbers
get_well_num <- function(x) {
    wp384 <- outer(LETTERS[1:16], 1:24, paste0)
    wn <- integer(length(x))
    nn <- 1L
    for(y in x) {
        wn[nn] <- which(wp384 == y)
        nn <- nn + 1L
    }
    return(wn)
}

gc[grepl('plate', solution_id), experiment_number := (
    sprintf('999%i%i%s', round, get_well_num(well), substring(environment, 3, 3)) |> as.integer()
    )]


# 1. Plot curves----------------------------------------------------------------

# recalculate normalized depth
rcdepth <- ra |> 
    subset(select = c(O2:ammonium_chloride, round, experiment_number)) |> 
    melt(id.vars = c('round', 'experiment_number'), variable.name = 'condition') |> 
    merge(ingred[, .(INGREDIENT, MIN_VALUE, MAX_VALUE)], all.x = T, by.x = 'condition', by.y = 'INGREDIENT') |> 
    transform(norm_val = (MAX_VALUE - value) / (MAX_VALUE - MIN_VALUE)) |> 
    (\(x) x[, .(norm_depth = calc_depth(condition, value)), by = .(round, experiment_number)])()

# add depth value to each growth curve
gc <- merge(gc, rcdepth, all.x = T, by = c('round', 'experiment_number'))

# note that there are some redos from round 2 in the assays run in round 3 -- there are 7 of them
gc |> 
    subset(!is.na(experiment_number)) |> 
    (\(.) .[order(norm_depth)])() |> 
    transform(round = sprintf('Round %i', round)) |> 
    ggplot(aes(Time, value, text = paste(round, solution_id))) +
    ylab(expression(OD[600])) +
    xlab('Days') +
    geom_line(aes(group = experiment_number, color = norm_depth)) +
    scale_color_viridis_c(end = 0.95) +
    facet_grid(round ~ environment) +
    theme(legend.position = 'bottom')

# plotly::ggplotly()

gc |> 
    transform(sam_type = fifelse(solution_id == 'plate_control', 'control', 'sample')) |> 
    transform(round = sprintf('Round %i', round)) |> 
    ggplot(aes(Time, value, text = paste(round, solution_id))) +
    ylab(expression(OD[600])) +
    xlab('Days') +
    geom_line(aes(group = experiment_number, color = sam_type)) +
    scale_color_manual(values = c('red', 'black')) +
    facet_grid(round ~ environment) +
    theme(legend.position = 'bottom')


# 2. Max growth rates-----------------------------------------------------------

# rolling-median-then-mean smoother
rmms <- function(x, ...) {
    frollmedian(x, fill = NA, algo = 'exact', align = 'center', ...) |> frollmean(x = _, ...)
}

# How much to smooth growth curves? Compare smoothness
gc[round == 2 & solution_id == 'expt36_pH::7'
   ][, `:=` (value_smooth5 = rmms(value, 5),
             value_smooth7 = rmms(value, 7),
             value_smooth11 = rmms(value, 11))] |> 
    melt(id.vars = 'Time', measure.vars = c('value', 'value_smooth5', 'value_smooth7', 'value_smooth11')) |> 
    ggplot(aes(Time, value)) +
    geom_line() +
    scale_y_log10() +
    facet_wrap(~ variable, ncol = 2)

gc[round == 3 & solution_id == 'expt41_pH::7'
   ][, `:=` (value_smooth5 = rmms(value, 5),
             value_smooth7 = rmms(value, 7),
             value_smooth11 = rmms(value, 11))] |> 
    melt(id.vars = 'Time', measure.vars = c('value', 'value_smooth5', 'value_smooth7', 'value_smooth11')) |> 
    ggplot(aes(Time, value)) +
    geom_line() +
    scale_y_log10() +
    facet_wrap(~ variable, ncol = 2)
# Rule is to use as narrow a window as necessary

# smooth the growth curves with smoothing window
ww <- 11
gc[, `:=` (value_smooth = rmms(value, ww), time_hr = Time * 24), by = .(round, experiment_number)]

# log-transform smoothed values to have relative growth rates
gc[, value_smooth := log(value_smooth)]

# CALCULATE MAX RELATIVE GORWTH RATE
# calculate maximum relative growth rate on smoothed lines
mgr <- gc[, `:=` (vs_p1 = shift(value_smooth, 1, type = 'lag'),
                  time_p1 = shift(time_hr, 1, type = 'lag')), 
          by = .(round, experiment_number)
          ][, gpr := (value_smooth - vs_p1) / (time_hr - time_p1), 
            by = .(round, experiment_number)
            ][, gpr := rmms(gpr, 3),
              by = .(round, experiment_number)
              ][, .(mgr = max(gpr, na.rm = T),
                    mgr_time = time_hr[which.max(gpr)]),
                by = .(round, experiment_number, solution_id)]


# example 20 plots validating that we found max growth rate
gc |> 
    subset(round == 1) |> 
    subset(solution_id %in% mgr$solution_id[1:20]) |> 
ggplot(aes(time_hr, value_smooth)) +
    geom_line() +
    ylab('log OD 600') +
    geom_vline(aes(xintercept = mgr_time), color = 'red', linetype = 2,
               data = mgr[1:20]) +
    facet_wrap(~ solution_id, ncol = 7)


# average max growth rate for positive controls
pc <- gc[solution_id == 'plate_control'
         ][, environment := toupper(environment)][!(round == 2 & environment == 'PH7')
          ][!(round == 2 & well == 'E14')]

apc <- pc[, `:=` (vs_p1 = shift(value_smooth, 1, type = 'lag'),
                  time_p1 = shift(time_hr, 1, type = 'lag')), 
          by = .(round, experiment_number)
          ][, gpr := (value_smooth - vs_p1) / (time_hr - time_p1), 
            by = .(round, experiment_number)
            ][, gpr := rmms(gpr, 3),
              by = .(round, experiment_number)
              ][, .(mgr = max(gpr, na.rm = T),
                    mgr_time = time_hr[which.max(gpr)]),
                by = .(round, experiment_number, solution_id)]


# write out -- NOTE AGAIN: OD value were log-transformed so these are relative growth rates!
fwrite(mgr, 'max_growth_rates.csv')

apc[, .(avg_mgr = mean(mgr), sd_mgr = sd(mgr)), by = round] |> 
    fwrite('positive_ctrls_mgr.csv')


# 3. Diauxie shifts-------------------------------------------------------------

# example curves

#   example of single curve = round 2 expt36_pH::7
ex1 <- gc[round == 2 & solution_id == 'expt36_pH::7'
          ][, value_smooth := rmms(value, ww)
            ][, od_deriv := calc_deriv(Time, value_smooth, window_width_n = ww + 10)
              ][Time <= 1]

gridExtra::grid.arrange(
    (
        ggplot(ex1, aes(Time, value_smooth)) +
            geom_point(shape = 1) +
            ylab('OD-600')
    ),
    (
        ggplot(ex1, aes(Time, od_deriv)) +
            geom_point(shape = 1) +
            ylab('Derivative')
    ),
    heights = c(1, 1)
)


gc[round == 2 & solution_id == 'expt36_pH::7'
   ][, value_smooth := rmms(value, ww)
     ][, od_deriv := calc_deriv(Time, value_smooth, window_width_n = ww)
       ][Time <= 1
         ][, .(diauxie_time = find_local_extrema(Time, od_deriv, return = 'index', window_width_n = ww))]


#   example of double curve = round 3 expt54_pH::9
ex2 <- gc[round == 3 & solution_id == 'expt54_pH::9'
          ][, value_smooth := rmms(value, ww)
            ][, od_deriv := calc_deriv(Time, value_smooth, window_width_n = ww + 10)
              ][value_smooth > 0.1]

gridExtra::grid.arrange(
    (
        ggplot(ex2, aes(Time, value_smooth)) +
            geom_point(shape = 1) +
            ylab('OD-600')
    ),
    (
        ggplot(ex2, aes(Time, od_deriv)) +
            geom_point(shape = 1) +
            ylab('Derivative')
    ),
    heights = c(1, 1)
)


#   another = round 3 expt41_pH::7
ex3 <- gc[round == 3 & solution_id == 'expt41_pH::7'
          ][, value_smooth := rmms(value, ww)
            ][, od_deriv := calc_deriv(Time, value_smooth, window_width_n = ww + 10)
              ][value_smooth > 0.1]

gridExtra::grid.arrange(
    (
        ggplot(ex3, aes(Time, value_smooth)) +
            geom_point(shape = 1) +
            ylab('OD-600')
    ),
    (
        ggplot(ex3, aes(Time, od_deriv)) +
            geom_point(shape = 1) +
            ylab('Derivative')
    ),
    heights = c(1, 1)
)

#   example of curve with decrease of OD = round 2 expt24_pH::5
ex4 <- gc[round == 2 & solution_id == 'expt24_pH::5'
          ][, value_smooth := rmms(value, ww)
            ][, od_deriv := calc_deriv(Time, value_smooth, window_width_n = 3)
              ][value_smooth > 0.1]

gridExtra::grid.arrange(
    (
        ggplot(ex4, aes(Time, value_smooth)) +
            geom_point(shape = 1) +
            ylab('OD-600')
    ),
    (
        ggplot(ex4, aes(Time, od_deriv)) +
            geom_point(shape = 1) +
            ylab('Derivative')
    ),
    heights = c(1, 1)
)

# CONCLUSION:
#   


