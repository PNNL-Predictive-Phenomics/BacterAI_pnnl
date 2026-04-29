# look at positive controls from each plate and each pH level -- understanding 

library(data.table)
library(ggplot2)

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

# load growth curve data -- the above results data helps determine which experiments are bad
gc <- rbind(read_biotek('../expt_rounds/experiment_request/2025-04-15/data/c2af7184_data.xlsx')[, round := 1],
            read_biotek('../expt_rounds/experiment_request/2025-09-11/data/25e5dbfa.xlsx')[, round := 2],
            read_biotek('../expt_rounds/experiment_request/2025-09-22/data/aab1ab3c_data.xlsx')[, round := 3])

# load plate maps
pm <- rbind(fread('../expt_rounds/experiment_request/2025-04-15/plate_maps/PpAG5577$6b89594af61d036c3b9a9bf72cdc3e522d6dbea0/map.csv')[, round := 1],
            fread('../expt_rounds/experiment_request/2025-09-11/plate_maps/PpAG5577$6e84dc13d2d930cba3b2b5d36efc0767f5b7919d/map.csv')[, round := 2],
            fread('../expt_rounds/experiment_request/2025-09-22/plate_maps/PpAG5577$f6b26280bf6fe89f524b965dae10494d99fadb4a/map.csv')[, round := 3],
            fill = T)

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

pc <- gc[solution_id == 'plate_control'
         ][, environment := toupper(environment)]

# plot
ggplot(pc, aes(Time, value, text = sprintf('Well: %s', well))) +
    geom_line(aes(group = well)) +
    facet_grid(round ~ environment)

# plotly::ggplotly()
# CONCLUSION: there are some very high positive controls that should be removed
#   round 2, wells C12, D7, and E14
# in fact, I wouldn't trust ANY of the round 2, pH 7 positive controls

# FURTHER CONCLUSION:
# I should manually re-normalize fitness values for creating figures using corrected average positive control values

# average positive controls
apc <- pc[order(round, well, Time)
          ][!(round == 2 & environment == 'PH7')
            ][!(round == 2 & well == 'E14')
              ][!is.na(value)
                ][, .(delta_od = value[.N] - value[1]), by = .(round, well)]

apc[, .(avg_pc = mean(delta_od), sd_pc = sd(delta_od)), by = round] |> 
    fwrite('positive_ctrls.csv')

