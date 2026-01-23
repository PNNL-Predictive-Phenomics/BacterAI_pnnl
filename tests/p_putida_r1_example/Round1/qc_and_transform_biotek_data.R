# analysis and transformation of raw Biotek plate reader data into long-form CSVs
# I want to look at evaporation and plate-edge effects
# but mainly I need to match the format of data output from the Tecan system

library(data.table)
library(ggplot2)

setwd('~/Documents/projects/ppi/tecan_data/P_putida_growth_boundaries/Round1/')

source('~/Documents/projects/gg-theme-bram.R')
theme_set(theme_bram)

(wp96 <- outer(LETTERS[1:8], sprintf('%01i', 1:12), paste0))
matrix(1:96, nrow = 8, ncol = 12, byrow = F, dimnames = list(LETTERS[1:8], 1:12))

max_days <- 3

expt_request <- fread('batch_meta_2025-12-09T10.23.08.422757.csv') |> 
    transform(solution_id = paste0('expt', 1:length(pH), '_pH::', pH))

p2f <- list.files('experiment_request/plate_maps', full.names = T, recursive = T) |> 
    grepv('plate_to_file_id.csv', x = _) |> 
    fread()

map <- list.files('experiment_request/plate_maps', full.names = T, recursive = T) |> 
    grepv('/map.csv', x = _) |> 
    fread()

# reads in biotek data from multiple wavelengths and converts to long-form DT
read_biotek <- function(filename, nm = 600, long = T) {
    
    if(length(nm) != 1L) stop('Must supply a single wavelength number to search for')
    
    # reads in first column and scans for entries that can be forced to integers -- these are the wavelengths
    fr <- readxl::read_excel(filename, sheet = 1, skip = 0, range = readxl::cell_cols('A'), col_names = 'V1') |> 
        unlist() |> 
        as.integer() |> 
        suppressWarnings()
    
    # use integer matches to create row groups
    fr_grp <- fr |> 
        is.na() |> 
        xor(T, y = _) |> 
        cumsum()
    
    # match a single row group to the specified wavelength
    target_grp <- fr_grp[which(fr == nm)]
    if(length(target_grp) == 0) stop('No wavelength matches found in data at ', nm, ' nm')
    target_range <- which(fr_grp == target_grp)
    
    target_start <- min(target_range + 3L)
    target_stop <- max(target_range - 2L)
    range <- paste0('B', target_start, ':', 'CU', target_stop)
    
    # get range of data for target wavelength
    df <- readxl::read_excel(filename, sheet = 1, range = range, col_types = 'numeric') |> 
        suppressWarnings() |> 
        setDT() |> 
        (\(.) .[Time > 0])()
    
    if(long) {
        c2k <- setdiff(names(df), grepv(nm, names(df)))
        dfl <- melt(df[, ..c2k], id.vars = 'Time', variable.name = 'Well')
        dfl[, wavelength := nm]
        return(dfl[])
    } else {
        return(df)
    }
}


# plate 1 - pH 5----------------------------------------------------------------
# e30d6074

plate1 <- file.path('experiment_request', 'data', 'PPI_JE3959_baseline_260102_112441.xlsx')

datp1 <- rbind(read_biotek(plate1, 600)[Time <= max_days],
               read_biotek(plate1, 480)[Time <= max_days])


# plate 2 - pH 7----------------------------------------------------------------
# efc53db4

plate2 <- file.path('experiment_request', 'data', 'PPI_JE3959_baseline_260102.xlsx')

datp2 <- rbind(read_biotek(plate2, 600)[Time <= max_days],
               read_biotek(plate2, 480)[Time <= max_days])


# plate 4 - pH 9----------------------------------------------------------------
# bddde5fd

plate4 <- file.path('experiment_request', 'data', 'PPI_JE3959_baseline_260102_plate4.xlsx')

datp4 <- rbind(read_biotek(plate4, 600)[Time <= max_days],
               read_biotek(plate4, 480)[Time <= max_days])


# plate 3 - pH 7----------------------------------------------------------------
# bddde5fd

plate3 <- file.path('experiment_request', 'data', 'PPI_JE3959_Baseline_plate3.xlsx')

datp3 <- rbind(read_biotek(plate3, 600)[Time <= max_days],
               read_biotek(plate3, 480)[Time <= max_days])


# Estimate evaporation on each well of each plate-------------------------------

# NOTE: I did the well codes to well numbers incorrectly and so I can't use the well codes in the map file
p1_evap <- map[parent_plate == p2f$parent_plate[1] & control_type == 'evap_control']$parent_well_index
p2_evap <- map[parent_plate == p2f$parent_plate[2] & control_type == 'evap_control']$parent_well_index
p3_evap <- map[parent_plate == p2f$parent_plate[3] & control_type == 'evap_control']$parent_well_index
p4_evap <- map[parent_plate == p2f$parent_plate[4] & control_type == 'evap_control']$parent_well_index

# (translate to well codes until the plate map creation scheme is fixed)
p1_evap <- wp96[p1_evap]
p2_evap <- wp96[p2_evap]
p3_evap <- wp96[p3_evap]
p4_evap <- wp96[p4_evap]

evap_dat <- rbind(datp1[wavelength == 480 & Well %in% p1_evap][, `:=` (pH = 5, plate = 1)],
                  datp2[wavelength == 480 & Well %in% p2_evap][, `:=` (pH = 7, plate = 2)],
                  datp3[wavelength == 480 & Well %in% p3_evap][, `:=` (pH = 7, plate = 3)],
                  datp4[wavelength == 480 & Well %in% p4_evap][, `:=` (pH = 9, plate = 4)])

evap_dat[, pH_well := paste0('pH: ', pH,  ', ', Well)
         ][, plate_pH := paste0('plate: ', plate, ', pH: ', pH)]

ggplot(evap_dat, aes(Time, value, color = Well)) +
    geom_line(aes(group = pH_well)) + 
    ylab(expression(Abs[480])) +
    ylim(0, 1) +
    xlab('Days') +
    facet_wrap(~ plate_pH, ncol = 2)

# Evaporation should be able to be calculated as the increase in 480 absorbance
# wherein a doubling of the signal equals a halving of the volume
evap_per_plate <- evap_dat[order(Time, plate, pH, Well)
                           ][, .(frac_vol = value[1] / value[.N]), by = .(plate, pH, Well)
                             ][, .(frac_vol = mean(frac_vol)), by = .(plate, pH)]

# final volumes and evaporation varied quite a bit:
# plate 1: final volumes = 65%
# plate 2: final volumes = 88%
# plate 3: final volumes = 90%

# how does this affect final OD values?


# Plot growth curves------------------------------------------------------------

# NOTE: I did the well codes to well numbers incorrectly and so I can't use the well codes in the map file
map[, parent_well_corrected := wp96[parent_well_index]]

od_dat <- rbind(datp1[wavelength == 600][, plate := p2f$parent_plate[1]],
                datp2[wavelength == 600][, plate := p2f$parent_plate[2]],
                datp3[wavelength == 600][, plate := p2f$parent_plate[3]],
                datp4[wavelength == 600][, plate := p2f$parent_plate[4]])

od_dat <- merge(od_dat, 
                map[, .(parent_plate, parent_well_corrected, solution_id, control_type, environment)],
                by.x = c('plate', 'Well'),
                by.y = c('parent_plate', 'parent_well_corrected'),
                all.x = T)

od_dat <- od_dat[!is.na(solution_id)]

od_dat[, plate_num := .GRP, by = plate
       ][, plate_num := paste0('plate ', plate_num)]

ggplot(od_dat, 
       aes(Time, value, color = control_type)) +
    geom_line(aes(group = Well), alpha = 0.75) + 
    scale_color_manual(values = c('black', 'orange', 'red', 'darkgreen', 'blue')) +
    ylab(expression(OD[600])) +
    xlab('Days') +
    facet_wrap(~ plate_num + environment, ncol = 2)

# O2 vs. no-O2
od_dat |> 
    merge(expt_request[, .(solution_id, o2)], by = 'solution_id', all.x = T) |> 
    transform(o2 = fifelse(is.na(o2) & control_type != 'o2_control', 2, o2)) |> 
    transform(o2 = factor(o2, levels = 0:2, labels = c('O2-limited', 'O2-normal', 'ctrls (O2-normal)'))) |> 
    subset(!is.na(o2)) |> 
    ggplot(aes(Time, value, color = factor(o2))) +
    geom_line(aes(group = Well), alpha = 0.75) + 
    scale_color_manual(values = c('black', 'blue', 'darkblue')) +
    ylab(expression(OD[600])) +
    xlab('Days') +
    facet_wrap(~ plate_num + environment, ncol = 2)


# total C
od_dat |> 
    merge(expt_request[, .(solution_id, urea, ammonium_chloride,
                           d_glucose, sodium_citrate, sodium_octanoate, 
                           sodium_acetate, sodium_benzoate, d_xylose,
                           l_arabinose)], 
          by = 'solution_id', 
          all.x = T) |> 
    transform(total_c = (d_glucose * 6 + sodium_citrate * 6 + sodium_octanoate * 8 + 
                             sodium_acetate * 2 + sodium_benzoate * 7 + d_xylose * 5 + 
                             l_arabinose * 5 + urea * 1)) |> 
    ggplot(aes(Time, value, color = total_c)) +
    geom_line(aes(group = Well), alpha = 0.75) + 
    scale_color_viridis_c(end = 0.85) +
    ylab(expression(OD[600])) +
    xlab('Days') +
    facet_wrap(~ plate_num + environment, ncol = 2)

# total N
od_dat |> 
    merge(expt_request[, .(solution_id, urea, ammonium_chloride)], 
          by = 'solution_id', 
          all.x = T) |> 
    transform(total_c = (urea * 2 + ammonium_chloride * 1)) |> 
    ggplot(aes(Time, value, color = total_c)) +
    geom_line(aes(group = Well), alpha = 0.75) + 
    scale_color_viridis_c(end = 0.85) +
    ylab(expression(OD[600])) +
    xlab('Days') +
    facet_wrap(~ plate_num + environment, ncol = 2)

# CONCLUSIONS:
#   - IMPROVE POSITIVE CONTROLS. I will need to adjust the positive controls to have a lot more C and N, maybe 30 mM C, 10 mM N
#   - REVISIT O2-LIMITATION. I'm not sure that the O2 controls are doing what I want -- we see consistently faster growth in O2-negative wells
#       My guess is that the substrates (either formate, glycoerphosphoric acid, or the enzyme itself) are food for P putida
#       May need to save these for separate anaerobic environment experiments
#       Another issue is pH-dependence of the enzyme -- it may not work as well at pH=5 or pH=9
#   - EXCLUDE P-COUMARATE. P-coumarate is not possible to pipette
#   - FOR PLATEPLAN: MIX UP ALL PH CONDITIONS INTO ALL PLATES





# output to .asc----------------------------------------------------------------
# I am re-creating the typical plain-text ASCII output from my read protocol

# note that file names include shorter date/time codes than the end of the .asc files
# also note that the date/time on the file name indicates the START of the read,
# while the number at the bottom will show the END of the read

# Well positions  abs600nm  abs480nm
# A1              0.0892    0.0924
# ...             ...       ...
# H12             0.0875    0.0905
# Date of measurement: 2026-01-06/Time of measurement: 14:14:21

# plate 1
#   start: 2026-01-02, 12:24:49 PM

# make sure that 600 and 480 nm reads are on the same file,
# this matches Tecan output

make_asc <- F

if(make_asc) {
    datp1 <- datp1[order(Time)
    ][, read_num := 0
    ][wavelength == 600, read_num := .GRP, by = Time
    ][wavelength == 480, read_num := .GRP, by = Time]
    
    pn = p2f$file_id[1]
    start_time <- strptime('2026-01-02 12:24:49', format = '%Y-%m-%d %H:%M:%S')
    for(rn in unique(datp1$read_num)) {
        dwt <- datp1[read_num == rn]
        t <- max(dwt$Time)
        dwt <- dcast(dwt, Well ~ wavelength, value.var = 'value')
        dwt <- dwt[, .(Well, `600`, `480`)]
        setnames(dwt, new = c('Well positions', 'abs600nm', 'abs480nm'))
        
        # format the date and time values
        seconds_since_start <- t * 24 * 60 * 60
        read_time <- start_time + seconds_since_start
        the_date <- format(read_time, '%y%m%d')
        the_date2 <- format(read_time, '%Y-%m-%d')
        the_time <- format(read_time, '%H%M%S')
        the_time2 <- format(read_time, '%H:%M:%S')
        
        # write out
        outfile <- paste(pn, the_date, the_time, sep = '_') |> 
            (\(.) paste0(., '.asc'))()
        fwrite(dwt, file = file.path('experiment_request', 'data', outfile), sep = '\t')
        write(paste0('Date of measurement: ', the_date2, '/Time of measurement: ', the_time2), 
              file = file.path('experiment_request', 'data', outfile),
              append = T)
    }
    
    
    # plate 2
    #   start : 2026-01-02, 12:09:59 PM
    datp2 <- datp2[order(Time)
    ][, read_num := 0
    ][wavelength == 600, read_num := .GRP, by = Time
    ][wavelength == 480, read_num := .GRP, by = Time]
    
    pn = p2f$file_id[2]
    start_time <- strptime('2026-01-02 12:09:59', format = '%Y-%m-%d %H:%M:%S')
    for(rn in unique(datp2$read_num)) {
        dwt <- datp2[read_num == rn]
        t <- max(dwt$Time)
        dwt <- dcast(dwt, Well ~ wavelength, value.var = 'value')
        dwt <- dwt[, .(Well, `600`, `480`)]
        setnames(dwt, new = c('Well positions', 'abs600nm', 'abs480nm'))
        
        # format the date and time values
        seconds_since_start <- t * 24 * 60 * 60
        read_time <- start_time + seconds_since_start
        the_date <- format(read_time, '%y%m%d')
        the_date2 <- format(read_time, '%Y-%m-%d')
        the_time <- format(read_time, '%H%M%S')
        the_time2 <- format(read_time, '%H:%M:%S')
        
        # write out
        outfile <- paste(pn, the_date, the_time, sep = '_') |> 
            (\(.) paste0(., '.asc'))()
        fwrite(dwt, file = file.path('experiment_request', 'data', outfile), sep = '\t')
        write(paste0('Date of measurement: ', the_date2, '/Time of measurement: ', the_time2), 
              file = file.path('experiment_request', 'data', outfile),
              append = T)
    }
    
    # plate 4
    #   start: 2026-01-02, 12:23:30 PM
    datp4 <- datp4[order(Time)
    ][, read_num := 0
    ][wavelength == 600, read_num := .GRP, by = Time
    ][wavelength == 480, read_num := .GRP, by = Time]
    
    pn = p2f$file_id[4]
    start_time <- strptime('2026-01-02 12:23:30', format = '%Y-%m-%d %H:%M:%S')
    for(rn in unique(datp4$read_num)) {
        dwt <- datp4[read_num == rn]
        t <- max(dwt$Time)
        dwt <- dcast(dwt, Well ~ wavelength, value.var = 'value')
        dwt <- dwt[, .(Well, `600`, `480`)]
        setnames(dwt, new = c('Well positions', 'abs600nm', 'abs480nm'))
        
        # format the date and time values
        seconds_since_start <- t * 24 * 60 * 60
        read_time <- start_time + seconds_since_start
        the_date <- format(read_time, '%y%m%d')
        the_date2 <- format(read_time, '%Y-%m-%d')
        the_time <- format(read_time, '%H%M%S')
        the_time2 <- format(read_time, '%H:%M:%S')
        
        # write out
        outfile <- paste(pn, the_date, the_time, sep = '_') |> 
            (\(.) paste0(., '.asc'))()
        fwrite(dwt, file = file.path('experiment_request', 'data', outfile), sep = '\t')
        write(paste0('Date of measurement: ', the_date2, '/Time of measurement: ', the_time2), 
              file = file.path('experiment_request', 'data', outfile),
              append = T)
    }
    
    
    # plate 3
    #   start: 2026-01-14, 11:17:13 AM
    datp3 <- datp3[order(Time)
    ][, read_num := 0
    ][wavelength == 600, read_num := .GRP, by = Time
    ][wavelength == 480, read_num := .GRP, by = Time]
    
    pn = p2f$file_id[3]
    start_time <- strptime('2026-01-14 11:17:13', format = '%Y-%m-%d %H:%M:%S')
    for(rn in unique(datp3$read_num)) {
        dwt <- datp3[read_num == rn]
        t <- max(dwt$Time)
        dwt <- dcast(dwt, Well ~ wavelength, value.var = 'value')
        dwt <- dwt[, .(Well, `600`, `480`)]
        setnames(dwt, new = c('Well positions', 'abs600nm', 'abs480nm'))
        
        # format the date and time values
        seconds_since_start <- t * 24 * 60 * 60
        read_time <- start_time + seconds_since_start
        the_date <- format(read_time, '%y%m%d')
        the_date2 <- format(read_time, '%Y-%m-%d')
        the_time <- format(read_time, '%H%M%S')
        the_time2 <- format(read_time, '%H:%M:%S')
        
        # write out
        outfile <- paste(pn, the_date, the_time, sep = '_') |> 
            (\(.) paste0(., '.asc'))()
        fwrite(dwt, file = file.path('experiment_request', 'data', outfile), sep = '\t')
        write(paste0('Date of measurement: ', the_date2, '/Time of measurement: ', the_time2), 
              file = file.path('experiment_request', 'data', outfile),
              append = T)
    }
}






