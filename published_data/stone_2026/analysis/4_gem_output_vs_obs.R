# comparison of experimental data against genome-scale metabolic model output

library(data.table)
library(ggplot2)

source('calc_depth_function.R')

# load experiment data
# "feature" column = raw delta OD
# "fitness" column = delta OD normalized to average positive plate control (should be consistent across plates)
ra <- rbind(fread('../expt_rounds/Round1/results_all.csv'),
            fread('../expt_rounds/Round2/results_all.csv'),
            fread('../expt_rounds/Round3/results_all.csv'))

ingred <- readxl::read_excel('../expt_rounds/ingredients.xlsx', sheet = 1) |> setDT()
ingred <- ingred[INGREDIENT %in% names(ra)]

apc <- fread('positive_ctrls.csv')
apc_mgr <- fread('positive_ctrls_mgr.csv')
mgr <- fread('max_growth_rates.csv')

# load GEM-model OD output
get_batch <- function(data = '../expt_rounds/Batch_growth_predictions_AG5577_3_rounds.xlsx', round = 1) {
    readxl::read_excel(data, sheet = sprintf('round%i', round)) |> 
        setDT() |> 
        (\(.) if(round > 1) subset(., select = -1) else .)() |> 
        (\(.) setnames(., new = sprintf('%s_r%i', names(.), round)))()
}

gemr <- cbind(get_batch(round = 1),
              get_batch(round = 2),
              get_batch(round = 3))

setnames(gemr, old = names(gemr)[1], new = 'Time')

# re-calculate fitness values using updated average positive control per round (plate)
ra <- merge(ra, apc, all.x = T, by = 'round', sort = F)

ra[, fitness_old := fitness
   ][, fitness := feature / avg_pc]

# # output for Joonhoon to compare models with
# ra |> 
#     subset(bad == 0) |> 
#     subset(select = c(round:ammonium_chloride, experiment_number, feature)) |> 
#     transform(feature = fifelse(feature > 0, feature, 0)) |> 
#     setnames(old = 'feature', new = 'delta_od') |> 
#     fwrite('delta_od_for_gem_compare.csv')

# re-calculate max growth rate using average positive control per round
mgr <- merge(mgr, apc_mgr, all.x = T, by = 'round', sort = F)

mgr[, mgr_norm := mgr / avg_mgr]

# recalculate normalized depth
rcdepth <- ra |> 
    subset(select = c(O2:ammonium_chloride, round, experiment_number)) |> 
    melt(id.vars = c('round', 'experiment_number'), variable.name = 'condition') |> 
    merge(ingred[, .(INGREDIENT, MIN_VALUE, MAX_VALUE)], all.x = T, by.x = 'condition', by.y = 'INGREDIENT') |> 
    transform(norm_val = (MAX_VALUE - value) / (MAX_VALUE - MIN_VALUE)) |> 
    (\(x) x[, .(norm_depth = calc_depth(condition, value)), by = .(round, experiment_number)])()

ra <- merge(ra, rcdepth, all.x = T, by = c('experiment_number', 'round'))


# convert GEM output to long form
gemr <- melt(gemr, id.vars = 'Time', variable.name = 'experiment_round', value.name = 'od')

gemr[, `:=` (experiment_number = as.integer(sub('exp(\\d+)_r(\\d)', '\\1', experiment_round)),
             round = as.integer(sub('exp(\\d+)_r(\\d)', '\\2', experiment_round)))]


# total growth------------------------------------------------------------------

# calculate delta-OD
gem_feature <- gemr[order(Time)][, .(delta_od = od[Time == 24] - od[1]), by = .(experiment_number, round)]

# compare growth: estimated vs measured
cg <- merge(ra[bad == 0][, .(fitness, experiment_number, round, norm_depth)],
            gem_feature[, .(delta_od, experiment_number, round)],
            all.x = T,
            by = c('experiment_number', 'round')) |> 
    transform(fitness = fifelse(fitness < 0, 0, fitness))

cg |> 
    (\(.) cor.test(.$fitness, .$delta_od, alternative = 'greater'))()

cor_val <- cg |> 
    (\(.) cor(.$fitness, .$delta_od))() |> 
    round(2)

cg[, lm_pred := predict(lm(delta_od ~ fitness + 0), se.fit = F)]

o_vs_m <- ggplot(cg, 
       aes(fitness, delta_od, fill = norm_depth)) +
    geom_line(aes(y = lm_pred), color = 'gray50') +
    geom_point(shape = 21) +
    ylim(0, 0.2) +
    ylab(expression(Modeled ~ Delta*OD[600])) +
    xlab(expression(Fitness ~ (norm. ~ Delta*OD[600]))) +
    annotate('text', x = Inf, y = -Inf, hjust = 1.5, vjust = -1, label = sprintf('r = %s', cor_val)) +
    scale_fill_viridis_c(guide = guide_colorbar(title = 'Norm. depth')) +
    theme(legend.title = element_text(),
          legend.position = 'bottom',
          legend.margin = margin(t = -.02, unit = 'npc'))


# compare growth scatter residuals -- does model perform worse at higher depths?
# use studentized residuals which are standardized and which can be more consistent than rstandard() -- most robust diagnostic
cg$compare_resid <- lm(delta_od ~ fitness + 0, cg) |> rstudent() |> abs()

lm(compare_resid ~ norm_depth, cg) |> 
    summary()

ggplot(cg, 
       aes(norm_depth, compare_resid, fill = norm_depth)) +
    geom_point(shape = 21) +
    ylab('ASR') +
    xlab('Norm. depth') +
    labs(tag = 'C') +
    scale_fill_viridis_c(guide = guide_colorbar(title = 'Norm. depth')) +
    theme(legend.position = 'none',
          plot.margin = unit(c(5.5, 5.5, 36.5, 5.5), units = 'pt'))


# maximum growth rate-----------------------------------------------------------

# log transform growth for relative growth rate
gemr[, od := log(od)]

# get max growth rate per hour
gem_feature_mgr <- gemr[, od_p1 := shift(od, 1, type = 'lag'), by = .(round, experiment_number)
                        ][, gph := od - od_p1
                          ][, .(mgr = max(gph, na.rm = T)), by = .(round, experiment_number)]

# make sure to keep track of which experiments are "bad"
mgr <- merge(mgr, ra[, .(round, experiment_number, bad)], all.x = T, by = c('round', 'experiment_number'))

# compare against observed max growth rate
# (could also compare time of max growth)
cmgr <- merge(gem_feature_mgr,
              mgr[!(is.na(bad) | bad == 1)][, .(round, experiment_number, mgr, mgr_norm)],
              all.y = T,
              by = c('round', 'experiment_number'),
              suffixes = c('_mod', '_obs'))

# get depth
cmgr <- merge(cmgr, ra[, .(round, experiment_number, norm_depth)], all.x = T, by = c('round', 'experiment_number'))

cmgr |> 
    (\(.) cor.test(.$mgr_obs, .$mgr_mod, alternative = 'greater'))()

cor_val_mgr <- cmgr |> 
    (\(.) cor(.$mgr_obs, .$mgr_mod))() |> 
    round(2)

cmgr[, lm_pred := predict(lm(mgr_mod ~ mgr_obs + 0), se.fit = F)]

# plot observed vs modeled max growth rate
o_vs_m_mgr <- ggplot(cmgr, 
       aes(mgr_obs, mgr_mod, fill = norm_depth)) +
    # geom_line(aes(y = lm_pred), color = 'gray50') +
    geom_point(shape = 21) +
    # ylim(0, 0.04) +
    ylab('Modeled maximum RGR') +
    xlab('Observed maximum RGR') +
    annotate('text', x = Inf, y = -Inf, hjust = 1.5, vjust = -1, label = sprintf('r = %s', cor_val_mgr)) +
    scale_fill_viridis_c(guide = guide_colorbar(title = 'Norm. depth')) +
    theme(legend.position = 'none',
          plot.margin = unit(c(5.5, 5.5, 36.5, 5.5), units = 'pt'))
# CONCLUSION:
#   MODELED GROWTH RATE WAS GENERALLY AN OVERESTIMATE OF OBSERVED
#   THERE IS A CLEARLY OBSERVABLE UPPER-TRIANGLE TYPE SHAPE


combn_plot <- gridExtra::grid.arrange(o_vs_m + labs(tag = 'A'),
                                      o_vs_m_mgr + labs(tag = 'B'),
                                      widths = c(1, 1))


# residuals significantly different by depth?
cmgr$compare_resid <- lm(mgr_mod ~ mgr_obs + 0, cmgr) |> rstudent() |> abs()

lm(compare_resid ~ norm_depth, cmgr) |> 
    summary()

ggplot(cmgr, 
       aes(norm_depth, compare_resid, fill = norm_depth)) +
    geom_point(shape = 21) +
    ylab('ASR') +
    xlab('Norm. depth') +
    labs(tag = 'D') +
    scale_fill_viridis_c(guide = guide_colorbar(title = 'Norm. depth')) +
    theme(legend.position = 'none',
          plot.margin = unit(c(5.5, 5.5, 36.5, 5.5), units = 'pt'))


