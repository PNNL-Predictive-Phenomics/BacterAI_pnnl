# analysis of conditions tested including overall "depth"
# these data were direction = 0 which is the top-down configuration
# i.e., we expect more challenging conditions as the simulations progress

library(data.table)
library(ggplot2)

source('calc_depth_function.R')

# load data
ra <- rbind(fread('../expt_rounds/Round1/results_all.csv'),
            fread('../expt_rounds/Round2/results_all.csv'),
            fread('../expt_rounds/Round3/results_all.csv'))

r4 <- fread('../expt_rounds/Round4/batch_meta_2026-01-09T14.55.04.356154.csv')

ingred <- readxl::read_excel('../expt_rounds/ingredients.xlsx', sheet = 1) |> setDT()
ingred <- ingred[INGREDIENT %in% names(ra)]


# preliminary test: Does O2 produce different fitness values?-------------------

# significant difference in y-value from O2?
ra |> 
    subset(bad == 0) |> 
    subset(fitness > 0) |> 
    lm(fitness ~ O2, data = _) |> 
    anova()
# F = 0.11, P = 0.73

ra |> 
    subset(bad == 0) |> 
    subset(fitness > 0) |> 
    transform(O2_tf = fifelse(O2 == 1, 'O2 normal', 'O2 limited')) |> 
    ggplot(aes(O2_tf, fitness)) +
    geom_boxplot(aes(group = O2_tf), outliers = F) +
    geom_jitter(width = 0.15) +
    facet_wrap(~ round) +
    labs(title = 'O2 limitation effect on growth') +
    ylab(expression(Fitness ~ (norm. ~ Delta*OD[600]))) +
    theme(axis.title.x = element_blank())

# I don't think O2 limitation is successfully working to change fitness -- assume all values are 1


# explore experimental data-----------------------------------------------------

# add round 4 conditions to see that BacterAI continues to explore new depth values
r4[, experiment_number := 1:.N]
r4_cols <- intersect(names(ra), names (r4))

# get requested conditions (ignoring O2-limitation data)
rc <- rbind(ra, r4[, ..r4_cols], fill = T) |> 
    subset(select = c(O2:ammonium_chloride, round, experiment_number, frontier_type, depth)) |> 
    melt(id.vars = c('round', 'experiment_number', 'frontier_type', 'depth'), variable.name = 'condition') |> 
    merge(ingred[, .(INGREDIENT, MIN_VALUE, MAX_VALUE)], all.x = T, by.x = 'condition', by.y = 'INGREDIENT') |> 
    transform(norm_val = (MAX_VALUE - value) / (MAX_VALUE - MIN_VALUE))

# calculate depth
rcdepth <- rc[, .(norm_depth = calc_depth(condition, value)), 
              by = .(round, experiment_number, frontier_type, depth)]

# depth vs depth
ggplot(rcdepth, aes(depth, norm_depth)) +
    geom_point(shape = 1)

# show how depth changes over rounds
ggplot(rcdepth, aes(round, norm_depth)) +
    geom_boxplot(aes(group = round), outliers = F) +
    geom_jitter(aes(shape = frontier_type, color = frontier_type), width = 0.15) +
    ylab('Norm. depth') +
    ylim(0, 10.5) +
    labs(title = 'Experiment depth over around',
         subtitle = 'Depth = distance from ideal conditions') +
    scale_color_manual(values = c('blue', 'black')) + 
    scale_shape_manual(values = c(2, 19))
# NOTE: I don't think the frontier / not-frontier distinction can be easily shown here for normalized depth


# show how conditions were explored
rc |> 
    transform(round = factor(round)) |> 
    ggplot(aes(condition, norm_val)) +
    geom_jitter(aes(shape = round, color = round), width = 0.075) +
    scale_color_manual(values = c('gray50', 'blue', 'red', 'lightgreen')) + 
    scale_shape_manual(values = c(3, 1, 1, 1)) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))

# each condition across the rounds
rc |> 
    transform(round = factor(round)) |> 
    ggplot(aes(round, value)) +
    geom_violin(aes(fill = round)) +
    # geom_jitter(aes(shape = round, color = round), width = 0.075) +
    scale_color_manual(values = c('gray50', 'blue', 'red', 'lightgreen')) + 
    scale_fill_manual(values = c('gray50', 'blue', 'red', 'lightgreen')) + 
    scale_shape_manual(values = c(3, 1, 1, 1)) +
    facet_wrap(~ condition, ncol = 3, scales = 'free_y')


# show just round 1
rc |> 
    subset(round == 1) |> 
    subset(!condition %in% c('O2', 'pH', 'mme_trace_minerals')) |> 
    transform(condition = factor(condition, 
                                 levels = c('d_glucose', 'sodium_citrate', 'sodium_octanoate', 'sodium_acetate', 'sodium_benzoate', 
                                            'urea', 'ammonium_chloride',
                                            'sodium_chloride', 'potassium_chloride', 'mme_trace_minerals'),
                                 labels = c('D-glucose', 'Citrate (-Na)', 'Octanoate (-Na)', 'Acetate (-Na)', 'Benzoate (-Na)',
                                            'Urea', 'Ammonium (-Cl)', 'NaCl', 'KCl', 'Trace Minerals (Wolfe)'))) |> 
    ggplot(aes(condition, norm_val)) +
    geom_jitter(shape = 1, size = 1, color = 'gray25', width = 0.175) +
    ylab('Norm. val') +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          axis.title.x = element_blank())


# show some single experiments from rounds 2 and 3
rc |> 
    subset((experiment_number == 48 & round == 3) | (experiment_number == 58 & round == 2) | experiment_number == 42 & round == 2) |> 
    transform(experiment_number = factor(experiment_number, levels = c(58, 48, 42), labels = c('first', 'second', 'last'))) |> 
    subset(!condition %in% c('O2', 'pH', 'mme_trace_minerals')) |> 
    (\(.) .[, .(val_level = seq(0, 1, length.out = 11), norm_val = norm_val), by = .(condition, experiment_number)
            ][, at_level := norm_val >= val_level])() |> 
    transform(condition = factor(condition, 
                                 levels = c('d_glucose', 'sodium_citrate', 'sodium_octanoate', 'sodium_acetate', 'sodium_benzoate', 
                                            'urea', 'ammonium_chloride',
                                            'sodium_chloride', 'potassium_chloride', 'mme_trace_minerals'),
                                 labels = c('D-glucose', 'Citrate (-Na)', 'Octanoate (-Na)', 'Acetate (-Na)', 'Benzoate (-Na)',
                                            'Urea', 'Ammonium (-Cl)', 'NaCl', 'KCl', 'Trace Minerals (Wolfe)'))) |> 
    ggplot(aes(condition, val_level, fill = at_level)) +
    geom_point(shape = 21, size = 1.75) +
    ylab('Norm. val') +
    facet_grid(. ~ experiment_number) +
    scale_fill_manual(values = c('white', 'black')) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          axis.title.x = element_blank(),
          strip.text = element_blank(),
          legend.position = 'none')


# run stats -- not sure aobut independence or homogeneity of variance -- use bootstrap
iters <- 999L
null_t <- numeric(iters)

for(i in 1:iters) {
    rcdepth[, null_round := sample(round, length(round), replace = T)]
    null_t[i] <- rcdepth |> 
        subset(null_round > 1) |> 
        lm(norm_depth ~ null_round, data = _) |> 
        summary() |> 
        (\(.) ifelse(.$coefficients[2, 'Estimate'] >= 0, 
                     .$coefficients[2, 'Pr(>|t|)'],
                     0))()
}

# number of times null exceeds observed?
rcdepth |> 
    subset(round > 1) |> 
    lm(norm_depth ~ round, data = _) |> 
    summary() |> 
    (\(.) sum(.$coefficients[2, 't value'] < null_t) / iters)()
# none


rcdepth |> 
    subset(round > 1) |> 
    lm(norm_depth ~ round, data = _) |> 
    anova()

