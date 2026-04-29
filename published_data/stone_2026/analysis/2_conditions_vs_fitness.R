# analysis of conditions vs fitness

library(data.table)
library(ggplot2)

source('calc_depth_function.R')

# load data
ra <- rbind(fread('../data/Round1/results_all.csv'),
            fread('../data/Round2/results_all.csv'),
            fread('../data/Round3/results_all.csv'))

ingred <- readxl::read_excel('../data/ingredients.xlsx', sheet = 1) |> setDT()
ingred <- ingred[INGREDIENT %in% names(ra)]

apc <- fread('positive_ctrls.csv')

# recalculate normalized depth
rcdepth <- ra |> 
    subset(select = c(O2:ammonium_chloride, round, experiment_number)) |> 
    melt(id.vars = c('round', 'experiment_number'), variable.name = 'condition') |> 
    merge(ingred[, .(INGREDIENT, MIN_VALUE, MAX_VALUE)], all.x = T, by.x = 'condition', by.y = 'INGREDIENT') |> 
    transform(norm_val = (MAX_VALUE - value) / (MAX_VALUE - MIN_VALUE)) |> 
    (\(x) x[, .(norm_depth = calc_depth(condition, value)), by = .(round, experiment_number)])()

ra <- merge(ra, rcdepth, all.x = T, by = c('experiment_number', 'round'))

# re-calculate fitness values using updated average positive control per round (plate)
ra <- merge(ra, apc, all.x = T, by = 'round', sort = F)

ra[, fitness_old := fitness
   ][, fitness := feature / avg_pc]

# fitness values < 0 are all non-growth conditions, re-zero
ra[fitness < 0, fitness := 0]


# result of normalized depth on fitness
plot_fitness <- function(data, target = c()) {
    data |> 
        transform(round = factor(round)) |> 
        ggplot(aes(get(target), fitness, color = round)) +
        geom_hline(yintercept = 1) +
        geom_point(shape = 1) +
        xlab(target) +
        ylab(expression(Fitness ~ (norm. ~ Delta*OD[600]))) +
        scale_color_manual(values = c('gray50', 'blue', 'red'),
                           guide = guide_legend(override.aes = list(shape = 15, size = 2))) + 
        scale_shape_manual(values = c(2, 19)) +
        theme(legend.position = 'inside',
              legend.direction = 'horizontal',
              legend.position.inside = c(0.5, 1),
              legend.justification.inside = c(1, 1))
}

# fitness vs. normalized depth
depth_v_fitness <- ra |> 
    plot_fitness('norm_depth') +
    xlab('Norm. depth') +
    geom_abline(slope = -0.95063, intercept = 9.23394, color = 'gray25', linetype = 2) +
    geom_abline(slope = -0.78768, intercept = 6.32099, color = 'gray60', linetype = 1) +
    scale_x_continuous(sec.axis = dup_axis(name = '')) +
    theme(plot.margin = unit(c(-10.5, 5.5, 5.5, 5.5), units = 'pt'))
# observe a likely growth boundary towards the upper-right of the gray points
# there seems to be a certain depth under which certain growth values are no longer achievable

# add boxplots above the plot
depth_box <- rcdepth |> 
    transform(round = factor(round, levels = c(1:3))) |> 
    ggplot(aes(round, norm_depth, fill = round)) +
    geom_boxplot(aes(group = round), outlier.shape = 1, outlier.alpha = 0) +
    scale_fill_manual(values = c('gray50', 'blue', 'red')) +
    coord_flip() +
    theme_void() +
    theme(legend.position = 'none',
          plot.margin = unit(c(0, .01, 0, .13), 'npc'))

comb_plot <- gridExtra::grid.arrange(depth_box, depth_v_fitness, heights = c(1, 4))


# total salt vs fitness -- worse predictor of ftiness than total depth
ra |> 
    transform(total_salt = sodium_chloride + potassium_chloride) |> 
    plot_fitness('total_salt') +
    xlab('Total salt (mM)')

# total carbon vs fitness
ra |> 
    transform(total_carbon = (d_glucose * 6 + sodium_citrate * 6 + sodium_octanoate * 8 + sodium_acetate * 2 + sodium_benzoate * 7)) |> 
    plot_fitness('total_carbon') +
    xlab('Total carbon (mM)')
# in top-left corner this time, we have some indications of a limit on achievable growth under
# a certain concentration


# let's look at what growth and non-growth look like at above depth of 7.5
ra[norm_depth > 7.7][, c(1:15, 26, 28)][order(fitness)]

# Let's explore Nitrogen!
nsf <- ra |> 
    transform(`Growth\nabove\nthreshold` = fifelse(fitness > 0.25, T, F)) |> 
    ggplot(aes(ammonium_chloride, urea, fill = fitness, color = `Growth\nabove\nthreshold`, shape = `Growth\nabove\nthreshold`)) +
    geom_point() +
    xlab('Ammonium chloride (mM)') +
    ylab('Urea (mM)') +
    labs(tag = 'A') +
    scale_fill_viridis_c(option = 'magma', guide = guide_colorbar(title = 'Fitness')) +
    scale_shape_manual(values = c(1, 22)) +
    scale_color_manual(values = c('gray50', 'black'),
                       guide = guide_legend(override.aes = list(shape = c(1, 15)))) +
    theme(legend.position = 'bottom',
          legend.title = element_text(size = 11),
          legend.box = 'horizontal',
          legend.direction = 'horizontal')

nsd <- ra |> 
    transform(`Growth\nabove\nthreshold` = fifelse(fitness > 0.25, T, F)) |> 
    ggplot(aes(ammonium_chloride, urea, fill = norm_depth, color = `Growth\nabove\nthreshold`, shape = `Growth\nabove\nthreshold`)) +
    geom_point() +
    xlab('Ammonium chloride (mM)') +
    ylab('Urea (mM)') +
    labs(tag = 'B') +
    scale_fill_viridis_c(guide = guide_colorbar(title = 'Norm. depth')) +
    scale_shape_manual(values = c(1, 22)) +
    scale_color_manual(values = c('gray50', 'black'),
                       guide = guide_legend(override.aes = list(shape = c(1, 15)))) +
    guides(shape = 'none', color = 'none') +
    theme(legend.position = 'bottom',
          legend.title = element_text(size = 11),
          legend.box = 'horizontal',
          legend.direction = 'horizontal',
          legend.justification = 'right')

comb_plot <- gridExtra::grid.arrange(nsf, nsd, widths = c(1, 1))


# Let's explore C:N ratios
cnsf <- ra |> 
    transform(`Growth\nabove\nthreshold` = fifelse(fitness > 0.25, T, F),
              total_carbon = (d_glucose * 6 + sodium_citrate * 6 + sodium_octanoate * 8 + sodium_acetate * 2 + sodium_benzoate * 7),
              total_nitrogen = (ammonium_chloride + 2 * urea)) |> 
    ggplot(aes(total_carbon, total_nitrogen, fill = fitness, color = `Growth\nabove\nthreshold`, shape = `Growth\nabove\nthreshold`)) +
    geom_point() +
    xlab('Total carbon (mM)') +
    ylab('Total nitrogen (mM)') +
    labs(tag = 'A') +
    scale_fill_viridis_c(option = 'magma', guide = guide_colorbar(title = 'Fitness')) +
    scale_shape_manual(values = c(1, 22)) +
    scale_color_manual(values = c('gray50', 'black'),
                       guide = guide_legend(override.aes = list(shape = c(1, 15)))) +
    theme(legend.position = 'bottom',
          legend.title = element_text(size = 11),
          legend.box = 'horizontal',
          legend.direction = 'horizontal')

cnsd <- ra |> 
    transform(`Growth\nabove\nthreshold` = fifelse(fitness > 0.25, T, F),
              total_carbon = (d_glucose * 6 + sodium_citrate * 6 + sodium_octanoate * 8 + sodium_acetate * 2 + sodium_benzoate * 7),
              total_nitrogen = (ammonium_chloride + 2 * urea)) |>     
    ggplot(aes(total_carbon, total_nitrogen, fill = norm_depth, color = `Growth\nabove\nthreshold`, shape = `Growth\nabove\nthreshold`)) +
    geom_point() +
    xlab('Total carbon (mM)') +
    ylab('Total nitrogen (mM)') +
    labs(tag = 'B') +
    scale_fill_viridis_c(guide = guide_colorbar(title = 'Norm. depth')) +
    scale_shape_manual(values = c(1, 22)) +
    scale_color_manual(values = c('gray50', 'black'),
                       guide = guide_legend(override.aes = list(shape = c(1, 15)))) +
    guides(shape = 'none', color = 'none') +
    theme(legend.position = 'bottom',
          legend.title = element_text(size = 11),
          legend.box = 'horizontal',
          legend.direction = 'horizontal',
          legend.justification = 'right')

comb_plot <- gridExtra::grid.arrange(cnsf, cnsd, widths = c(1, 1))


# what's the C:N ratio slope?
ra |> 
    transform(`Growth\nabove\nthreshold` = fifelse(fitness > 0.25, T, F),
              total_carbon = (d_glucose * 6 + sodium_citrate * 6 + sodium_octanoate * 8 + sodium_acetate * 2 + sodium_benzoate * 7),
              total_nitrogen = (ammonium_chloride + 2 * urea)) |> 
    lm(total_nitrogen ~ total_carbon, data = _) |> 
    (\(.) sprintf('C:N ratio: %0.1f', 1 / .$coefficients['total_carbon']))()



# get limits / thresholds for plotting
library(quantreg)

# quantile regression to see where fitness and depth were limited
ra |> 
    rq(fitness ~ norm_depth, data = _, tau = 0.1) |> 
    summary()



# use LASSO regression to map the most important features
library(glmnet)

# a. run initial lasso to get idea of parameter information capacity
x_vals <- ra |> 
    transform(unique_id = paste(experiment_number, round, sep = '_')) |> 
    subset(select = c(d_glucose:ammonium_chloride, unique_id)) |> 
    as.matrix(rownames = 'unique_id') |> 
    Matrix()

y_vals <- ra |> 
    subset(select = fitness) |> 
    unlist()

lasso_init <- glmnet(x_vals, y_vals, family = 'gaussian', standardize = T)
lasso_expln <- print(lasso_init)

# execute CV LASSO model - takes only a second or two
set.seed(1726)
lasso_cv <- cv.glmnet(x_vals, y_vals, type.measure = 'mse', nfolds = 5, family = 'gaussian', standardize = T)
print(lasso_cv)
plot(lasso_cv)
# (Lambda is the shrinkage parameter). So "min" option allows more parameters to minimize MSE, "1se" is more parismonious
# Allows from 5 to 7 parameters

# plot increasing explanatory power of model with more parameters
# show most parsimonious fit with red dashed lines
ggplot(lasso_expln,
       aes(Df, `%Dev`)) +
    xlab('Model DF') +
    ylab('LASSO performance\n(% deviance)') +
    geom_segment(x = 5, y = -Inf, yend = setDT(lasso_expln)[Df == 5, max(`%Dev`)], color = 'red', linetype = 2) +
    geom_segment(x = -Inf, xend = 5, y = setDT(lasso_expln)[Df == 5, max(`%Dev`)], color = 'red', linetype = 2) +
    geom_line() +
    geom_point(shape = 21)

setDT(lasso_expln)[Df == 5, max(`%Dev`)]

# use informed lambda value to identify important parameters
important_parameters <- coef(lasso_init, s = lasso_cv$lambda.1se) |>  # or lambda.min or lambda.1se
    as('matrix') |> 
    as.data.table(keep.rownames = T) |> 
    setnames(new = c('d_num', 'coefficient')) |> 
    subset(coefficient != 0) |> 
    (\(.) .[order(-coefficient)])()

important_parameters


# we will do a total salt in addition to all the other important parameters
ra |> 
    transform(total_salt = sodium_chloride + potassium_chloride) |> 
    quantreg::rq(fitness ~ total_salt, data = _, tau = 0.5) |> 
    summary()

total_salt_fig <- ra |> 
    transform(total_salt = sodium_chloride + potassium_chloride) |> 
    plot_fitness('total_salt') +
    geom_segment(aes(x = 0, xend = max(total_salt),
                     y = 4.19481, yend = 4.19481 - 0.01766 * max(total_salt)),
                 color = 'gray50') +
    # geom_abline(intercept = 4.19481, slope = -0.01766, color = 'gray50')
    xlab('Total salt (mM)') +
    labs(tag = 'F') +
    theme(legend.position = 'none')

    gridExtra::grid.arrange(
        ra |> plot_fitness('ammonium_chloride') + xlab('Ammonium chloride (mM)') + labs(tag = 'A'),
        ra |> plot_fitness('urea') + xlab('Urea (mM)') + labs(tag = 'B') + theme(legend.position = 'none'),
        ra |> plot_fitness('mme_trace_minerals') + xlab('Wolfe trace minerals') + labs(tag = 'C') + theme(legend.position = 'none'),
        ra |> transform(pH = pH + ((round - 2) / 6)) |>  plot_fitness('pH') + labs(tag = 'D') + theme(legend.position = 'none'),
        ra |> plot_fitness('sodium_octanoate') + xlab('Sodium octanoate (mM)') + labs(tag = 'E') + theme(legend.position = 'none'),
        total_salt_fig,
        widths = c(1, 1), heights = c(1, 1, 1))




# What low pH conditions produced the most growth? What contributed to their success?
ra |> 
    transform(total_salt = sodium_chloride + potassium_chloride,
              total_c = d_glucose * 6 + sodium_citrate * 6 + sodium_octanoate * 8 + sodium_acetate * 2 + sodium_benzoate * 7 + urea * 1,
              total_n = ammonium_chloride * 1 + urea * 2) |> 
    subset(pH == 5) |> 
    subset(select = c(fitness, O2:sodium_benzoate, sodium_chloride:ammonium_chloride, norm_depth:total_n)) |> 
    (\(x) cor(x))() |> 
    subset(select = fitness)
# most important parameters were sodium acetate an sodium benzoate, both of which were negatively associated with fitness

ra |> 
    subset(pH == 5) |> 
    subset(select = c(fitness, experiment_number:sodium_benzoate, sodium_chloride:ammonium_chloride)) |> 
    melt(id.vars = c('round', 'experiment_number', 'fitness'), variable.name = 'condition') |> 
    transform(growth_tf = fifelse(fitness > 1, 'growth', 'no growth')) |> 
    (\(x) x[, value := value / max(value), by = condition])() |> 
    ggplot(aes(growth_tf, value, color = growth_tf)) +
    geom_boxplot() +
    geom_jitter(width = 0.1) +
    labs(title = 'Growth vs. No-growth conditions in pH 5') +
    facet_wrap(~ condition, ncol = 4) +
    scale_color_manual(values = c('darkgreen', 'gray10')) +
    theme(axis.title.x = element_blank(),
          # axis.text.x = element_text(angle = 60, hjust = 1),
          legend.position = 'none')

 