# function which calculates the standardized depth of the experiment based on expected effect on growth
# depth = distance from ideal conditions
# I.e., greater depth should produce lower growth
# value ranges are normalized to 1 prior to depth calculation
calc_depth <- function(conds, vals) {
    nom_val = numeric(1)
    max_val = numeric(1)
    depth = 0
    ci <- character(1)
    vi <- numeric(1)
    di <- numeric(1)
    for(i in 1:length(conds)) {
        ci <- conds[i]
        vi <- vals[i]
        #
        nom_val <- ingred[INGREDIENT == ci]$NOMINAL_VALUE
        max_val <- ingred[INGREDIENT == ci]$MAX_VALUE
        # calculate depth
        if(nom_val == max_val) {
            di <- ((max_val - vi) / max_val)
        } else if(nom_val == 0) {
            di <- (vi / max_val)
        } else if(ci == 'pH') {
            di <- (abs(vi - 7) / 2)
        }
        depth <- depth + round(di, digits = 3)
    }
    return(depth)
}