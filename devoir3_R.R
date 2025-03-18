#------Code en R-------#
set.seed(2405979)
mu <- 0  
sigma <- 1  
sigma_epsilon <- 1  
x <- rnorm(n = 100, mean = mu, sd = sigma)
e <- rnorm(n = 100, mean = 0, sd = sigma_epsilon)
beta_0 <- 1
beta_1 <- 2
beta_2 <- -2
beta_3 <- -0.5
y <- beta_0 + beta_1 * x + beta_2 * x^2 + beta_3 * x^3 + e

#voir section  6.5 Lab: Linear Models and Regularization Methods dans ISLr
X_matrix <- poly(x, 10, raw = TRUE)
data <- data.frame(y = y, X_matrix)
#https://www.statology.org/regsubsets-in-r/
best_subset <- regsubsets(y ~ ., data = data, nvmax = 10, method = "exhaustive")
summary_best <- summary(best_subset)
#Trouver les valeurs minimales et plot comme demandé dans ISLr
#pour les valeurs de cp,bic etc on peut les extraire de summary_best
cp_minimal <- which.min(summary_best$cp)
#----cp-----#
#https://ggplot2.tidyverse.org/reference/ggplot.html
# Note you will need to use the data.frame() function to create a single data set containing both X and Y .
cp_minimal <- which.min(summary_best$cp)
data.frame(cp = summary_best$cp, subset_size = 1:10) %>%
  ggplot(aes(x = subset_size, y = cp)) + 
  geom_line(color = "purple") + 
  geom_point(size = 2, color = "blue") +
  geom_point(aes(x = cp_minimal, y = summary_best$cp[cp_minimal]), color = "red", size = 10, shape = 4) +
  labs(x = "Number of Predictors", y = "Cp")

# ---- BIC ----
bic_minimal <- which.min(summary_best$bic)
data.frame(bic = summary_best$bic, subset_size = 1:10) %>%
  ggplot(aes(x = subset_size, y = bic)) + 
  geom_line(color = "purple") + 
  geom_point(size = 2, color = "blue") +
  geom_point(aes(x = bic_minimal, y = summary_best$bic[bic_minimal]), color = "red", size = 10, shape = 4) +
  labs( x = "Number of Predictors", y = "BIC")

# ---- Adjusted R^2 ----
r_carre_ajuste_maximal <- which.max(summary_best$adjr2)
data.frame(adj_r2 = summary_best$adjr2, subset_size = 1:10) %>%
  ggplot(aes(x = subset_size, y = adj_r2)) + 
  geom_line(color = "purple") + 
  geom_point(size = 2, color = "blue") +
  geom_point(aes(x = r_carre_ajuste_maximal, y = summary_best$adjr2[r_carre_ajuste_maximal]), color = "red", size = 10, shape = 4) +
  labs( x = "Number of Predictors", y = "Adjusted R-squared")
cat ("Optimal: \n")
cat("Cp:", cp_minimal, "BIC:", bic_minimal, "Adjusted R^2:", r_carre_ajuste_maximal, "\n")
#ici on va procéeder de la même manière mais en prenant la méthode forward et backward
#forward stepwise selection
model_fwd <- regsubsets(y ~ ., data = data, nvmax = 10, method = "forward")
summary_fwd <- summary(model_fwd)

cp_minimal_fwd <- which.min(summary_fwd$cp)
data.frame(cp = summary_fwd$cp, subset_size = 1:10) %>%
  ggplot(aes(x = subset_size, y = cp)) + 
  geom_line(color = "purple") + 
  geom_point(size = 2, color = "blue") +
  geom_point(aes(x = cp_minimal_fwd, y = summary_fwd$cp[cp_minimal_fwd]), color = "red", size = 10, shape = 4) +
  labs(x = "Number of Predictors", y = "Cp")
bic_minimal_fwd <- which.min(summary_fwd$bic)
data.frame(bic = summary_fwd$bic, subset_size = 1:10) %>%
  ggplot(aes(x = subset_size, y = bic)) + 
  geom_line(color = "purple") + 
  geom_point(size = 2, color = "blue") +
  geom_point(aes(x = bic_minimal_fwd, y = summary_fwd$bic[bic_minimal_fwd]), color = "red", size = 10, shape = 4) +
  labs(x = "Number of Predictors", y = "BIC")
r_carre_ajuste_maximal_fwd <- which.max(summary_fwd$adjr2)
data.frame(adj_r2 = summary_fwd$adjr2, subset_size = 1:10) %>%
  ggplot(aes(x = subset_size, y = adj_r2)) + 
  geom_line(color = "purple") + 
  geom_point(size = 2, color = "blue") +
  geom_point(aes(x = r_carre_ajuste_maximal_fwd, y = summary_fwd$adjr2[r_carre_ajuste_maximal_fwd]), color = "red", size = 10, shape = 4) +
  labs(x = "Number of Predictors", y = "Adjusted R-squared")

cat ("forward: \n")
cat("Cp:", cp_minimal_fwd, "BIC:", bic_minimal_fwd, "Adjusted R^2:", r_carre_ajuste_maximal_fwd, "\n")

#backward stepwise selection
model_bwd <- regsubsets(y ~ ., data = data, nvmax = 10, method = "backward")
summary_bwd <- summary(model_bwd)
cp_minimal_bwd <- which.min(summary_bwd$cp)
bic_minimal_bwd <- which.min(summary_bwd$bic)
r_carre_ajuste_maximal_bwd <- which.max(summary_bwd$adjr2)
data.frame(cp = summary_bwd$cp, subset_size = 1:10) %>%
  ggplot(aes(x = subset_size, y = cp)) + 
  geom_line(color = "purple") + 
  geom_point(size = 2, color = "blue") +
  geom_point(aes(x = cp_minimal_bwd, y = summary_bwd$cp[cp_minimal_bwd]), color = "red", size = 10, shape = 4) +
  labs(x = "Number of Predictors", y = "Cp")

data.frame(bic = summary_bwd$bic, subset_size = 1:10) %>%
  ggplot(aes(x = subset_size, y = bic)) + 
  geom_line(color = "purple") + 
  geom_point(size = 2, color = "blue") +
  geom_point(aes(x = bic_minimal_bwd, y = summary_bwd$bic[bic_minimal_bwd]), color = "red", size = 4, shape = 4) +
  labs(x = "Number of Predictors", y = "BIC")
data.frame(adj_r2 = summary_bwd$adjr2, subset_size = 1:10) %>%
  ggplot(aes(x = subset_size, y = adj_r2)) + 
  geom_line(color = "purple") + 
  geom_point(size = 2, color = "blue") +
  geom_point(aes(x = r_carre_ajuste_maximal_bwd, y = summary_bwd$adjr2[r_carre_ajuste_maximal_bwd]), color = "red", size = 10, shape = 4) +
  labs(x = "Number of Predictors", y = "Adjusted R-squared")
cat ("backward: \n")
cat("Cp:", cp_minimal_bwd, "BIC:", bic_minimal_bwd, "Adjusted R^2:", r_carre_ajuste_maximal_bwd, "\n")

print(coef(best_subset, id = cp_minimal))
print(coef(best_subset, id = bic_minimal))
print(coef(best_subset, id = r_carre_ajuste_maximal))
print(coef(model_fwd, id = cp_minimal_fwd))
print(coef(model_fwd, id = bic_minimal_fwd))
print(coef(model_fwd, id = r_carre_ajuste_maximal_fwd))
print(coef(model_bwd, id = cp_minimal_bwd))
print(coef(model_bwd, id = bic_minimal_bwd))
print(coef(model_bwd, id = r_carre_ajuste_maximal_bwd))
#voir apd de page 282 sur ISLR
#We first set a random seed so that the results obtained will be reproducible. -> juste moi matricule
set.seed(2405979)
#ridge.mod <-glmnet(x, y, alpha = 0, lambda = grid)
#lasso.mod <-glmnet(x[train, ], y[train], alpha = 1,lambda = grid)

#fit a lasso model to the simulated data, Use cross-validation to select the optimal value of λ
#Create plots of the cross-validation error as a function of λ. Report the resulting coefficient estimates, and discuss the results obtained.
#as for question 1 I will use 5-Kfold
model_lasso <- cv.glmnet(y = y, x = X_matrix, alpha = 1, lambda = 10^seq(1, -2, length = 100),  standardize = TRUE, nfolds = 5)
plot_data <- data.frame(lambda = model_lasso$lambda, cv_mse = model_lasso$cvm, nonzero_coeff = model_lasso$nzero)
#https://www.statology.org/lasso-regression-in-r/
#on va refaire comme les autres questions avec mse et lambda
ggplot(plot_data, aes(x = lambda, y = cv_mse)) + 
  geom_line(color = "purple") + 
  geom_point(size = 2, color = "blue") +
  geom_point(aes(x = model_lasso$lambda.min, y = min(model_lasso$cvm)), color = "red", size = 10, shape = 4) +
  scale_x_continuous(trans = 'log10')+ 
  scale_y_continuous(trans = 'log10')+ 
  labs(x = "Lambda", y = "Cross-Validation MSE")
cat("Lambda optimal: \n")
cat(model_lasso$lambda.min)
meilleur_lasso <- glmnet(y = y, x = X_matrix, alpha = 1)
coefficient_pour_lasso <- predict(meilleur_lasso, s=model_lasso$lambda.min, type = "coefficients")
print(coefficient_pour_lasso)
#Now generate a response vector Y according to the model Y = b0 + b7*x^7 +e
set.seed(2405979)
beta_0 <- 10
beta_7 <- 5
y <- beta_0 + beta_7*x^7 + e
#on refait tout pour ce modèle
#perform best subset selection and the lasso. Discuss the results obtained.
X_matrix <- poly(x, 10, raw = TRUE)
model_best <- regsubsets(y ~ ., data = X_matrix, nvmax = 10, method = "exhaustive")
model_summary <- summary(model_best)

#on extrait à nv -> copié collé du premier code
cp_minimal <- which.min(summary_best$cp)
#----cp-----#
#https://ggplot2.tidyverse.org/reference/ggplot.html
# Note you will need to use the data.frame() function to create a single data set containing both X and Y .
cp_minimal <- which.min(summary_best$cp)
data.frame(cp = summary_best$cp, subset_size = 1:10) %>%
  ggplot(aes(x = subset_size, y = cp)) + 
  geom_line(color = "purple") + 
  geom_point(size = 2, color = "blue") +
  geom_point(aes(x = cp_minimal, y = summary_best$cp[cp_minimal]), color = "red", size = 10, shape = 4) +
  labs( x = "Number of Predictors", y = "Cp")

# ---- BIC ----
bic_minimal <- which.min(summary_best$bic)
data.frame(bic = summary_best$bic, subset_size = 1:10) %>%
  ggplot(aes(x = subset_size, y = bic)) + 
  geom_line(color = "purple") + 
  geom_point(size = 2, color = "blue") +
  geom_point(aes(x = bic_minimal, y = summary_best$bic[bic_minimal]), color = "red", size = 10, shape = 4) +
  labs( x = "Number of Predictors", y = "BIC")

# ---- Adjusted R^2 ----
r_carre_ajuste_maximal <- which.max(summary_best$adjr2)
data.frame(adj_r2 = summary_best$adjr2, subset_size = 1:10) %>%
  ggplot(aes(x = subset_size, y = adj_r2)) + 
  geom_line(color = "purple") + 
  geom_point(size = 2, color = "blue") +
  geom_point(aes(x = r_carre_ajuste_maximal, y = summary_best$adjr2[r_carre_ajuste_maximal]), color = "red", size = 10, shape = 4) +
  labs( x = "Number of Predictors", y = "Adjusted R-squared")
cat ("Optimal: \n")
cat("Cp:", cp_minimal, "BIC:", bic_minimal, "Adjusted R^2:", r_carre_ajuste_maximal, "\n")



model_lasso <- cv.glmnet(y = y, x = X_matrix, alpha = 1, lambda = 10^seq(1, -2, length = 100), standardize = TRUE, nfolds = 5)
plot_data <- data.frame(lambda = model_lasso$lambda, cv_mse = model_lasso$cvm, nonzero_coeff = model_lasso$nzero)
#https://www.statology.org/lasso-regression-in-r/
#on va refaire comme les autres questions avec mse et lambda
ggplot(plot_data, aes(x = lambda, y = cv_mse)) + 
  geom_line(color = "purple") + 
  geom_point(size = 2, color = "blue") +
  geom_point(aes(x = model_lasso$lambda.min, y = min(model_lasso$cvm)), color = "red", size = 10, shape = 4) +
  scale_x_continuous(trans = 'log10')+ 
  scale_y_continuous(trans = 'log10')+ 
  labs(x = "Lambda", y = "Cross-Validation MSE")
  cat("Lambda optimal: \n")
cat(model_lasso$lambda.min)
meilleur_lasso <- glmnet(y = y, x = X_matrix, alpha = 1)
coefficient_pour_lasso <- predict(meilleur_lasso, s=model_lasso$lambda.min, type = "coefficients")
print(coefficient_pour_lasso)
