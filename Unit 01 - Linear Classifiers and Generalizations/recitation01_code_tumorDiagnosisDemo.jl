# Scikit-Learn:
#  - Python: https://scikit-learn.org/stable/
#  - Julia port/wrapper: https://github.com/cstjean/ScikitLearn.jl

# Import libraries
using Statistics, StatsPlots, ScikitLearn
using ScikitLearn.CrossValidation: cross_val_score
@sk_import datasets: load_breast_cancer
@sk_import preprocessing : scale
@sk_import linear_model: SGDClassifier

# Load and prepare data
cancer_data = load_breast_cancer()
y = cancer_data["target"] # Training labels ('malignant = 0', 'benign = 1')
X = cancer_data["data"] # 30 attributes; https://scikit-learn.org/stable/datasets/index.html#breast-cancer-dataset
X = scale(X) # scale each data attribute to zero-mean and unit variance

# Plot the first 2 attributes of training points
colors = [y == 1 ? "blue" : "red" for y in y]
labels = [y == 1 ? "benign" : "malign" for y in y]
scatter(X[:,1],X[:,2], colour=colors, title="Classified tumors",xlabel="Tumor Radius", ylabel="Tumor Texture", group=labels)


# Compute cross-validation scores
alpha = eps():0.005:1 # Range of hyperparameter values 1E-15 to 1 by 0.005
val_scores = zeros(length(alpha)) # Initialize validation score for each alpha value
for (i,a) in enumerate(alpha) # for each alpha value
    # Set up SVM with hinge loss and l2 norm regularization
    model = SGDClassifier(loss="hinge", penalty="l2", alpha=a)
    # Calculate cross validation scores for 5-fold cross-validation
    score = cross_val_score(model, X, y; cv=5)
    val_scores[i] = mean(score) # Calculate mean of the 5 scores
end

# Plot how cross-validation score changes with alpha
plot(alpha,val_scores,xlims=(0,1), xlabel="alpha",ylabel="Mean Cross-Validation Accuracy",label="")

# Determine the alpha that maximizes the cross-validation score
ind = argmax(val_scores)
alpha_star = alpha[ind]
println("alpha_star = $alpha_star")
vline!([alpha_star], label="alpha*")

# Train model with alpha_star
model_star = SGDClassifier(loss="hinge", penalty="l2", alpha=alpha_star)
model_trained = model_star.fit(X,y)
print("Training Accuracy = $(model_trained.score(X,y))")

# Plot decision boundary of trained model
slope = model_trained.coef_[1,2]/-model_trained.coef_[1,1]
x1 = -10:0.5:10
y1 = slope*x1
scatter(X[:,1],X[:,2], colour=colors, title="Classified tumors",xlims=(-4,4),ylims=(-6,6), xlabel="Tumor Radius", ylabel="Tumor Texture", group=labels)
plot!(x1,y1, linestyle = :dot, label="classifier")
