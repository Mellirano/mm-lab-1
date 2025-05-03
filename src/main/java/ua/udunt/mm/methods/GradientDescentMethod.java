package ua.udunt.mm.methods;

import ua.udunt.mm.model.FunctionSystem;
import ua.udunt.mm.util.JacobianUtil;

/**
 * Implements the Gradient Descent method for solving nonlinear systems of equations
 * by minimizing the sum of squared residuals (objective function).
 * Uses adaptive step size (lambda) and numerical Jacobian for gradient computation.
 */
public class GradientDescentMethod {

    /**
     * Tolerance for the gradient norm (‖∇P(x)‖).
     * <p>
     * If the norm of the gradient becomes smaller than this threshold,
     * the algorithm considers the point to be close enough to a stationary point
     * and stops iterating.
     * <p>
     * Typical value: 1e-6 (0.000001)
     */
    private static final double TOL_GRAD = 0.000001;

    /**
     * Tolerance for the objective function value (P(x)).
     * <p>
     * If the squared residual sum becomes smaller than this threshold,
     * the current solution is considered a accurate enough approximation
     * of the root of the system.
     * <p>
     * Typical value: 1e-6 (0.000001)
     */
    private static final double TOL_P = 0.000001;


    /**
     * Tolerance for the change in solution vector (‖Δx‖).
     * <p>
     * If the update step is smaller than this threshold,
     * the method assumes convergence and terminates.
     * <p>
     * Typical value: 1e-6 (0.000001)
     */
    private static final double TOL_X = 0.000001;

    /**
     * Minimum allowed step size (lambda) in the gradient descent update.
     * <p>
     * If the step size becomes smaller than this threshold during
     * backtracking line search, the algorithm assumes progress is no longer possible
     * and exits to prevent infinite loops with vanishing steps.
     * <p>
     * Typical value: 1e-12 (0.000000000001)
     */
    private static final double MIN_LAMBDA = 0.000000000001;

    private static final String ITERATION_HEADER =
            "Iter |       x[i]        |      delta[x[i]]       |   lambda       |     nabla[P[i]]          |   |nabla[P]|     |    P(x)   ";
    private static final String ITERATION_DIVIDER =
            "-----+-------------------+------------------+----------+---------------------+-----------+------------";

    /**
     * Solves the nonlinear system by applying the gradient descent method to the objective function:
     * P(x) = sum(f_i(x)^2).
     *
     * @param system        the function system
     * @param initialGuess  starting point for the solution
     * @param initialLambda initial learning rate (step size)
     * @param decayRate     multiplicative decay factor for lambda
     * @param maxIterations maximum number of iterations
     * @return solution vector x that approximately minimizes P(x)
     */
    public static double[] solve(FunctionSystem system, double[] initialGuess, double initialLambda, double decayRate, int maxIterations) {
        int variableCount = system.size();
        double[] currentSolution = initialGuess.clone();
        double currentObjective = objective(system, currentSolution);
        double[] gradient = gradient(system, currentSolution);
        double gradientNorm = norm(gradient);

        printHeader();

        int iteration = 0;
        double[] deltaX = new double[variableCount];
        printIteration(iteration, currentSolution, deltaX, 0.0, gradient, gradientNorm, currentObjective);

        while (iteration < maxIterations) {
            if (gradientNorm < TOL_GRAD || Math.abs(currentObjective) < TOL_P) {
                break;
            }

            double lambda = initialLambda;
            deltaX = new double[variableCount];
            double[] newSolution;
            double newObjective;

            // Backtracking line search
            do {
                newSolution = calculateNewSolution(currentSolution, gradient, lambda, deltaX);
                newObjective = objective(system, newSolution);
                lambda *= decayRate;
            } while (newObjective >= currentObjective && lambda >= MIN_LAMBDA);

            if (lambda < MIN_LAMBDA) break;

            double[] oldSolution = currentSolution.clone();
            currentSolution = newSolution;
            currentObjective = newObjective;
            gradient = gradient(system, currentSolution);
            gradientNorm = norm(gradient);
            iteration++;

            printIteration(iteration, currentSolution, deltaX, lambda, gradient, gradientNorm, currentObjective);

            double[] solutionDifference = calculateDifference(currentSolution, oldSolution);
            if (norm(solutionDifference) < TOL_X) break;
        }

        System.out.println("Convergence achieved");
        return currentSolution;
    }

    /**
     * Calculates new x based on gradient and step size lambda.
     *
     * @param currentSolution current x
     * @param gradient        current gradient
     * @param lambda          learning rate
     * @param deltaX          output delta x (filled in-place)
     * @return updated x
     */
    private static double[] calculateNewSolution(double[] currentSolution, double[] gradient, double lambda, double[] deltaX) {
        double[] newSolution = new double[currentSolution.length];
        for (int i = 0; i < currentSolution.length; i++) {
            deltaX[i] = -lambda * gradient[i];
            newSolution[i] = currentSolution[i] + deltaX[i];
        }
        return newSolution;
    }

    /**
     * Computes the difference between current and previous x.
     *
     * @param currentSolution new x
     * @param oldSolution     previous x
     * @return difference vector
     */
    private static double[] calculateDifference(double[] currentSolution, double[] oldSolution) {
        double[] difference = new double[currentSolution.length];
        for (int i = 0; i < currentSolution.length; i++) {
            difference[i] = currentSolution[i] - oldSolution[i];
        }
        return difference;
    }

    /**
     * Prints the table header for iteration output.
     */
    private static void printHeader() {
        System.out.println(ITERATION_HEADER);
        System.out.println(ITERATION_DIVIDER);
    }

    /**
     * Prints iteration details in a formatted table row.
     */
    private static void printIteration(int iteration, double[] solution, double[] deltaX, double lambda,
                                       double[] gradient, double gradientNorm, double objective) {
        System.out.printf("%3d |", iteration);
        for (double value : solution) {
            System.out.printf(" %+9.5f", value);
        }
        System.out.print(" |");
        for (double value : deltaX) {
            System.out.printf(" %+9.5f", value);
        }
        System.out.printf(" | %+8.5f |", lambda);
        for (double value : gradient) {
            System.out.printf(" %+9.5f", value);
        }
        System.out.printf(" | %+9.5f | %+9.5f%n", gradientNorm, objective);
    }

    /**
     * Computes the gradient of the objective function P(x) = sum(f_i^2)
     * using the formula: ∇P = 2 * J^T * f(x).
     *
     * @param system the function system
     * @param x      the current x
     * @return gradient vector ∇P(x)
     */
    private static double[] gradient(FunctionSystem system, double[] x) {
        int variableCount = x.length;
        double[] grad = new double[variableCount];
        double[] functionValues = system.evaluate(x);
        double[][] jacobian = JacobianUtil.computeJacobian(system, x);

        for (int j = 0; j < variableCount; j++) {
            for (int i = 0; i < variableCount; i++) {
                grad[j] += 2 * jacobian[i][j] * functionValues[i];
            }
        }
        return grad;
    }

    /**
     * Computes the Euclidean norm of a vector.
     *
     * @param vec input vector
     * @return L2 norm
     */
    private static double norm(double[] vec) {
        double sum = 0.0;
        for (double v : vec) {
            sum += v * v;
        }
        return Math.sqrt(sum);
    }

    /**
     * Computes the objective function value P(x) = ∑(f_i(x))^2.
     *
     * @param system the function system
     * @param x      the point at which to evaluate
     * @return scalar value of objective function
     */
    private static double objective(FunctionSystem system, double[] x) {
        double[] functionValues = system.evaluate(x);
        double sum = 0.0;
        for (double value : functionValues) {
            sum += value * value;
        }
        return sum;
    }

}