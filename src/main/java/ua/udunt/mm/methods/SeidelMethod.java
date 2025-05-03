package ua.udunt.mm.methods;

import ua.udunt.mm.model.FunctionSystem;
import ua.udunt.mm.util.JacobianUtil;

/**
 * Implements the Seidel (Gauss–Seidel) method for solving nonlinear systems of equations
 * using fixed-point iteration with immediate updates of variables.
 */
public class SeidelMethod {

    private static final String ITERATION_HEADER =
            "Iter |       x[i]        |      delta[x[i]]       |     F[i]        |   ||delta[x]||   |   ||F||   ";
    private static final String ITERATION_DIVIDER =
            "-----+-------------------+------------------+------------------+-----------+----------";

    /**
     * Solves the system using the Seidel iteration method.
     *
     * @param system       the function system to solve
     * @param initialGuess the initial guess for the variables
     * @param eps          the convergence tolerance
     * @param maxIter      the maximum number of iterations
     * @return the computed solution vector
     */
    public static double[] solve(FunctionSystem system, double[] initialGuess, double eps, int maxIter) {
        int n = system.size();
        double[] currentX = initialGuess.clone();
        double[] previousX = new double[n];
        double[] residuals;

        JacobianUtil.checkConvergence(system, currentX);
        System.out.println(ITERATION_HEADER);
        System.out.println(ITERATION_DIVIDER);

        for (int iter = 0; iter < maxIter; iter++) {
            System.arraycopy(currentX, 0, previousX, 0, n);
            updateSolution(system, currentX, n);
            residuals = computeResidualsAndDeltaX(currentX, previousX, n);

            double normDeltaX = norm(residuals);
            double normResiduals = norm(residuals);

            printIterationInfo(iter + 1, currentX, residuals, residuals, normDeltaX, normResiduals);

            if (normDeltaX < eps && normResiduals < eps) {
                System.out.println("Convergence achieved");
                break;
            }
        }

        return currentX;
    }

    /**
     * Updates the solution vector using the Gauss–Seidel approach:
     * each new value is used immediately in later calculations.
     *
     * @param system   the function system
     * @param currentX the current solution vector, updated in-place
     * @param size     the number of variables
     */
    private static void updateSolution(FunctionSystem system, double[] currentX, int size) {
        for (int i = 0; i < size; i++) {
            double[] tempX = currentX.clone();
            double[] functionResults = system.evaluate(tempX);
            currentX[i] = functionResults[i];
        }
    }

    /**
     * Computes the difference between current and previous approximations
     * and returns it as residuals.
     *
     * @param currentX  current approximation
     * @param previousX previous approximation
     * @param size      number of variables
     * @return residuals vector (delta x)
     */
    private static double[] computeResidualsAndDeltaX(double[] currentX, double[] previousX, int size) {
        double[] residuals = new double[size];
        for (int i = 0; i < size; i++) {
            residuals[i] = currentX[i] - previousX[i];
        }
        return residuals;
    }

    /**
     * Prints formatted output of a single iteration, including current variable values,
     * step size (delta), residuals, and their norms.
     *
     * @param iteration     the current iteration number
     * @param x             the solution vector
     * @param deltaX        the change in a solution
     * @param residuals     the computed residuals
     * @param normDeltaX    the norm of deltaX
     * @param normResiduals the norm of residuals
     */
    private static void printIterationInfo(int iteration, double[] x, double[] deltaX, double[] residuals,
                                           double normDeltaX, double normResiduals) {
        System.out.printf("%3d |", iteration);
        for (double xi : x) System.out.printf(" %+9.5f", xi);
        System.out.print(" |");
        for (double dxi : deltaX) System.out.printf(" %+9.5f", dxi);
        System.out.print(" |");
        for (double fi : residuals) System.out.printf(" %+9.5f", fi);
        System.out.printf(" | %+9.5f | %+9.5f%n", normDeltaX, normResiduals);
    }

    /**
     * Computes the Euclidean norm (L2 norm) of a vector.
     *
     * @param v the vector
     * @return the norm
     */
    private static double norm(double[] v) {
        double sum = 0.0;
        for (double val : v) sum += val * val;
        return Math.sqrt(sum);
    }

}