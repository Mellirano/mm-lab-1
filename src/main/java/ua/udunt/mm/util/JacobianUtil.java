package ua.udunt.mm.util;

import ua.udunt.mm.model.FunctionSystem;

/**
 * Utility class for computing the Jacobian matrix and checking convergence conditions
 * for iterative methods of solving nonlinear systems of equations.
 */
public class JacobianUtil {

    private static final double STEP_SIZE = 0.000001;
    private static final String FORMAT_CONVERGENCE_MESSAGE = "||J|| = %.6f -> %s%n";

    /**
     * Computes the Jacobian matrix for a given system of functions at a specified point.
     *
     * @param system    the function system
     * @param variables the point at which the Jacobian is computed
     * @return the computed Jacobian matrix
     */
    public static double[][] computeJacobian(FunctionSystem system, double[] variables) {
        int size = system.size();
        double[][] jacobian = new double[size][size];
        double[] functionValues = system.evaluate(variables);

        for (int i = 0; i < size; i++) {
            double[] perturbedVariables = variables.clone();
            perturbedVariables[i] += STEP_SIZE;
            double[] perturbedFunctionValues = system.evaluate(perturbedVariables);

            for (int j = 0; j < size; j++) {
                jacobian[j][i] = (perturbedFunctionValues[j] - functionValues[j]) / STEP_SIZE;
            }
        }
        return jacobian;
    }

    /**
     * Checks the convergence condition based on the infinity norm of the Jacobian matrix
     * and prints the result and matrix to the console.
     *
     * @param system           the function system
     * @param initialVariables the point at which convergence is evaluated
     */
    public static void checkConvergence(FunctionSystem system, double[] initialVariables) {
        double[][] jacobian = computeJacobian(system, initialVariables);
        printJacobianMatrix(jacobian);
        double maxRowSum = calculateMaxRowSum(jacobian);
        boolean isConvergenceLikely = maxRowSum < 1.0;

        System.out.printf(FORMAT_CONVERGENCE_MESSAGE, maxRowSum,
                isConvergenceLikely ? "CONVERGENCE LIKELY" : "NO GUARANTEE OF CONVERGENCE");
    }

    /**
     * Calculates the infinity norm (maximum absolute row sum) of the matrix.
     *
     * @param matrix the matrix to evaluate
     * @return the maximum row sum
     */
    private static double calculateMaxRowSum(double[][] matrix) {
        double maxSum = 0.0;
        for (double[] row : matrix) {
            double rowSum = 0.0;
            for (double value : row) {
                rowSum += Math.abs(value);
            }
            maxSum = Math.max(maxSum, rowSum);
        }
        return maxSum;
    }

    /**
     * Prints the Jacobian matrix in a formatted layout to the console.
     *
     * @param matrix the matrix to print
     */
    public static void printJacobianMatrix(double[][] matrix) {
        System.out.println("Jacobian matrix: ");
        for (double[] row : matrix) {
            for (double value : row) {
                System.out.printf("  %.6f", value);
            }
            System.out.println();
        }
    }

}