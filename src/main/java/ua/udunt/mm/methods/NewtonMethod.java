package ua.udunt.mm.methods;

import ua.udunt.mm.model.FunctionSystem;
import ua.udunt.mm.util.JacobianUtil;

/**
 * Implements the Newton method for solving nonlinear systems of equations.
 * Uses numerical Jacobian computation and Gaussian elimination for solving the linear system.
 */
public class NewtonMethod {

    private static final String ITERATION_HEADER =
            "Iter |       x[i]        |      delta[x[i]]       |     F[i]        |   ||delta[x]||   |   ||F||   ";
    private static final String ITERATION_DIVIDER =
            "-----+-------------------+------------------+------------------+-----------+----------";

    /**
     * Solves the given nonlinear system using Newton's method.
     *
     * @param system        the system of equations to solve
     * @param initialGuess  the initial approximation
     * @param tolerance     the convergence tolerance
     * @param maxIterations the maximum number of iterations allowed
     * @return solution vector approximating the root
     */
    public static double[] solve(FunctionSystem system, double[] initialGuess, double tolerance, int maxIterations) {
        double[] currentX = initialGuess.clone();
        printIterationHeader();

        for (int iteration = 0; iteration < maxIterations; iteration++) {
            double[] functionValues = system.evaluate(currentX);
            double[][] jacobian = JacobianUtil.computeJacobian(system, currentX);
            double[] deltaX = solveLinearSystem(jacobian, negateArray(functionValues));

            updateSolution(currentX, deltaX);

            double deltaNorm = calculateNorm(deltaX);
            double functionNorm = calculateNorm(functionValues);

            printIterationInfo(iteration + 1, currentX, deltaX, functionValues, deltaNorm, functionNorm);

            if (hasConverged(deltaNorm, functionNorm, tolerance)) {
                System.out.println("Convergence achieved");
                break;
            }
        }

        return currentX;
    }

    /**
     * Negates the elements of the input array.
     *
     * @param array input array
     * @return negated array
     */
    private static double[] negateArray(double[] array) {
        double[] result = new double[array.length];
        for (int i = 0; i < array.length; i++) {
            result[i] = -array[i];
        }
        return result;
    }

    /**
     * Solves a linear system Ax = b using Gaussian elimination with partial pivoting.
     *
     * @param coefficients matrix A
     * @param constants    vector b
     * @return solution vector x
     */
    private static double[] solveLinearSystem(double[][] coefficients, double[] constants) {
        int size = constants.length;
        for (int pivot = 0; pivot < size; pivot++) {
            pivotRows(coefficients, constants, pivot);
            eliminateBelowPivot(coefficients, constants, pivot);
        }
        return backSubstitution(coefficients, constants);
    }

    /**
     * Swaps the pivot row with the row of maximum value in the current column.
     */
    private static void pivotRows(double[][] coefficients, double[] constants, int pivot) {
        int size = constants.length;
        int maxRow = pivot;
        for (int i = pivot + 1; i < size; i++) {
            if (Math.abs(coefficients[i][pivot]) > Math.abs(coefficients[maxRow][pivot])) {
                maxRow = i;
            }
        }
        swap(coefficients, constants, pivot, maxRow);
    }

    /**
     * Performs row elimination below the pivot.
     */
    private static void eliminateBelowPivot(double[][] coefficients, double[] constants, int pivot) {
        int size = constants.length;
        for (int row = pivot + 1; row < size; row++) {
            double factor = coefficients[row][pivot] / coefficients[pivot][pivot];
            constants[row] -= factor * constants[pivot];
            for (int col = pivot; col < size; col++) {
                coefficients[row][col] -= factor * coefficients[pivot][col];
            }
        }
    }

    /**
     * Backward substitution step to solve an upper triangular system.
     */
    private static double[] backSubstitution(double[][] coefficients, double[] constants) {
        int size = constants.length;
        double[] solution = new double[size];
        for (int i = size - 1; i >= 0; i--) {
            double sum = constants[i];
            for (int j = i + 1; j < size; j++) {
                sum -= coefficients[i][j] * solution[j];
            }
            solution[i] = sum / coefficients[i][i];
        }
        return solution;
    }

    /**
     * Swaps two rows in the matrix and right-hand side vector.
     */
    private static void swap(double[][] coefficients, double[] constants, int row1, int row2) {
        double[] tempRow = coefficients[row1];
        coefficients[row1] = coefficients[row2];
        coefficients[row2] = tempRow;

        double tempValue = constants[row1];
        constants[row1] = constants[row2];
        constants[row2] = tempValue;
    }

    /**
     * Computes the Euclidean norm (L2) of a vector.
     */
    private static double calculateNorm(double[] array) {
        double sum = 0.0;
        for (double value : array) {
            sum += value * value;
        }
        return Math.sqrt(sum);
    }

    /**
     * Updates the current approximation with the correction vector.
     */
    private static void updateSolution(double[] currentX, double[] deltaX) {
        for (int i = 0; i < currentX.length; i++) {
            currentX[i] += deltaX[i];
        }
    }

    /**
     * Checks whether convergence criteria are satisfied.
     */
    private static boolean hasConverged(double deltaNorm, double functionNorm, double tolerance) {
        return deltaNorm < tolerance && functionNorm < tolerance;
    }

    /**
     * Prints the header for the iteration output table.
     */
    private static void printIterationHeader() {
        System.out.println(ITERATION_HEADER);
        System.out.println(ITERATION_DIVIDER);
    }

    /**
     * Prints information about the current iteration in formatted columns.
     */
    private static void printIterationInfo(int iteration, double[] x, double[] deltaX, double[] functionValues,
                                           double deltaNorm, double functionNorm) {
        System.out.printf("%3d |", iteration);
        for (double value : x) {
            System.out.printf(" %+9.5f", value);
        }
        System.out.print(" |");
        for (double value : deltaX) {
            System.out.printf(" %+9.5f", value);
        }
        System.out.print(" |");
        for (double value : functionValues) {
            System.out.printf(" %+9.5f", value);
        }
        System.out.printf(" | %+9.5f | %+9.5f%n", deltaNorm, functionNorm);
    }

}