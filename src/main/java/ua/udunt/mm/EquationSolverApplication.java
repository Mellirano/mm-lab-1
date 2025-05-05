package ua.udunt.mm;

import ua.udunt.mm.methods.GradientDescentMethod;
import ua.udunt.mm.methods.NewtonMethod;
import ua.udunt.mm.methods.SeidelMethod;
import ua.udunt.mm.methods.SimpleIterationMethod;
import ua.udunt.mm.model.FunctionSystem;

public class EquationSolverApplication {

    private static final double[] INITIAL_GUESS = {0.5, 0.5};
    private static final double EPS = 0.0001;
    private static final int MAX_ITERATIONS = 100;
    private static final double K = 0.75;
    private static final double LAMBDA = 0.1;

    private static final FunctionSystem ITERATION_SYSTEM = new FunctionSystem() {
        @Override
        public double[] evaluate(double[] initialGuess) {
            double x = initialGuess[0]; // x
            double y = initialGuess[1]; // y

            //Canonical form
            double x1 = (2 - Math.cos(y)) / 2;
            double y2 = Math.sin(x + 1) - 1.2;

            return new double[]{x1, y2};
        }

        @Override
        public int size() {
            return 2;
        }
    };

    private static final FunctionSystem EQUATION_SYSTEM = new FunctionSystem() {
        @Override
        public double[] evaluate(double[] initialGuess) {
            double x = initialGuess[0]; // x
            double y = initialGuess[1]; // y

            double x1 = Math.tan(x * y + 0.4) - Math.pow(x, 2);
            double y2 = 0.06 * Math.pow(x, 2) + 2 * Math.pow(x, 2) - 1;

            return new double[]{x1, y2};
        }

        @Override
        public int size() {
            return 2;
        }
    };

    public static void main(String[] args) {
        System.out.println("=== Task 1 ===");

        System.out.println("\n--- Simple Iteration Method ---");
        double[] simpleIterationsResult = SimpleIterationMethod.solve(ITERATION_SYSTEM, INITIAL_GUESS, EPS, MAX_ITERATIONS);
        System.out.printf("Result (Simple Iteration Method): [%+9.5f, %+9.5f]%n", simpleIterationsResult[0], simpleIterationsResult[1]);

        System.out.println("\n--- Seidel Method ---");
        double[] seidelResult = SeidelMethod.solve(ITERATION_SYSTEM, INITIAL_GUESS, EPS, MAX_ITERATIONS);
        System.out.printf("Result (Seidel Method): [%+9.5f, %+9.5f]%n", seidelResult[0], seidelResult[1]);


        System.out.println("\n=== Task 2 ===");

        System.out.println("\n--- Newton Method ---");
        double[] newtonResult = NewtonMethod.solve(EQUATION_SYSTEM, INITIAL_GUESS, EPS, MAX_ITERATIONS);
        System.out.printf("Result (Newton Method): [%+9.5f, %+9.5f]%n", newtonResult[0], newtonResult[1]);

        System.out.println("\n--- Gradient Descent Method ---");
        double[] gradientResult = GradientDescentMethod.solve(EQUATION_SYSTEM, INITIAL_GUESS, LAMBDA, K, MAX_ITERATIONS);
        System.out.printf("Result (Gradient Descent Method): [%+9.5f, %+9.5f]%n", gradientResult[0], gradientResult[1]);
    }

}