package ua.udunt.mm.model;

public interface FunctionSystem {

    double[] evaluate(double[] initialGuess);

    int size();

}
