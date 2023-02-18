package ai.demo.gpt2.util;

public interface Util
{
    float[] addVectors(float[] vector1, float[] vector2);

    float dotProduct(float[] vector1, float[] vector2);

    float[] multiplyVectorByScalar(float[] vector, float scalar);

    float[] multiplyVectorByMatrix(float[] vector, float[][] matrix);

    float[] multiplyVectorByTransposedMatrix(float[] vector, float[][] matrix);

    float[][] splitVector(float[] vector, int count);

    float[] flattenMatrix(float[][] matrix);

    float average(float[] vector);

    float[] softmax(float[] vector);

    float[] normalize(float[] vector, float epsilon);

    float[][] sort(float[][] matrix);

    int weightedRandomPick(float[] probabilities);
}
