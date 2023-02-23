import org.nd4j.linalg.factory.Nd4j;

public class UtilNd4j implements Util
{
    @Override
    public float[] addVectors(float[] vector1, float[] vector2)
    {
        return Nd4j.create(vector1).add(Nd4j.create(vector2)).toFloatVector();
    }

    @Override
    public float dotProduct(float[] vector1, float[] vector2)
    {
        return Nd4j.create(vector1).mmul(Nd4j.create(vector2)).getFloat(0);
    }

    @Override
    public float[] multiplyVectorByScalar(float[] vector, float scalar)
    {
        return Nd4j.create(vector).mul(scalar).toFloatVector();
    }

    @Override
    public float[] multiplyVectorByMatrix(float[] vector, float[][] matrix)
    {
        return Nd4j.create(new float[][] {vector}).mmul(Nd4j.create(matrix)).toFloatVector();
    }

    @Override
    public float[] multiplyVectorByTransposedMatrix(float[] vector, float[][] matrix)
    {
        float[][] a2 = new float[1][vector.length];
        a2[0] = vector;

        return Nd4j.create(a2).mmul(Nd4j.create(matrix).transpose()).toFloatVector();
    }

    @Override
    public float[][] splitVector(float[] vector, int count)
    {
        return Nd4j.create(vector).reshape(count, vector.length / count).toFloatMatrix();
    }

    @Override
    public float[] flattenMatrix(float[][] matrix)
    {
        return Nd4j.create(matrix).reshape(matrix.length * matrix[0].length).toFloatVector();
    }

    @Override
    public float average(float[] vector)
    {
        return Nd4j.create(vector).meanNumber().floatValue();
    }

    @Override
    public float[] softmax(float[] vector)
    {
        // TODO: Possibly there is a faster Nd4j implementation, now I simply called the standard implementation
        return UtilType.STANDARD.util.softmax(vector);
    }

    @Override
    public float[] normalize(float[] vector, float epsilon)
    {
        // TODO: Possibly there is a faster Nd4j implementation, now I simply called the standard implementation
        return UtilType.STANDARD.util.normalize(vector, epsilon);
    }

    @Override
    public float[][] sort(float[][] matrix)
    {
        return Nd4j.sortRows(Nd4j.create(matrix), 0, false).toFloatMatrix();
    }

    @Override
    public int weightedRandomPick(float[] probabilities)
    {
        // TODO: Possibly there is a faster Nd4j implementation, now I simply called the standard implementation
        return UtilType.STANDARD.util.weightedRandomPick(probabilities);
    }
}
