package ai.demo.gpt2.util;

import org.junit.Test;

public class StandardUtilTest
{
    private static final Util standardUtil = new StandardUtil();
    private static final Util nd4jUtil = new Nd4jUtil();

    @Test
    public void addVectorsTest()
    {
        UtilTest.addVectorsTest(standardUtil);
        UtilTest.addVectorsTest(nd4jUtil);
    }

    @Test
    public void multiplyVectorByScalarTest()
    {
        UtilTest.multiplyVectorByScalarTest(standardUtil);
        UtilTest.multiplyVectorByScalarTest(nd4jUtil);
    }

    @Test
    public void dotProductTest()
    {
        UtilTest.dotProductTest(standardUtil);
        UtilTest.dotProductTest(nd4jUtil);
    }

    @Test
    public void multiplyVectorByMatrixTest()
    {
        UtilTest.multiplyVectorByMatrixTest(standardUtil);
        UtilTest.multiplyVectorByMatrixTest(nd4jUtil);
    }

    @Test
    public void multiplyVectorByTransposedMatrixTest()
    {
        UtilTest.multiplyVectorByTransposedMatrixTest(standardUtil);
        UtilTest.multiplyVectorByTransposedMatrixTest(nd4jUtil);
    }

    @Test
    public void splitVectorTest()
    {
        UtilTest.splitVectorTest(standardUtil);
        UtilTest.splitVectorTest(nd4jUtil);
    }

    @Test
    public void flattenMatrixTest()
    {
        UtilTest.flattenMatrixTest(standardUtil);
        UtilTest.flattenMatrixTest(nd4jUtil);
    }

    @Test
    public void averageTest()
    {
        UtilTest.averageTest(standardUtil);
        UtilTest.averageTest(nd4jUtil);
    }

    @Test
    public void softmaxTest()
    {
        UtilTest.softmaxTest(standardUtil);
        UtilTest.softmaxTest(nd4jUtil);
    }
}

