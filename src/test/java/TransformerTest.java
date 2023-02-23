import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class TransformerTest
{
    @Test
    public void pickBestTokenTest()
    {
        float[] a = {-5.53454e-12f, 6, -7.454454e2f, 8.3454e-2f, -2};

        assertEquals(1, pickBestToken(a));
    }

    public int pickBestToken(float[] tokenProbabilities)
    {
        // This implementation simply picks up the token with the highest probability,
        // but it would be possible to implement here the topK and topP selection mechanism.

        int bestTokenId = 0;
        float maxProbability = Float.MIN_VALUE;

        for (int i = 0; i < tokenProbabilities.length; i++)
        {
            float probability = tokenProbabilities[i];

            if (probability > maxProbability)
            {
                bestTokenId = i;
                maxProbability = probability;
            }
        }

        return bestTokenId;
    }
}
