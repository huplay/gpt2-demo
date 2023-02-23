import java.util.ArrayList;
import java.util.List;

import static java.lang.Math.*;
import static java.lang.Math.pow;

public class TransformerDecoder
{
    private final int embeddingSize;
    private final int headCount;
    private final int attentionDividend;

    private final Parameters.DecoderParameters params;
    private final Util util;

    private final float epsilon;

    private final List<float[][]> storedKeys = new ArrayList<>();
    private final List<float[][]> storedValues = new ArrayList<>();

    public TransformerDecoder(Config config, Parameters.DecoderParameters params, float epsilon)
    {
        this.embeddingSize = config.modelType.embeddingSize;
        this.headCount = config.modelType.headCount;

        // A fejenkénti (head-enkénti) vektorméret mindig 64, tehát a gyök vektorméret mindig 8, így gond nélkül egésszé konvertálhatjuk
        this.attentionDividend = (int) sqrt(embeddingSize / headCount);

        this.params = params;
        this.util = config.utilType.util;
        this.epsilon = epsilon;
    }

    /**
     * Dekóder logika
     *
     * @param input - bemenő embedding értékek
     * @return kimenő embedding értékek
     */
    public float[] calculate(float[] input)
    {
        // Figyelem (attention) mechanizmus
        float[] result = attentionBlock(input);

        // Előrecsatolt (egyszerű) neurális hálózat
        return feedForwardBlock(result);
    }

    private float[] attentionBlock(float[] input)
    {
        // Normalizáció
        float[] result = normalize(input, params.norm1Weights, params.norm1Biases);

        // Attention-réteg
        result = attention(result);

        // Residual connection, vagyis az átalakítás előtti érték hozzáadása az eredményhez
        return util.addVectors(result, input);
    }

    private float[] feedForwardBlock(float[] input)
    {
        // Normalizáció
        float[] result = normalize(input, params.norm2Weights, params.norm2Biases);

        // Neurális hálózat rétegei
        result = feedForward(result);

        // Residual connection, vagyis az átalakítás előtti érték hozzáadása az eredményhez
        return util.addVectors(result, input);
    }

    private float[] normalize(float[] input, float[] weights, float[] biases)
    {
        // Standard normalizáció
        float[] result = util.normalize(input, epsilon);

        // A tanulás során előállított súlyok és eltolás hozzáadása
        for (int i = 0; i < input.length; i++)
        {
            result[i] = result[i] * weights[i] + biases[i];
        }

        return result;
    }

    private float[] attention(float[] embedding)
    {
        // Az aktuális tokenre vonatkozó query, key és value vektorok előállítása:
        float[] query = util.multiplyVectorByMatrix(embedding, params.queryWeighs);
        query = util.addVectors(query, params.queryBiases);

        float[] key = util.multiplyVectorByMatrix(embedding, params.keyWeighs);
        key = util.addVectors(key, params.keyBiases);

        float[] value = util.multiplyVectorByMatrix(embedding, params.valueWeighs);
        value = util.addVectors(value, params.valueBiases);

        // Mindhárom vektort fejenként (head) további részekre hasítjuk:
        float[][] queries = util.splitVector(query, headCount);
        float[][] keys = util.splitVector(key, headCount);
        float[][] values = util.splitVector(value, headCount);

        // A key és value vektorokat eltároljuk (ezek elérhetők lesznek a következő tokenek feldolgozása során is)
        storedKeys.add(keys);
        storedValues.add(values);

        float[][] sums = new float[headCount][embeddingSize / headCount];

        // Fejenként (head) pontozzuk (kiszámítunk egy score értéket) a korábbi tokeneket az aktuális nézőpontjából (beleértve saját magát is).
        // (Tehát nem csak a korábbiakat, saját magát is pontozza a token,
        // ezért volt jó, hogy az aktuális token key és value értékét már korábban hozzáadtuk az eltároltakhoz.)
        for (int head = 0; head < headCount; head++)
        {
            // Számoljuk ki a score-t
            float[] scores = new float[storedKeys.size()];
            for (int pos = 0; pos < storedKeys.size(); pos++)
            {
                // A score az aktuális token query vektorának és a másik key vektorának szorzatával áll elő (dot product)
                scores[pos] = util.dotProduct(queries[head], storedKeys.get(pos)[head]) / attentionDividend;
            }

            // Softmax
            scores = util.softmax(scores);

            // A kapott score-t szorozzuk meg a value vektorokkal, és adjuk ezeket össze
            for (int pos = 0; pos < storedKeys.size(); pos++)
            {
                float[] sum = util.multiplyVectorByScalar(storedValues.get(pos)[head], scores[pos]);
                sums[head] = util.addVectors(sums[head], sum);
            }
        }

        // Fűzzük össze a fejenként (head) kapott értékeket tartalmazó mátrix sorait egyetlen hosszú vektorrá
        float[] flatSums = util.flattenMatrix(sums);

        // Szorozzuk meg az eredeményül kapott vektort a tanítás során előállt attention projection sújokkal...
        float[] result = util.multiplyVectorByMatrix(flatSums, params.projectionWeights);

        // ... és adjuk hozzá az eltolást (bias) is
        return util.addVectors(result, params.projectionBiases);
    }

    private float[] feedForward(float[] embedding)
    {
        // Egy nagyon egyszerű neurális hálózatunk van, amely csupán két rétegből áll:
        // - az első réteg az embedding-hossz négyszeresének megfelelő számú neuronból áll (gelu aktivációs függvény használatával)
        // - a második réteg az embedding-hosszal megegyező számú neuronból áll (aktivációs függvény nélkül)

        // Egy egyszerű előrecsatolt neuron-réteg eredményének kiszámítása megegyezik egy vektor és a súlyok mátrixának szorzatával,
        // majd az eltolás (bias) hozzáadásával

        // Első neuron-réteg
        float[] output = util.multiplyVectorByMatrix(embedding, params.feedForwardLayer1Weights);
        output = util.addVectors(output, params.feedForwardLayer1Biases);

        // Az első réteg esetén használjuk az aktivációs függvényt (gelu)
        for (int neuron = 0; neuron < 4 * embeddingSize; neuron++)
        {
            output[neuron] = gelu(output[neuron]);
        }

        // Második neuron-réteg (nincs aktivációs függvény-hívás)
        output = util.multiplyVectorByMatrix(output, params.feedForwardLayer2Weights);
        return util.addVectors(output, params.feedForwardLayer2Biases);
    }

    // Gaussian Error Linear Unit (GELU) aktivációs függvény (közelítő megvalósítás)
    private static float gelu(float value)
    {
        return (float) (0.5 * value * (1 + tanh(sqrt(2 / PI) * (value + 0.044715 * pow(value, 3)))));
    }

    /**
     * A dekóder által eltárolt értékek törlése
     */
    public void clear()
    {
        storedKeys.clear();
        storedValues.clear();
    }
}
