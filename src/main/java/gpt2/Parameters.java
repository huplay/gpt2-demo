package gpt2;

import java.io.File;
import java.io.FileInputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;

/**
 * Storage of parameters. Weights and biases (+ token and position embeddings) generated during the training.
 */
public class Parameters
{
    public final float[][] tokenEmbeddings;
    public final float[][] positionEmbeddings;

    public final float[] normFinalWeights;
    public final float[] normFinalBiases;

    public final DecoderParameters[] decoderParameters;

    public static class DecoderParameters
    {
        public float[][] queryWeighs;
        public float[] queryBiases;

        public float[][] keyWeighs;
        public float[] keyBiases;

        public float[][] valueWeighs;
        public float[] valueBiases;

        public float[][] projectionWeights;
        public float[] projectionBiases;

        public float[] norm1Weights;
        public float[] norm1Biases;

        public float[] norm2Weights;
        public float[] norm2Biases;

        public float[][] feedForwardLayer1Weights;
        public float[] feedForwardLayer1Biases;

        public float[][] feedForwardLayer2Weights;
        public float[] feedForwardLayer2Biases;
    }

    public Parameters(Config config)
    {
        ModelType modelType = config.modelType;
        int size = modelType.embeddingSize;
        String path = config.parametersPath + "/" + modelType.name() + "/";

        this.tokenEmbeddings = readMatrix(path + "wte.dat", modelType.tokenCount, size);
        this.positionEmbeddings = readMatrix(path + "wpe.dat", modelType.contextSize, size);

        this.normFinalWeights = readVector(path + "ln_f(g).dat", size);
        this.normFinalBiases = readVector(path + "ln_f(b).dat", size);

        int count = modelType.decoderCount;

        this.decoderParameters = new DecoderParameters[count];
        for (int i = 0; i < count; i++)
        {
            this.decoderParameters[i] = new DecoderParameters();
            String decoderPath = path + "decoder" + (i + 1) + "/";

            this.decoderParameters[i].queryWeighs = readMatrix(decoderPath + "attn.query(w).dat", size, size);
            this.decoderParameters[i].queryBiases = readVector(decoderPath + "attn.query(b).dat", size);

            this.decoderParameters[i].keyWeighs = readMatrix(decoderPath + "attn.key(w).dat", size, size);
            this.decoderParameters[i].keyBiases = readVector(decoderPath + "attn.key(b).dat", size);

            this.decoderParameters[i].valueWeighs = readMatrix(decoderPath + "attn.value(w).dat", size, size);
            this.decoderParameters[i].valueBiases = readVector(decoderPath + "attn.value(b).dat", size);

            this.decoderParameters[i].projectionWeights = readMatrix(decoderPath + "attn.c_proj(w).dat", size, size);
            this.decoderParameters[i].projectionBiases = readVector(decoderPath + "attn.c_proj(b).dat", size);

            this.decoderParameters[i].norm1Weights = readVector(decoderPath + "ln_1(g).dat", size);
            this.decoderParameters[i].norm1Biases = readVector(decoderPath + "ln_1(b).dat", size);

            this.decoderParameters[i].norm2Weights = readVector(decoderPath + "ln_2(g).dat", size);
            this.decoderParameters[i].norm2Biases = readVector(decoderPath + "ln_2(b).dat", size);

            this.decoderParameters[i].feedForwardLayer1Weights = readMatrix(decoderPath + "mlp.c_fc(w).dat", size, size * 4);
            this.decoderParameters[i].feedForwardLayer1Biases = readVector(decoderPath + "mlp.c_fc(b).dat", size * 4);

            this.decoderParameters[i].feedForwardLayer2Weights = readMatrix(decoderPath + "mlp.c_proj(w).dat", size * 4, size);
            this.decoderParameters[i].feedForwardLayer2Biases = readVector(decoderPath + "mlp.c_proj(b).dat", size);
        }
    }

    private float[] readVector(String fileName, int size)
    {
        return read(fileName, size);
    }

    private float[][] readMatrix(String fileName, int rows, int cols)
    {
        float[] numbers = read(fileName, rows * cols);
        return Util.splitVector(numbers, rows);
    }

    private float[] read(String fileName, int size)
    {
        File file = new File(fileName);

        if (file.exists())
        {
            if (file.length() != size * 4)
            {
                throw new RuntimeException("The size of the file (" + fileName + ", " + file.length() + ") is incorrect. Expected: " + size * 4);
            }

            return readFile(file);
        }
        else
        {
            // Handling files split into parts
            List<File> partFiles = findPartFiles(fileName);

            if ( ! partFiles.isEmpty())
            {
                float[][] parts = new float[partFiles.size()][size];

                // Read all the part files
                int i = 0;
                int sumSize = 0;
                for (File partFile : partFiles)
                {
                    parts[i] = readFile(partFile);
                    sumSize += parts[i].length;
                    i++;
                }

                if (sumSize != size)
                {
                    throw new RuntimeException("The sum size of the file parts (" + sumSize * 4 + ") is incorrect. Expected: " + (size * 4));
                }

                // Concatenate the parts into a single array
                float[] ret = new float[sumSize];

                int index = 0;
                for (float[] part : parts)
                {
                    for (float value : part)
                    {
                        ret[index] = value;
                        index++;
                    }
                }

                return ret;
            }
            else
            {
                throw new RuntimeException("Parameter file not found: " + fileName);
            }
        }
    }

    private List<File> findPartFiles(String fileName)
    {
        List<File> partFiles = new ArrayList<>();

        int i = 1;
        while (true)
        {
            File partFile = new File(fileName + ".part" + i);

            if (partFile.exists()) partFiles.add(partFile);
            else break;

            i++;
        }

        return partFiles;
    }

    private float[] readFile(File file)
    {
        int size = (int) (file.length() / 4);

        float[] array = new float[size];

        try (FileInputStream stream = new FileInputStream(file))
        {
            FileChannel inChannel = stream.getChannel();

            ByteBuffer buffer = inChannel.map(FileChannel.MapMode.READ_ONLY, 0, inChannel.size());

            buffer.order(ByteOrder.BIG_ENDIAN);
            FloatBuffer floatBuffer = buffer.asFloatBuffer();
            floatBuffer.get(array);
        }
        catch (Exception e)
        {
            throw new RuntimeException("Parameter file read error. (" + file.getName() + ")");
        }

        return array;
    }
}
