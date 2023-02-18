package ai.demo.gpt2;

import java.io.FileInputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;

public class Parameters
{
    public float[][] tokenEmbeddings;
    public float[][] positionEmbeddings;

    public float[] normFinalWeights;
    public float[] normFinalBiases;

    public DecoderParameters[] decoderParameters;

    public static class DecoderParameters
    {
        public float[][] attentionWeighs;
        public float[] attentionBiases;
        public float[][] attentionProjectionWeights;
        public float[] attentionProjectionBiases;

        public float[] norm1Weights;
        public float[] norm1Biases;

        public float[][] layer1Weights;
        public float[] layer1Biases;
        public float[][] layer2Weights;
        public float[] layer2Biases;

        public float[] norm2Weights;
        public float[] norm2Biases;
    }

    public Parameters(Config config)
    {
        ModelType modelType = config.modelType;
        int embeddingSize = modelType.embeddingSize;
        String path = config.parametersPath + "/" + modelType.name() + "/";

        try
        {
            this.tokenEmbeddings = read2D(path + "wte.dat", modelType.tokenCount, embeddingSize);
            this.positionEmbeddings = read2D(path + "wpe.dat", modelType.contextSize, embeddingSize);

            this.normFinalWeights = read1D(path + "ln_f(g).dat", embeddingSize);
            this.normFinalBiases = read1D(path + "ln_f(b).dat", embeddingSize);

            int count = modelType.decoderCount;

            this.decoderParameters = new DecoderParameters[count];
            for (int i = 0; i < count; i++)
            {
                this.decoderParameters[i] = new DecoderParameters();
                String decoderPath = path + "decoder" + (i + 1) + "/";

                this.decoderParameters[i].attentionWeighs = read2D(decoderPath + "attn.c_attn(w).dat", embeddingSize, 3 * embeddingSize);
                this.decoderParameters[i].attentionBiases = read1D(decoderPath + "attn.c_attn(b).dat", 3 * embeddingSize);
                this.decoderParameters[i].attentionProjectionWeights = read2D(decoderPath + "attn.c_proj(w).dat", embeddingSize, embeddingSize);
                this.decoderParameters[i].attentionProjectionBiases = read1D(decoderPath + "attn.c_proj(b).dat", embeddingSize);

                this.decoderParameters[i].norm1Weights = read1D(decoderPath + "ln_1(g).dat", embeddingSize);
                this.decoderParameters[i].norm1Biases = read1D(decoderPath + "ln_1(b).dat", embeddingSize);

                this.decoderParameters[i].layer1Weights = read2D(decoderPath + "mlp.c_fc(w).dat", embeddingSize, 4 * embeddingSize);
                this.decoderParameters[i].layer1Biases = read1D(decoderPath + "mlp.c_fc(b).dat", 4 * embeddingSize);
                this.decoderParameters[i].layer2Weights = read2D(decoderPath + "mlp.c_proj(w).dat", 4 * embeddingSize, embeddingSize);
                this.decoderParameters[i].layer2Biases = read1D(decoderPath + "mlp.c_proj(b).dat", embeddingSize);

                this.decoderParameters[i].norm2Weights = read1D(decoderPath + "ln_2(g).dat", embeddingSize);
                this.decoderParameters[i].norm2Biases = read1D(decoderPath + "ln_2(b).dat", embeddingSize);
            }
        }
        catch (Exception e)
        {
            System.out.println("Parameter file reading error. Path: " + path);
            System.exit(0);
        }
    }

    private static float[] read1D(String fileName, int size)
    {
        float[] array = new float[size];

        try (FileInputStream stream = new FileInputStream(fileName))
        {
            FileChannel inChannel = stream.getChannel();

            ByteBuffer buffer = inChannel.map(FileChannel.MapMode.READ_ONLY, 0, inChannel.size());

            buffer.order(ByteOrder.BIG_ENDIAN);
            FloatBuffer floatBuffer = buffer.asFloatBuffer();
            floatBuffer.get(array);
        }
        catch (Exception e)
        {
            throw new RuntimeException("Parameter file read error. (" + fileName + ")");
        }

        return array;
    }

    private static float[][] read2D(String fileName, int rows, int cols)
    {
        float[][] array = new float[rows][cols];

        try (FileInputStream stream = new FileInputStream(fileName))
        {
            FileChannel inChannel = stream.getChannel();

            ByteBuffer buffer = inChannel.map(FileChannel.MapMode.READ_ONLY, 0, inChannel.size());

            buffer.order(ByteOrder.BIG_ENDIAN);
            FloatBuffer floatBuffer = buffer.asFloatBuffer();

            for (int i = 0; i < rows; i++)
            {
                floatBuffer.get(array[i], 0, cols);
            }
        }
        catch (Exception e)
        {
            throw new RuntimeException("Parameter file read error. (" + fileName + ")");
        }

        return array;
    }
}
